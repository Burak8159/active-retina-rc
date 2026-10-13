"""Resource-adjusted, localization-aware paired-field DRTiD development model.

The official DRTiD test partition is excluded. Only official training eyes are
split by patient into development train/validation. This is not a certification
or exact reproduction of the published CrossFiT implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .train_drtid_paired_fusion import (
    FIELDS,
    hard_boundary_rank_loss,
    ordinal_loss,
    patient_split,
    read_pairs,
    referable_score,
    safe_metrics,
)


MEAN = torch.tensor([0.372487, 0.217266, 0.119367]).view(3, 1, 1)
STD = torch.tensor([0.281526, 0.179457, 0.109162]).view(3, 1, 1)


def read_optic_disc_centers(path: Path) -> dict[str, tuple[float, float]]:
    centers = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = Path(row["image"]).stem
            point = (float(row["opticdisc_x"]), float(row["opticdisc_y"]))
            if name in centers:
                raise ValueError(f"Duplicate DRTiD localization row: {name}")
            if not all(np.isfinite(point)) or not all(0 <= value <= 1 for value in point):
                raise ValueError(f"Invalid optic-disc localization for {name}: {point}")
            centers[name] = point
    return centers


class LocalizedPairDataset(Dataset):
    def __init__(
        self,
        pairs: list[dict[str, Any]],
        centers: dict[str, tuple[float, float]],
        image_size: int,
        augment_flips: bool = False,
    ):
        self.pairs = pairs
        self.centers = centers
        self.image_size = image_size
        self.augment_flips = augment_flips
        for pair in pairs:
            for field in FIELDS:
                name = Path(pair["views"][field]).stem
                if name not in centers:
                    raise KeyError(f"Missing DRTiD optic-disc localization for {name}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        item = self.pairs[index]
        views = []
        coordinates = []
        for field in FIELDS:
            path = item["views"][field]
            with Image.open(path) as image:
                image = image.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            views.append((tensor - MEAN) / STD)
            coordinates.append(self.centers[Path(path).stem])
        stacked_views = torch.stack(views)
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
        if self.augment_flips:
            flip_x, flip_y = bool(torch.rand(()) < 0.5), bool(torch.rand(()) < 0.5)
            if flip_x:
                stacked_views = torch.flip(stacked_views, dims=(-1,))
                coordinates[:, 0] = 1.0 - coordinates[:, 0]
            if flip_y:
                stacked_views = torch.flip(stacked_views, dims=(-2,))
                coordinates[:, 1] = 1.0 - coordinates[:, 1]
        return (
            stacked_views,
            coordinates,
            int(item["grade"]),
            int(item["referable_dr"]),
            index,
        )


def sincos_2d(coords: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("Positional embedding width must be divisible by four")
    quarter = dim // 4
    omega = torch.arange(quarter, device=coords.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter, 1)))
    phases_x = coords[..., 0:1] * omega
    phases_y = coords[..., 1:2] * omega
    return torch.cat(
        [phases_x.sin(), phases_x.cos(), phases_y.sin(), phases_y.cos()], dim=-1
    )


def aligned_field_positions(
    base: torch.Tensor, centers: torch.Tensor, grid_h: int, grid_w: int
) -> torch.Tensor:
    """Align field-0 positions to field-1 using [0,1]-normalized landmarks."""
    batch = centers.shape[0]
    delta = centers[:, 1] - centers[:, 0]
    first = base.expand(batch, -1, -1).clone()
    first[..., 0] += delta[:, 0:1] * grid_w
    first[..., 1] += delta[:, 1:2] * grid_h
    second = base.expand(batch, -1, -1)
    return torch.stack([first, second], dim=1)


class LocalizedCrossFieldModel(nn.Module):
    def __init__(self, hidden: int = 512, layers: int = 3, mask_threshold: float = 0.06):
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            "resnet50.a1_in1k", pretrained=False, num_classes=0, global_pool=""
        )
        self.reduce = nn.Conv2d(self.encoder.num_features, hidden, kernel_size=1, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8 if hidden % 8 == 0 else 4,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_field = nn.TransformerEncoder(layer, num_layers=layers)
        self.view_embedding = nn.Parameter(torch.zeros(1, 2, 1, hidden))
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(0.1), nn.Linear(hidden, 5)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.mask_threshold = mask_threshold
        self.hidden = hidden
        self.register_buffer("grade_weight", torch.zeros(5))
        self.view_heads = nn.ModuleList(nn.Linear(self.encoder.num_features, 5) for _ in FIELDS)
        self.encoder_trainable = False

    def load_kaggle_state(self, checkpoint: Path) -> dict[str, Any]:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {key.removeprefix("module."): value for key, value in raw["net"].items()}
        fc_weight, fc_bias = state.pop("fc.weight"), state.pop("fc.bias")
        result = self.encoder.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"Kaggle encoder tensors did not map exactly: {result}")
        with torch.no_grad():
            for head in self.view_heads:
                head.weight.copy_(fc_weight)
                head.bias.copy_(fc_bias)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.view_heads.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        return {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_epoch": raw.get("epoch"),
            "source_kappa": raw.get("kappa"),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.view_heads.eval()
        return self

    def forward(self, images: torch.Tensor, centers: torch.Tensor):
        batch, views, channels, height, width = images.shape
        flat = images.reshape(batch * views, channels, height, width)
        with torch.set_grad_enabled(self.encoder_trainable):
            maps = self.encoder.forward_features(flat)
            pooled = maps.mean(dim=(2, 3))
            grade_logits = torch.stack(
                [head(pooled.reshape(batch, views, -1)[:, i]) for i, head in enumerate(self.view_heads)],
                dim=1,
            )
        _, channels_out, grid_h, grid_w = maps.shape
        raw_energy = maps.mean(dim=1).flatten(1)
        low = raw_energy.amin(dim=1, keepdim=True)
        high = raw_energy.amax(dim=1, keepdim=True)
        normalized_energy = (raw_energy - low) / (high - low).clamp_min(1e-6)
        background = normalized_energy < self.mask_threshold
        empty = background.all(dim=1)
        if empty.any():
            strongest = normalized_energy.argmax(dim=1)
            background[empty, strongest[empty]] = False

        tokens = self.reduce(maps).flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch, views, grid_h * grid_w, self.hidden)
        yy, xx = torch.meshgrid(
            torch.arange(grid_h, device=images.device, dtype=torch.float32),
            torch.arange(grid_w, device=images.device, dtype=torch.float32),
            indexing="ij",
        )
        base = torch.stack([xx, yy], dim=-1).reshape(1, grid_h * grid_w, 2)
        centers = centers.to(device=images.device, dtype=torch.float32)
        field_positions = aligned_field_positions(base, centers, grid_h, grid_w)
        position = sincos_2d(field_positions, self.hidden)
        tokens = tokens + position + self.view_embedding
        tokens = tokens.reshape(batch, views * grid_h * grid_w, self.hidden)
        key_padding = background.reshape(batch, views * grid_h * grid_w)
        cross = self.cross_field(tokens, src_key_padding_mask=key_padding)
        cross = cross.reshape(batch, views, grid_h * grid_w, self.hidden)
        per_view = cross.mean(dim=2)
        paired = per_view.max(dim=1).values
        residual_logits = self.residual(paired)
        fused_logits = grade_logits.mean(dim=1) + residual_logits
        return fused_logits, grade_logits, residual_logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--localization-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--unfreeze-layer4", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-6)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=1.0)
    parser.add_argument("--hard-top-fraction", type=float, default=0.25)
    parser.add_argument("--grade2-rank-weight", type=float, default=2.0)
    parser.add_argument("--view-rank-weight", type=float, default=0.5)
    parser.add_argument("--boundary-rank-weight", type=float, default=4.0)
    parser.add_argument("--boundary-rank-margin", type=float, default=2.0)
    parser.add_argument("--augment-flips", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest, localization = Path(args.manifest), Path(args.localization_csv)
    checkpoint = Path(args.checkpoint)
    pairs = read_pairs(manifest, Path.cwd())
    train_pairs, valid_pairs = patient_split(pairs, 0.2, args.seed)
    center_map = read_optic_disc_centers(localization)
    train_ds = LocalizedPairDataset(train_pairs, center_map, args.image_size, args.augment_flips)
    valid_ds = LocalizedPairDataset(valid_pairs, center_map, args.image_size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    model = LocalizedCrossFieldModel(args.hidden, args.layers).to(args.device)
    checkpoint_info = model.load_kaggle_state(checkpoint)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["state_dict"], strict=True)
    if args.unfreeze_layer4:
        for name, parameter in model.encoder.named_parameters():
            if name.startswith("layer4."):
                parameter.requires_grad = True
        model.encoder_trainable = True
    model.to(args.device)
    encoder_parameters = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    head_parameters = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("encoder.")
    ]
    parameter_groups = [{"params": head_parameters, "lr": args.lr}]
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=str(args.device).startswith("cuda"))
    criterion = nn.CrossEntropyLoss()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_specificity = -1.0
    best_state = None
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for images, centers, grade, _label, _index in train_loader:
            images = images.to(args.device, non_blocking=True)
            centers = centers.to(args.device, non_blocking=True)
            grade = grade.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if str(args.device).startswith("cuda") else "cpu",
                dtype=torch.float16,
                enabled=str(args.device).startswith("cuda"),
            ):
                fused, _views, residual = model(images, centers)
                loss = criterion(fused.float(), grade)
            loss = loss + 0.5 * ordinal_loss(fused.float(), grade)
            fused_rank = hard_boundary_rank_loss(
                fused.float(), grade, margin=args.rank_margin,
                top_fraction=args.hard_top_fraction, grade2_weight=args.grade2_rank_weight,
            )
            grade1_negative = grade == 1
            grade2_positive = grade == 2
            scores = referable_score(fused.float())
            if grade1_negative.any() and grade2_positive.any():
                adjacent = nn.functional.softplus(
                    args.boundary_rank_margin
                    - (scores[grade2_positive, None] - scores[None, grade1_negative])
                ).mean()
            else:
                adjacent = scores.new_tensor(0.0)
            view_rank = torch.stack([
                hard_boundary_rank_loss(
                    _views[:, view].float(), grade, margin=args.rank_margin,
                    top_fraction=args.hard_top_fraction, grade2_weight=args.grade2_rank_weight,
                )
                for view in range(2)
            ]).mean()
            loss = (
                loss
                + args.rank_weight * fused_rank
                + args.view_rank_weight * view_rank
                + args.boundary_rank_weight * adjacent
            )
            loss = loss + 1e-4 * residual.float().square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(grade)
            count += len(grade)

        model.eval()
        scores = np.empty(len(valid_pairs), dtype=np.float64)
        labels = np.empty(len(valid_pairs), dtype=np.int64)
        grades = np.empty(len(valid_pairs), dtype=np.int64)
        with torch.inference_mode():
            for images, centers, grade, label, indices in valid_loader:
                logits, _view_logits, _residual = model(
                    images.to(args.device, non_blocking=True),
                    centers.to(args.device, non_blocking=True),
                )
                idx = indices.numpy()
                scores[idx] = referable_score(logits).float().cpu().numpy()
                labels[idx] = label.numpy()
                grades[idx] = grade.numpy()
        metrics = safe_metrics(labels, scores)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, count),
            "validation": metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        specificity = float(metrics.get("specificity_at_zero_miss", -1.0))
        if specificity > best_specificity:
            best_specificity, best_epoch = specificity, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({
                "state_dict": best_state,
                "epoch": epoch,
                "image_size": args.image_size,
                "hidden": args.hidden,
                "layers": args.layers,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "localization_sha256": hashlib.sha256(localization.read_bytes()).hexdigest(),
                "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
                "train_patient_ids": sorted({str(row["patient_id"]) for row in train_pairs}),
                "validation_patient_ids": sorted({str(row["patient_id"]) for row in valid_pairs}),
                "loss_config": {
                    "ordinal_weight": 0.5,
                    "rank_weight": args.rank_weight,
                    "rank_margin": args.rank_margin,
                    "hard_top_fraction": args.hard_top_fraction,
                    "grade2_rank_weight": args.grade2_rank_weight,
                    "view_rank_weight": args.view_rank_weight,
                    "boundary_rank_weight": args.boundary_rank_weight,
                    "boundary_rank_margin": args.boundary_rank_margin,
                },
                "training_args": vars(args),
                "train_flip_augmentation": args.augment_flips,
                "encoder_trainable": model.encoder_trainable,
                "encoder_trainable_parameters": sum(
                    p.numel() for p in model.encoder.parameters() if p.requires_grad
                ),
                "status": "DEVELOPMENT_CHECKPOINT_NOT_CERTIFIED",
            }, output / "best_localized_crossfit.pt")
        (output / "progress.json").write_text(
            json.dumps({"status": "IN_PROGRESS", "history": history}, indent=2) + "\n",
            encoding="utf-8",
        )

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        for images, centers, grade, label, indices in valid_loader:
            logits, _view_logits, _residual = model(
                images.to(args.device, non_blocking=True), centers.to(args.device, non_blocking=True)
            )
            idx = indices.numpy()
            scores[idx] = referable_score(logits).float().cpu().numpy()
            labels[idx] = label.numpy()
            grades[idx] = grade.numpy()
    summary = {
        "status": "DEVELOPMENT_ONLY_LOCALIZED_CROSS_FIELD_EXPERIMENT",
        "boundary": "Patient-disjoint internal holdout from the current combined development manifest. The manifest was forcibly re-split across original source partitions, including previously inspected data; this run is development-only, not official-split or confirmatory evidence.",
        "n_train": len(train_pairs),
        "n_validation": len(valid_pairs),
        "train_patients": len({str(row["patient_id"]) for row in train_pairs}),
        "validation_patients": len({str(row["patient_id"]) for row in valid_pairs}),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "localization_sha256": hashlib.sha256(localization.read_bytes()).hexdigest(),
        "kaggle_checkpoint": checkpoint_info,
        "init_checkpoint": (
            {
                "path": str(Path(args.init_checkpoint)),
                "sha256": hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
            }
            if args.init_checkpoint else None
        ),
        "image_size": args.image_size,
        "hidden": args.hidden,
        "layers": args.layers,
        "loss_config": {
            "ordinal_weight": 0.5,
            "rank_weight": args.rank_weight,
            "rank_margin": args.rank_margin,
            "hard_top_fraction": args.hard_top_fraction,
            "grade2_rank_weight": args.grade2_rank_weight,
            "view_rank_weight": args.view_rank_weight,
            "boundary_rank_weight": args.boundary_rank_weight,
            "boundary_rank_margin": args.boundary_rank_margin,
        },
        "train_flip_augmentation": args.augment_flips,
        "encoder_trainable": model.encoder_trainable,
        "encoder_trainable_parameters": sum(
            p.numel() for p in model.encoder.parameters() if p.requires_grad
        ),
        "best_epoch": best_epoch,
        "best_zero_miss_specificity": best_specificity,
        "best_metrics": safe_metrics(labels, scores),
        "history": history,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["eye_id", "grade", "referable_dr", "score"])
        for i, pair in enumerate(valid_pairs):
            writer.writerow([pair["eye_id"], int(grades[i]), int(labels[i]), float(scores[i])])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
