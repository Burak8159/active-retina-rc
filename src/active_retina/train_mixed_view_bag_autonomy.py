"""Train a mixed single-/paired-view model at the annotated eye level.

DDR rows are single-image cases. DRTiD and DeepDRiD labels describe macula/optic-disc
pairs, so their images are grouped into one eye-level example rather than duplicated
as independent labeled cases. All metrics remain development-only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from active_retina.train_quality_gated_autonomy import (
    evaluate_policy,
    hard_boundary_rank_loss,
    masked_bce_with_logits,
    masked_cross_entropy,
    read_rows,
    select_policy,
)


class EyeBags(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: Any, root: Path) -> None:
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            eye_id = str(row["eye_id"])
            bag = grouped.setdefault(
                eye_id,
                {
                    "eye_id": eye_id,
                    "patient_id": str(row["patient_id"]),
                    "source_dataset": str(row["source_dataset"]),
                    "grade": int(row["grade"]),
                    "referable_dr": int(row["referable_dr"]),
                    "quality_known": int(row["source_dataset"] == "ddr_hf_specificity300"),
                    "ungradable": int(row["source_dataset"] == "ddr_hf_specificity300" and int(row["grade"]) == 5),
                    "views": [],
                },
            )
            for field in ("patient_id", "source_dataset", "grade", "referable_dr"):
                value = int(row[field]) if field in {"grade", "referable_dr"} else str(row[field])
                if bag[field] != value:
                    raise ValueError(f"Conflicting {field} for eye {eye_id}")
            bag["views"].append((str(row["field"]), str(row["image_path"]), str(row["image_id"])))

        self.items = []
        self.transform = transform
        self.root = root
        for bag in grouped.values():
            if bag["source_dataset"] in {"drtid_paired", "deepdrid_paired"}:
                fields = {view[0] for view in bag["views"]}
                if fields != {"macula", "optic_disc"}:
                    raise ValueError(f"Incomplete DRTiD pair {bag['eye_id']}: {sorted(fields)}")
                bag["views"].sort(key=lambda view: (view[0] != "macula", view[2]))
            elif len(bag["views"]) != 1:
                raise ValueError(f"Expected one DDR image for {bag['eye_id']}")
            self.items.append(bag)
        if not self.items:
            raise ValueError("No eye-level examples in split")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        tensors = []
        ids = []
        for _field, raw_path, image_id in item["views"]:
            path = Path(raw_path)
            if not path.exists():
                normalized = raw_path.replace("\\", "/")
                if "/external/" in normalized:
                    path = self.root / "external" / normalized.split("/external/", 1)[1]
                else:
                    path = self.root / path
            with Image.open(path) as image:
                tensors.append(self.transform(image.convert("RGB")))
            ids.append(image_id)
        mask = torch.zeros(2, dtype=torch.bool)
        mask[: len(tensors)] = True
        if len(tensors) == 1:
            tensors.append(tensors[0].clone())
            ids.append(ids[0])
        return (
            torch.stack(tensors),
            mask,
            int(item["grade"]),
            int(item["referable_dr"]),
            int(item["quality_known"]),
            int(item["ungradable"]),
            item["eye_id"],
            item["patient_id"],
            item["source_dataset"],
            "|".join(ids),
        )


class MixedViewModel(nn.Module):
    def __init__(
        self, model_name: str, pretrained: bool, image_size: int, preserve_pretrained_risk: bool = False
    ) -> None:
        super().__init__()
        import timm

        self.preserve_pretrained_risk = preserve_pretrained_risk
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1 if preserve_pretrained_risk else 0, global_pool="avg"
        )
        width = int(self.encoder.num_features)
        self.grade_head = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Dropout(0.2), nn.Linear(width, 5))
        self.risk_head = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Dropout(0.2), nn.Linear(width, 1))
        self.quality_head = nn.Sequential(nn.Linear(width * 3, width // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(width // 2, 1))
        if preserve_pretrained_risk:
            nn.init.zeros_(self.risk_head[-1].weight)
            nn.init.zeros_(self.risk_head[-1].bias)
        cfg = timm.data.resolve_model_data_config(self.encoder)
        cfg["input_size"] = (3, image_size, image_size)
        self.train_transform = timm.data.create_transform(**cfg, is_training=True)
        self.eval_transform = timm.data.create_transform(**cfg, is_training=False)

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor):
        batch, views, channels, height, width = images.shape
        flat_images = images.reshape(batch * views, channels, height, width)
        if self.preserve_pretrained_risk:
            encoded = self.encoder.forward_features(flat_images)
            flat_features = self.encoder.forward_head(encoded, pre_logits=True)
            pretrained_logits = self.encoder.forward_head(encoded).reshape(batch, views, -1).squeeze(-1)
        else:
            flat_features = self.encoder(flat_images)
            pretrained_logits = None
        features = flat_features.reshape(batch, views, -1)
        mask = view_mask.unsqueeze(-1).to(features.dtype)
        mean = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        maximum = features.masked_fill(~view_mask.unsqueeze(-1), torch.finfo(features.dtype).min).max(dim=1).values
        difference = torch.abs(features[:, 0] - features[:, 1]) * (view_mask[:, 1] & view_mask[:, 0]).unsqueeze(-1)
        fused = torch.cat((mean, maximum, difference), dim=1)
        risk = self.risk_head(fused).squeeze(1)
        if pretrained_logits is not None:
            prior_risk = aggregate_pretrained_view_risk(pretrained_logits, view_mask)
            risk = prior_risk + risk
        return self.grade_head(fused), risk, self.quality_head(fused).squeeze(1)


class BoundaryPairBatchSampler(Sampler[list[int]]):
    """Oversample within-source grade-1/grade-2 pairs without touching held-out data."""

    def __init__(self, items: list[dict[str, Any]], batches_per_epoch: int, seed: int) -> None:
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        groups: dict[tuple[str, int], list[int]] = {}
        for index, item in enumerate(items):
            groups.setdefault((item["source_dataset"], int(item["grade"])), []).append(index)
        self.boundary_sources = [
            source for source in sorted({key[0] for key in groups})
            if groups.get((source, 1)) and groups.get((source, 2))
        ]
        self.grade_zero = [idx for (source, grade), values in groups.items() if grade == 0 for idx in values]
        self.grade_severe = [idx for (source, grade), values in groups.items() if grade in (3, 4) for idx in values]
        self.ungradable = [idx for (source, grade), values in groups.items() if grade == 5 for idx in values]
        if not self.boundary_sources:
            raise ValueError("Boundary-pair sampling requires grade-1 and grade-2 eyes in at least one source")
        if not self.grade_zero or not self.grade_severe:
            raise ValueError("Sampler requires grade-0 and grade-3/4 training eyes")
        if not self.ungradable:
            raise ValueError("Sampler requires known ungradable training eyes for quality supervision")
        self.groups = groups

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for step in range(self.batches_per_epoch):
            slot = step % 10
            if slot == 8:
                # Retain a direct quality signal while the encoder is fine-tuned.
                yield [rng.choice(self.ungradable), rng.choice(self.grade_zero)]
            elif slot == 9:
                # Keep the ordinal head exposed to non-boundary grades.
                yield [rng.choice(self.grade_zero), rng.choice(self.grade_severe)]
            else:
                source = self.boundary_sources[(step // 2) % len(self.boundary_sources)]
                yield [rng.choice(self.groups[(source, 1)]), rng.choice(self.groups[(source, 2)])]

    def __len__(self) -> int:
        return self.batches_per_epoch


class BoundaryFocusedEyeBatchSampler(Sampler[list[int]]):
    """Stratify single-eye updates around grades 1/2 while retaining controls."""

    def __init__(self, items: list[dict[str, Any]], samples_per_epoch: int, seed: int) -> None:
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        groups: dict[tuple[str, int], list[int]] = {}
        for index, item in enumerate(items):
            groups.setdefault((item["source_dataset"], int(item["grade"])), []).append(index)
        self.boundary_sources = [
            source for source in sorted({key[0] for key in groups})
            if groups.get((source, 1)) and groups.get((source, 2))
        ]
        self.ungradable = [idx for (source, grade), values in groups.items() if grade == 5 for idx in values]
        self.controls = [idx for (source, grade), values in groups.items() if grade in (0, 3, 4) for idx in values]
        if not self.boundary_sources or not self.ungradable or not self.controls:
            raise ValueError("Focused sampling requires grade-1/2, ungradable, and grade-0/3/4 training eyes")
        self.groups = groups

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for step in range(self.samples_per_epoch):
            slot = step % 10
            if slot == 8:
                yield [rng.choice(self.ungradable)]
            elif slot == 9:
                yield [rng.choice(self.controls)]
            else:
                source = self.boundary_sources[(step // 4) % len(self.boundary_sources)]
                grade = 1 + (step % 2)
                yield [rng.choice(self.groups[(source, grade)])]

    def __len__(self) -> int:
        return self.samples_per_epoch


class SourceBalancedEyeBatchSampler(Sampler[list[int]]):
    """Equalize dataset exposure while preserving each source's observed grade mix."""

    def __init__(self, items: list[dict[str, Any]], batches_per_epoch: int, seed: int) -> None:
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        self.by_source: dict[str, list[int]] = {}
        self.by_source_grade: dict[tuple[str, int], list[int]] = {}
        for index, item in enumerate(items):
            source = str(item["source_dataset"])
            grade = int(item["grade"])
            self.by_source.setdefault(source, []).append(index)
            self.by_source_grade.setdefault((source, grade), []).append(index)
        self.sources = sorted(self.by_source)
        quality_negatives = [
            idx for (source, grade), group in self.by_source_grade.items()
            if source == "ddr_hf_specificity300" and grade < 5 for idx in group
        ]
        self.quality_pair = (self.by_source_grade.get(("ddr_hf_specificity300", 5), []), quality_negatives)
        if len(self.sources) < 2 or not self.quality_pair[0] or not self.quality_pair[1]:
            raise ValueError("Source balancing requires multiple datasets and labeled DDR quality examples")

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        bad_quality, good_quality = self.quality_pair
        for step in range(self.batches_per_epoch):
            if step % 10 == 9:
                yield [rng.choice(bad_quality), rng.choice(good_quality)]
            else:
                source = self.sources[(step // 3) % len(self.sources)]
                group = self.by_source[source]
                yield [rng.choice(group), rng.choice(group)]

    def __len__(self) -> int:
        return self.batches_per_epoch


def predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    columns: dict[str, list[Any]] = {
        key: []
        for key in ("grade", "referable", "quality_known", "ungradable", "risk", "ordinal_risk", "quality", "eye_id", "patient_id", "source_dataset", "image_ids")
    }
    with torch.inference_mode():
        for images, mask, grade, referable, quality_known, ungradable, eye_id, patient_id, source, image_ids in loader:
            grade_logits, risk, quality = model(images.to(device), mask.to(device))
            values = {
                "grade": grade.numpy(),
                "referable": referable.numpy(),
                "quality_known": quality_known.numpy(),
                "ungradable": ungradable.numpy(),
                "risk": torch.sigmoid(risk.float()).cpu().numpy(),
                "ordinal_risk": ordinal_referable_score(grade_logits).cpu().numpy(),
                "quality": torch.sigmoid(quality.float()).cpu().numpy(),
                "eye_id": eye_id,
                "patient_id": patient_id,
                "source_dataset": source,
                "image_ids": image_ids,
            }
            for key, value in values.items():
                columns[key].extend(value.tolist() if isinstance(value, np.ndarray) else list(value))
    return {key: np.asarray(value) for key, value in columns.items()}


def filter_prediction_sources(
    scores: dict[str, np.ndarray], sources: list[str] | None
) -> dict[str, np.ndarray]:
    if not sources:
        return scores
    mask = np.isin(scores["source_dataset"].astype(str), sources)
    if not mask.any():
        raise ValueError(f"No prediction rows match evaluation sources: {sources}")
    return {key: value[mask] for key, value in scores.items()}


def select_filtered_policy(
    scores: dict[str, np.ndarray], sources: list[str] | None, score_key: str = "risk"
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    filtered = filter_prediction_sources(scores, sources)
    policy = select_policy(
        {
            "grade": filtered["grade"],
            "risk": filtered[score_key],
            "quality": filtered["quality"],
            "quality_known": filtered["quality_known"],
            "ungradable": filtered["ungradable"],
        }
    )
    return filtered, policy


def ordinal_referable_score(grade_logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(grade_logits.float(), dim=1)[:, 2:5].sum(dim=1)


def aggregate_pretrained_view_risk(logits: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~view_mask, torch.finfo(logits.dtype).min).max(dim=1).values


def select_manifest_sources(
    rows: list[dict[str, Any]], sources: list[str] | None, split: str
) -> list[dict[str, Any]]:
    if not sources:
        return rows
    selected = [row for row in rows if str(row["source_dataset"]) in sources]
    if not selected:
        raise ValueError(f"No {split} manifest rows match sources: {sources}")
    return selected


def write_predictions(path: Path, split: str, scores: dict[str, np.ndarray], policy: dict[str, Any]) -> None:
    referred = (scores["risk"].astype(float) >= float(policy["risk_threshold"])) | (
        scores["quality"].astype(float) >= float(policy["quality_threshold"])
    )
    keys = ["eye_id", "patient_id", "source_dataset", "grade", "referable", "quality_known", "ungradable", "risk", "ordinal_risk", "quality"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", *keys, "image_ids", "refer_action"])
        writer.writeheader()
        for i in range(len(scores["grade"])):
            row = {key: scores[key][i].item() if isinstance(scores[key][i], np.generic) else scores[key][i] for key in keys}
            row.update(split=split, image_ids=scores["image_ids"][i], refer_action=int(referred[i]))
            writer.writerow(row)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--model-name", default="convnext_tiny.fb_in22k_ft_in1k")
    parser.add_argument("--preserve-pretrained-risk", action="store_true",
                        help="Use the pretrained scalar disease head as a risk prior and learn a zero-initialized residual.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--boundary-pair-batches", type=int, default=0,
                        help="Use source-matched grade-1/grade-2 batches; 0 keeps ordinary shuffled sampling.")
    parser.add_argument("--boundary-focused-samples", type=int, default=0,
                        help="Use stratified one-eye updates around grades 1/2; requires batch size 1.")
    parser.add_argument("--source-balanced-batches", type=int, default=0,
                        help="Equalize source exposure with natural within-source grades; requires batch size 2.")
    parser.add_argument("--evaluation-sources", nargs="+", default=None,
                        help="Restrict policy selection and aggregate metrics to these source_dataset values.")
    parser.add_argument("--training-sources", nargs="+", default=None,
                        help="Restrict training to these source_dataset values; held-out splits remain patient-disjoint.")
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--rank-top-fraction", type=float, default=0.25)
    parser.add_argument("--grade2-rank-weight", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    seed_all(args.seed)
    root = Path.cwd()
    splits = read_rows(Path(args.manifest))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # A warm start already supplies encoder weights; avoid a needless hub lookup.
    model = MixedViewModel(
        args.model_name, pretrained=not bool(args.init_checkpoint), image_size=args.image_size,
        preserve_pretrained_risk=args.preserve_pretrained_risk,
    ).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("model_name") != args.model_name:
            raise ValueError("Warm-start checkpoint backbone does not match --model-name")
        model.load_state_dict(initial["state_dict"])
    train_rows = select_manifest_sources(splits["train"], args.training_sources, "train")
    valid_rows = select_manifest_sources(splits["valid"], args.evaluation_sources, "validation")
    test_rows = select_manifest_sources(splits["test"], args.evaluation_sources, "test")
    train_data = EyeBags(train_rows, model.train_transform, root)
    valid_data = EyeBags(valid_rows, model.eval_transform, root)
    test_data = EyeBags(test_rows, model.eval_transform, root)
    pin = device.type == "cuda"
    sampler_modes = sum(bool(value) for value in (
        args.boundary_pair_batches, args.boundary_focused_samples, args.source_balanced_batches
    ))
    if sampler_modes > 1:
        raise ValueError("Choose only one explicit sampling mode")
    if args.source_balanced_batches:
        if args.batch_size != 2:
            raise ValueError("--source-balanced-batches requires --batch-size 2")
        sampler = SourceBalancedEyeBatchSampler(train_data.items, args.source_balanced_batches, args.seed)
        train_loader = DataLoader(train_data, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=pin)
    elif args.boundary_focused_samples:
        if args.batch_size != 1:
            raise ValueError("--boundary-focused-samples requires --batch-size 1")
        sampler = BoundaryFocusedEyeBatchSampler(train_data.items, args.boundary_focused_samples, args.seed)
        train_loader = DataLoader(train_data, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=pin)
    elif args.boundary_pair_batches:
        if args.batch_size != 2:
            raise ValueError("--boundary-pair-batches requires --batch-size 2")
        batch_sampler = BoundaryPairBatchSampler(train_data.items, args.boundary_pair_batches, args.seed)
        train_loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=pin)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.backbone_lr},
            {"params": list(model.grade_head.parameters()) + list(model.risk_head.parameters()) + list(model.quality_head.parameters()), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    best_state = None
    best_record = None
    best_key = (-1.0, -2.0)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_examples = 0.0, 0
        for images, mask, grade, referable, quality_known, ungradable, *_ in train_loader:
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            grade = grade.to(device)
            referable = referable.float().to(device)
            quality_known = quality_known.bool().to(device)
            ungradable = ungradable.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                grade_logits, risk_logits, quality_logits = model(images, mask)
                gradable = grade < 5
                grade_loss = masked_cross_entropy(grade_logits, grade, gradable)
                risk_loss = masked_bce_with_logits(risk_logits, referable, gradable)
                quality_loss = masked_bce_with_logits(quality_logits, ungradable, quality_known)
                rank_loss = hard_boundary_rank_loss(
                    risk_logits, grade, referable, gradable,
                    args.rank_margin, args.rank_top_fraction, args.grade2_rank_weight,
                )
                loss = grade_loss + risk_loss + quality_loss + args.rank_weight * rank_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(grade)
            n_examples += len(grade)

        valid_pred = predictions(model, valid_loader, device)
        policy_valid_pred, policy = select_filtered_policy(valid_pred, args.evaluation_sources)
        _, ordinal_policy = select_filtered_policy(valid_pred, args.evaluation_sources, "ordinal_risk")
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, n_examples),
            "validation_policy": policy,
            "validation_metrics": evaluate_policy(
                {"grade": policy_valid_pred["grade"], "risk": policy_valid_pred["risk"], "quality": policy_valid_pred["quality"],
                 "quality_known": policy_valid_pred["quality_known"], "ungradable": policy_valid_pred["ungradable"]}, policy
            ),
            "ordinal_validation_policy": ordinal_policy,
            "ordinal_validation_metrics": evaluate_policy(
                {"grade": policy_valid_pred["grade"], "risk": policy_valid_pred["ordinal_risk"], "quality": policy_valid_pred["quality"],
                 "quality_known": policy_valid_pred["quality_known"], "ungradable": policy_valid_pred["ungradable"]}, ordinal_policy
            ),
        }
        history.append(record)
        if args.boundary_pair_batches:
            record["training_sampler"] = "source_matched_grade1_grade2_with_quality_and_extreme_grade_batches"
            record["boundary_pair_batches"] = args.boundary_pair_batches
        elif args.boundary_focused_samples:
            record["training_sampler"] = "stratified_grade1_grade2_single_eye_updates_with_quality_and_extreme_grade_controls"
            record["boundary_focused_samples"] = args.boundary_focused_samples
        elif args.source_balanced_batches:
            record["training_sampler"] = "equal_source_exposure_natural_within_source_grades_with_quality_controls"
            record["source_balanced_batches"] = args.source_balanced_batches
        key = (float(policy["specificity_gradable_nonreferable"]), -float(policy["referral_rate"]))
        if key > best_key:
            best_key = key
            best_record = copy.deepcopy(record)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_name": args.model_name, "state_dict": best_state, "best_epoch": best_record,
                        "args": vars(args), "claim_boundary": "DEVELOPMENT_CANDIDATE_NOT_CERTIFIED"}, args.artifact)
        (out / "progress.json").write_text(json.dumps({"status": "TRAINING_IN_PROGRESS", "history": history}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True), flush=True)

    if best_state is None or best_record is None:
        raise RuntimeError("Training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    valid_pred = predictions(model, valid_loader, device)
    policy_valid_pred, policy = select_filtered_policy(valid_pred, args.evaluation_sources)
    _, ordinal_policy = select_filtered_policy(valid_pred, args.evaluation_sources, "ordinal_risk")
    test_pred = predictions(model, test_loader, device)
    policy_test_pred = filter_prediction_sources(test_pred, args.evaluation_sources)
    valid_input = {key: policy_valid_pred[key] for key in ("grade", "risk", "quality", "quality_known", "ungradable")}
    test_input = {key: policy_test_pred[key] for key in ("grade", "risk", "quality", "quality_known", "ungradable")}
    ordinal_valid_input = {key: policy_valid_pred[key] for key in ("grade", "quality", "quality_known", "ungradable")}
    ordinal_valid_input["risk"] = policy_valid_pred["ordinal_risk"]
    ordinal_test_input = {key: policy_test_pred[key] for key in ("grade", "quality", "quality_known", "ungradable")}
    ordinal_test_input["risk"] = policy_test_pred["ordinal_risk"]
    write_predictions(out / "validation_predictions.csv", "valid", valid_pred, policy)
    write_predictions(out / "test_predictions_development_only.csv", "test", test_pred, policy)
    summary = {
        "status": "DEVELOPMENT_CANDIDATE_NOT_CERTIFIED",
        "claim_boundary": "The mixed manifest has been used for prior development and is not an untouched certification cohort.",
        "unit_of_analysis": "eye-level bag; one DDR image or paired DRTiD/DeepDRiD macula+optic-disc images",
        "manifest": args.manifest,
        "model_name": args.model_name,
        "training_sources": args.training_sources,
        "evaluation_sources": args.evaluation_sources,
        "best_epoch": best_record,
        "selected_validation_policy": policy,
        "validation_metrics": evaluate_policy(valid_input, policy),
        "test_metrics_development_only": evaluate_policy(test_input, policy),
        "selected_ordinal_validation_policy": ordinal_policy,
        "ordinal_validation_metrics": evaluate_policy(ordinal_valid_input, ordinal_policy),
        "ordinal_test_metrics_development_only": evaluate_policy(ordinal_test_input, ordinal_policy),
        "train_eyes": len(train_data), "validation_eyes": len(valid_data), "test_eyes": len(test_data),
        "view_counts": {
            split: {"eyes": len(dataset), "one_view": sum(len(x["views"]) == 1 for x in dataset.items),
                   "two_view": sum(len(x["views"]) == 2 for x in dataset.items)}
            for split, dataset in (("train", train_data), ("valid", valid_data), ("test", test_data))
        },
        "quality_label_boundary": "DDR grade 5 is ungradable; DRTiD quality labels are unknown and excluded from quality loss.",
        "history": history,
        "artifact": args.artifact,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "progress.json").write_text(json.dumps({"status": "COMPLETE", "epochs": args.epochs}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
