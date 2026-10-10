"""Train an eye-level mixed-view model with cumulative ordinal DR supervision."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from active_retina.train_mixed_view_bag_autonomy import (
    EyeBags,
    MixedViewModel,
    write_predictions,
)
from active_retina.train_quality_gated_autonomy import (
    evaluate_policy,
    hard_boundary_rank_loss,
    masked_bce_with_logits,
    read_rows,
    select_policy,
)


class CumulativeOrdinalHead(nn.Module):
    """Four ordered logits for P(grade > k), k=0..3."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(width * 3, width), nn.GELU(), nn.Dropout(0.2), nn.Linear(width, 1)
        )
        self.base_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.raw_increments = nn.Parameter(torch.full((3,), 0.54132485))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        increments = nn.functional.softplus(self.raw_increments)
        cutpoints = torch.cat((self.base_cutpoint[None], self.base_cutpoint + increments.cumsum(0)))
        return self.score(features) - cutpoints.unsqueeze(0)


class CumulativeOrdinalModel(MixedViewModel):
    def __init__(self, model_name: str, image_size: int) -> None:
        super().__init__(model_name, pretrained=False, image_size=image_size)
        width = int(self.encoder.num_features)
        self.ordinal_head = CumulativeOrdinalHead(width)
        self.grade_head = nn.Identity()
        self.risk_head = nn.Identity()

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor):
        batch, views, channels, height, width = images.shape
        features = self.encoder(images.reshape(batch * views, channels, height, width)).reshape(batch, views, -1)
        mask = view_mask.unsqueeze(-1).to(features.dtype)
        mean = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        maximum = features.masked_fill(~view_mask.unsqueeze(-1), torch.finfo(features.dtype).min).max(dim=1).values
        paired = (view_mask[:, 1] & view_mask[:, 0]).unsqueeze(-1)
        difference = torch.abs(features[:, 0] - features[:, 1]) * paired
        fused = torch.cat((mean, maximum, difference), dim=1)
        ordinal_logits = self.ordinal_head(fused)
        risk_logits = ordinal_logits[:, 1]
        quality_logits = self.quality_head(fused).squeeze(1)
        return ordinal_logits, risk_logits, quality_logits


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    columns: dict[str, list[Any]] = {
        key: [] for key in (
            "grade", "referable", "quality_known", "ungradable", "risk", "quality",
            "eye_id", "patient_id", "source_dataset", "image_ids",
        )
    }
    with torch.inference_mode():
        for images, mask, grade, referable, quality_known, ungradable, eye_id, patient_id, source, image_ids in loader:
            _, risk_logits, quality_logits = model(images.to(device), mask.to(device))
            values = {
                "grade": grade.numpy(), "referable": referable.numpy(),
                "quality_known": quality_known.numpy(), "ungradable": ungradable.numpy(),
                "risk": risk_logits.float().sigmoid().cpu().numpy(),
                "quality": quality_logits.float().sigmoid().cpu().numpy(),
                "eye_id": eye_id, "patient_id": patient_id,
                "source_dataset": source, "image_ids": image_ids,
            }
            for key, value in values.items():
                columns[key].extend(value.tolist() if isinstance(value, np.ndarray) else list(value))
    return {key: np.asarray(value) for key, value in columns.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=8e-7)
    parser.add_argument("--head-lr", type=float, default=8e-5)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--rank-top-fraction", type=float, default=0.25)
    parser.add_argument("--grade2-rank-weight", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model_name = str(initial["model_name"])
    base = CumulativeOrdinalModel(model_name, args.image_size)
    source = initial["state_dict"]
    warm_state = {
        key: value for key, value in source.items()
        if key.startswith("encoder.") or key.startswith("quality_head.")
    }
    missing, unexpected = base.load_state_dict(warm_state, strict=False)
    if unexpected or any(not (key.startswith("ordinal_head.") or key.startswith("grade_head.") or key.startswith("risk_head.")) for key in missing):
        raise ValueError(f"Unexpected warm-start mismatch; missing={missing}, unexpected={unexpected}")
    model = base.to(device)

    root = Path.cwd()
    splits = read_rows(Path(args.manifest))
    train = EyeBags(splits["train"], model.train_transform, root)
    valid = EyeBags(splits["valid"], model.eval_transform, root)
    test = EyeBags(splits["test"], model.eval_transform, root)
    pin = device.type == "cuda"
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.backbone_lr},
            {"params": list(model.ordinal_head.parameters()) + list(model.quality_head.parameters()), "lr": args.head_lr},
        ],
        weight_decay=2e-4,
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
            images, mask = images.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            grade = grade.to(device)
            referable = referable.float().to(device)
            quality_known = quality_known.bool().to(device)
            ungradable = ungradable.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                ordinal_logits, risk_logits, quality_logits = model(images, mask)
                gradable = grade < 5
                ordinal_targets = (grade.unsqueeze(1) > torch.arange(4, device=device)).float()
                ordinal_loss = masked_bce_with_logits(ordinal_logits, ordinal_targets, gradable)
                boundary_loss = masked_bce_with_logits(risk_logits, referable, gradable)
                quality_loss = masked_bce_with_logits(quality_logits, ungradable, quality_known)
                rank_loss = hard_boundary_rank_loss(
                    risk_logits, grade, referable, gradable,
                    args.rank_margin, args.rank_top_fraction, args.grade2_rank_weight,
                )
                loss = ordinal_loss + args.boundary_weight * boundary_loss + quality_loss + args.rank_weight * rank_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(grade)
            n_examples += len(grade)

        valid_pred = predict(model, DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin), device)
        val_input = {key: valid_pred[key] for key in ("grade", "risk", "quality", "quality_known", "ungradable")}
        policy = select_policy(val_input)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, n_examples),
            "validation_policy": policy,
            "validation_metrics": evaluate_policy(val_input, policy),
        }
        history.append(record)
        key = (float(policy["specificity_gradable_nonreferable"]), -float(policy["referral_rate"]))
        if key > best_key:
            best_key, best_record = key, copy.deepcopy(record)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_name": model_name, "state_dict": best_state, "best_epoch": best_record,
                        "args": vars(args), "claim_boundary": "DEVELOPMENT_CANDIDATE_NOT_CERTIFIED"}, args.artifact)
        (out / "progress.json").write_text(json.dumps({"status": "TRAINING_IN_PROGRESS", "history": history}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True), flush=True)

    if best_state is None or best_record is None:
        raise RuntimeError("Training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    valid_pred, test_pred = predict(model, valid_loader, device), predict(model, test_loader, device)
    policy = select_policy({key: valid_pred[key] for key in ("grade", "risk", "quality", "quality_known", "ungradable")})
    valid_input = {key: valid_pred[key] for key in ("grade", "risk", "quality", "quality_known", "ungradable")}
    test_input = {key: test_pred[key] for key in ("grade", "risk", "quality", "quality_known", "ungradable")}
    write_predictions(out / "validation_predictions.csv", "valid", valid_pred, policy)
    write_predictions(out / "test_predictions_development_only.csv", "test", test_pred, policy)
    summary = {
        "status": "DEVELOPMENT_CANDIDATE_NOT_CERTIFIED",
        "claim_boundary": "Prior development has used this manifest; test results are development-only, not an untouched certificate.",
        "unit_of_analysis": "one DDR eye or paired DRTiD eye",
        "ordinal_objective": "cumulative BCE for grade>0, grade>1, grade>2, grade>3; extra binary and ranking weight at the grade-1/2 referral boundary",
        "manifest": args.manifest,
        "model_name": model_name,
        "best_epoch": best_record,
        "selected_validation_policy": policy,
        "validation_metrics": evaluate_policy(valid_input, policy),
        "test_metrics_development_only": evaluate_policy(test_input, policy),
        "train_eyes": len(train), "validation_eyes": len(valid), "test_eyes": len(test),
        "view_counts": {name: {"eyes": len(dataset), "one_view": sum(len(x["views"]) == 1 for x in dataset.items),
                               "two_view": sum(len(x["views"]) == 2 for x in dataset.items)}
                        for name, dataset in (("train", train), ("valid", valid), ("test", test))},
        "history": history,
        "artifact": args.artifact,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "progress.json").write_text(json.dumps({"status": "COMPLETE", "epochs": args.epochs}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
