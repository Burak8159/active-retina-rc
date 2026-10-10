"""Train a quality-gated DR referral model on the mixed public development set.

DDR grade 5 means ungradable, not a DR severity grade. The model therefore has
separate severity, referable-risk, and gradability outputs. The deployed action
is REFER when either risk is high or quality is inadequate; otherwise CLEAR.
All current-cohort results are development evidence, not certification.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


def read_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    out = {name: [] for name in ("train", "valid", "test")}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            split = str(row["source_split"]).lower().rsplit("_", 1)[-1]
            if split not in out:
                continue
            grade = int(row["grade"])
            dataset = str(row["source_dataset"])
            row["grade"] = grade
            row["referable_dr"] = int(row["referable_dr"])
            row["quality_known"] = int(dataset == "ddr_hf_specificity300")
            row["ungradable"] = int(row["quality_known"] and grade == 5)
            if grade == 5 and not row["quality_known"]:
                raise ValueError(f"Unexpected grade 5 outside DDR: {row['image_id']}")
            out[split].append(row)
    for split, rows in out.items():
        if not rows:
            raise ValueError(f"No {split} rows found")
        if any(int(r["grade"]) not in range(6) for r in rows):
            raise ValueError(f"Invalid grade in {split}")
    return out


def resolve_image(root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    normalized = raw.replace("\\", "/")
    if "/external/" in normalized:
        candidate = root / "external" / normalized.split("/external/", 1)[1]
        if candidate.exists():
            return candidate
    return path


class FundusRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: Any, root: Path) -> None:
        self.rows, self.transform, self.root = rows, transform, root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(resolve_image(self.root, row["image_path"])) as source:
            image = self.transform(source.convert("RGB"))
        return (
            image,
            int(row["grade"]),
            int(row["referable_dr"]),
            int(row["quality_known"]),
            int(row["ungradable"]),
            str(row["image_id"]),
        )


class QualityGatedModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, image_size: int) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        features = int(self.backbone.num_features)
        self.grade_head = nn.Linear(features, 5)
        self.risk_head = nn.Linear(features, 1)
        self.quality_head = nn.Linear(features, 1)
        cfg = timm.data.resolve_model_data_config(self.backbone)
        cfg["input_size"] = (3, image_size, image_size)
        self.train_transform = timm.data.create_transform(**cfg, is_training=True)
        self.eval_transform = timm.data.create_transform(**cfg, is_training=False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        return self.grade_head(features), self.risk_head(features).squeeze(1), self.quality_head(features).squeeze(1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    rows: dict[str, list[Any]] = {k: [] for k in ("grade", "risk", "quality", "quality_known", "ungradable", "image_id")}
    with torch.inference_mode():
        for images, grade, referable, quality_known, ungradable, image_ids in loader:
            _, risk, quality = model(images.to(device))
            rows["grade"].extend(grade.numpy().astype(int).tolist())
            rows["risk"].extend(torch.sigmoid(risk.float()).cpu().numpy().tolist())
            rows["quality"].extend(torch.sigmoid(quality.float()).cpu().numpy().tolist())
            rows["quality_known"].extend(quality_known.numpy().astype(int).tolist())
            rows["ungradable"].extend(ungradable.numpy().astype(int).tolist())
            rows["image_id"].extend(str(x) for x in image_ids)
    return {k: np.asarray(v) for k, v in rows.items()}


def apply_policy(pred: dict[str, np.ndarray], risk_threshold: float, quality_threshold: float) -> np.ndarray:
    return (pred["risk"].astype(float) >= risk_threshold) | (pred["quality"].astype(float) >= quality_threshold)


def select_policy(pred: dict[str, np.ndarray]) -> dict[str, Any]:
    grade = pred["grade"].astype(int)
    disease_positive = (grade >= 2) & (grade <= 4)
    ungradable = pred["ungradable"].astype(bool) & pred["quality_known"].astype(bool)
    negative = (grade <= 1)
    if not disease_positive.any() or not negative.any():
        raise ValueError("Validation needs both DR-positive and DR-negative cases")

    if ungradable.any():
        min_ungradable_quality = float(pred["quality"][ungradable].min())
        q_candidates = {min_ungradable_quality}
        q_candidates.update(
            float(value)
            for value in pred["quality"][disease_positive]
            if float(value) <= min_ungradable_quality
        )
    else:
        # No quality labels exist in this evaluation cohort; do not infer a gate
        # from an unsupervised quality score or require an absent grade-5 class.
        q_candidates = {2.0}
    best: dict[str, Any] | None = None
    for q_threshold in sorted(q_candidates):
        q_refer = pred["quality"].astype(float) >= q_threshold
        remaining_positive = disease_positive & ~q_refer
        risk_threshold = (
            float(pred["risk"][remaining_positive].min()) if remaining_positive.any() else float("inf")
        )
        referred = q_refer | (pred["risk"].astype(float) >= risk_threshold)
        missed_dr = int((disease_positive & ~referred).sum())
        missed_ungradable = int((ungradable & ~referred).sum())
        specificity = float((~referred[negative]).mean())
        candidate = {
            "risk_threshold": risk_threshold,
            "quality_threshold": q_threshold,
            "specificity_gradable_nonreferable": specificity,
            "missed_referable_dr": missed_dr,
            "missed_ungradable": missed_ungradable,
            "n_referable_dr": int(disease_positive.sum()),
            "n_ungradable": int(ungradable.sum()),
            "n_gradable_nonreferable": int(negative.sum()),
            "referral_rate": float(referred.mean()),
            "n": int(len(grade)),
        }
        key = (specificity, -candidate["referral_rate"])
        best_key = (best["specificity_gradable_nonreferable"], -best["referral_rate"]) if best else (-1.0, -2.0)
        if key > best_key:
            best = candidate
    if best is None:
        raise RuntimeError("No policy candidates")
    return best


def evaluate_policy(pred: dict[str, np.ndarray], policy: dict[str, Any]) -> dict[str, Any]:
    grade = pred["grade"].astype(int)
    disease_positive = (grade >= 2) & (grade <= 4)
    ungradable = pred["ungradable"].astype(bool) & pred["quality_known"].astype(bool)
    negative = grade <= 1
    referred = apply_policy(pred, float(policy["risk_threshold"]), float(policy["quality_threshold"]))
    return {
        "n": int(len(grade)),
        "gradable_nonreferable": int(negative.sum()),
        "gradable_referable_dr": int(disease_positive.sum()),
        "known_ungradable": int(ungradable.sum()),
        "false_referrals_on_gradable_nonreferable": int((referred & negative).sum()),
        "missed_referable_dr": int((~referred & disease_positive).sum()),
        "missed_known_ungradable": int((~referred & ungradable).sum()),
        "specificity_gradable_nonreferable": float((~referred[negative]).mean()) if negative.any() else None,
        "sensitivity_referable_dr": float(referred[disease_positive].mean()) if disease_positive.any() else None,
        "ungradable_routed_to_refer": float(referred[ungradable].mean()) if ungradable.any() else None,
        "case_action_coverage": 1.0,
        "referral_rate": float(referred.mean()),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[mask], targets[mask])


def hard_boundary_rank_loss(
    logits: torch.Tensor,
    grade: torch.Tensor,
    referable: torch.Tensor,
    gradable: torch.Tensor,
    margin: float,
    top_fraction: float,
    grade2_weight: float,
) -> torch.Tensor:
    scores = logits[gradable].float()
    grades = grade[gradable]
    labels = referable[gradable] > 0.5
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return logits.sum() * 0.0
    pair_losses = nn.functional.softplus(margin - (positive[:, None] - negative[None, :]))
    positive_grades = grades[labels]
    pair_weights = torch.where(
        positive_grades[:, None] == 2,
        torch.full_like(pair_losses, grade2_weight),
        torch.ones_like(pair_losses),
    )
    weighted = pair_losses * pair_weights
    count = max(1, int(np.ceil(weighted.numel() * top_fraction)))
    return torch.topk(weighted.flatten(), count, largest=True).values.mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--model-name", default="convnext_tiny.fb_in22k_ft_in1k")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--rank-top-fraction", type=float, default=0.25)
    parser.add_argument("--grade2-rank-weight", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    seed_everything(args.seed)
    root = Path.cwd()
    splits = read_rows(Path(args.manifest))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = QualityGatedModel(args.model_name, True, args.image_size).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["state_dict"])
    train_loader = DataLoader(FundusRows(splits["train"], model.train_transform, root), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(FundusRows(splits["valid"], model.eval_transform, root), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(FundusRows(splits["test"], model.eval_transform, root), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": list(model.grade_head.parameters()) + list(model.risk_head.parameters()) + list(model.quality_head.parameters()), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_key: tuple[float, float] = (-1.0, -2.0)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, examples = 0.0, 0
        for images, grade, referable, quality_known, ungradable, _ in train_loader:
            images = images.to(device, non_blocking=True)
            grade = grade.to(device)
            referable = referable.float().to(device)
            quality_known = quality_known.bool().to(device)
            ungradable = ungradable.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                grade_logits, risk_logits, quality_logits = model(images)
                gradable = grade < 5
                grade_loss = masked_cross_entropy(grade_logits, grade, gradable)
                risk_loss = masked_bce_with_logits(risk_logits, referable, gradable)
                quality_loss = masked_bce_with_logits(quality_logits, ungradable, quality_known)
                rank_loss = hard_boundary_rank_loss(
                    risk_logits,
                    grade,
                    referable,
                    gradable,
                    args.rank_margin,
                    args.rank_top_fraction,
                    args.grade2_rank_weight,
                )
                loss = grade_loss + risk_loss + quality_loss + args.rank_weight * rank_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(grade)
            examples += len(grade)

        valid_pred = predict(model, valid_loader, device)
        selected = select_policy(valid_pred)
        row = {"epoch": epoch, "train_loss": total_loss / max(examples, 1), "validation_policy": selected}
        history.append(row)
        key = (float(selected["specificity_gradable_nonreferable"]), -float(selected["referral_rate"]))
        if key > best_key:
            best_key = key
            best_epoch = copy.deepcopy(row)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            artifact = Path(args.artifact)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_name": args.model_name, "state_dict": best_state, "best_epoch": best_epoch, "args": vars(args), "claim_boundary": "DEVELOPMENT_CANDIDATE_NOT_CERTIFIED"}, artifact)
        status = {
            "status": "TRAINING_IN_PROGRESS",
            "epoch": epoch,
            "epochs_planned": args.epochs,
            "best_epoch": best_epoch,
            "history": history,
            "args": vars(args),
            "test_evaluated": False,
        }
        (output_dir / "progress.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)

    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    model.load_state_dict(best_state)
    valid_pred = predict(model, valid_loader, device)
    policy = select_policy(valid_pred)
    valid_metrics = evaluate_policy(valid_pred, policy)
    test_pred = predict(model, test_loader, device)
    test_metrics = evaluate_policy(test_pred, policy)
    artifact = Path(args.artifact)
    torch.save({"model_name": args.model_name, "state_dict": model.cpu().state_dict(), "policy": policy, "best_epoch": best_epoch, "args": vars(args), "claim_boundary": "DEVELOPMENT_CANDIDATE_NOT_CERTIFIED"}, artifact)
    summary = {
        "status": "DEVELOPMENT_CANDIDATE_NOT_CERTIFIED",
        "reason": "Current public development partitions have been inspected during prior model work; no independent certification claim is made.",
        "task": "binary referral action: REFER for referable DR or predicted ungradable; otherwise CLEAR",
        "manifest": args.manifest,
        "model_name": args.model_name,
        "artifact": args.artifact,
        "best_epoch": best_epoch,
        "selected_validation_policy": policy,
        "validation_metrics": valid_metrics,
        "test_metrics_development_only": test_metrics,
        "history": history,
        "n_train": len(splits["train"]),
        "n_valid": len(splits["valid"]),
        "n_test": len(splits["test"]),
        "quality_supervision": "DDR grade 5 is treated as ungradable; DRTiD quality labels are unknown and do not train the quality head.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "progress.json").write_text(json.dumps({"status": "COMPLETE", "epoch": args.epochs, "test_evaluated": True, "best_epoch": best_epoch}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
