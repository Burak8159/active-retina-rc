"""Train an MMRDR candidate using development rows only; never reads test labels."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .train_every_image_autonomy import (
    FundusDataset,
    GradeReferableModel,
    class_weights,
    hard_boundary_loss,
    metrics,
    pairwise_boundary_loss,
    predict,
    read_manifest,
    referable_logit,
    set_seed,
    set_trainable,
    select_zero_miss_threshold,
)


def predict_quality_gate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    grades: list[int] = []
    risks: list[float] = []
    quality_scores: list[float] = []
    ids: list[str] = []
    with torch.inference_mode():
        for images, grade, _, image_ids in loader:
            logits = model(images.to(device)).float()
            risk = torch.logsumexp(logits[:, 2:5], dim=1) - torch.logsumexp(logits[:, :2], dim=1)
            quality = logits[:, 5] - torch.logsumexp(logits[:, :5], dim=1)
            grades.extend(int(value) for value in grade.numpy())
            risks.extend(float(value) for value in risk.cpu().numpy())
            quality_scores.extend(float(value) for value in quality.cpu().numpy())
            ids.extend(str(value) for value in image_ids)
    return np.asarray(grades, dtype=int), np.asarray(risks), np.asarray(quality_scores), ids


def select_quality_gated_policy(
    grades: np.ndarray,
    risks: np.ndarray,
    quality_scores: np.ndarray,
    groups: list[str] | None = None,
) -> dict[str, Any]:
    if groups is not None:
        if len(groups) != len(grades):
            raise ValueError("Group IDs must align with image-level validation predictions")
        grouped: dict[str, list[int]] = {}
        for index, group in enumerate(groups):
            grouped.setdefault(group, []).append(index)
        grades = np.asarray([max(grades[ix] for ix in indices) for indices in grouped.values()], dtype=int)
        risks = np.asarray([max(risks[ix] for ix in indices) for indices in grouped.values()], dtype=float)
        quality_scores = np.asarray([max(quality_scores[ix] for ix in indices) for indices in grouped.values()], dtype=float)
    positive = (grades >= 2) & (grades <= 4)
    negative = (grades >= 0) & (grades <= 1)
    ungradable = grades == 5
    risk_threshold = float(risks[positive].min()) if positive.any() else float(risks.max()) + 1e-8
    quality_threshold = float(quality_scores[ungradable].min()) if ungradable.any() else float(quality_scores.max()) + 1e-8
    refer = (risks >= risk_threshold) | (quality_scores >= quality_threshold)
    true_negative = int((negative & ~refer).sum())
    false_negative = int((positive & ~refer).sum())
    missed_ungradable = int((ungradable & ~refer).sum())
    return {
        "risk_threshold": risk_threshold,
        "quality_threshold": quality_threshold,
        "n_gradable": int(positive.sum() + negative.sum()),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "ungradable": int(ungradable.sum()),
        "true_positive": int((positive & refer).sum()),
        "false_negative_clearance": false_negative,
        "true_negative": true_negative,
        "false_positive_referral": int((negative & refer).sum()),
        "missed_ungradable": missed_ungradable,
        "sensitivity": float(1 - false_negative / max(1, int(positive.sum()))),
        "specificity": float(true_negative / max(1, int(negative.sum()))),
        "action_coverage": 1.0,
    }


def source_patient_sampling_weights(rows: list[dict[str, str]]) -> list[float]:
    """Give each source equal probability, then each patient and their images equally."""
    if not rows:
        raise ValueError("Cannot balance an empty training manifest")
    def source_key(value: str) -> str:
        name = value.lower()
        for prefix, canonical in (
            ("deepdrid", "DeepDRiD"), ("drtid", "DRTiD"),
            ("idrid", "IDRiD"), ("mmrdr", "MMRDR"), ("ddr", "DDR"),
        ):
            if name.startswith(prefix):
                return canonical
        return value

    source_patient_counts: Counter[tuple[str, str]] = Counter(
        (source_key(row["source_dataset"]), row["patient_id"]) for row in rows
    )
    patients_by_source: dict[str, set[str]] = {}
    for source, patient in source_patient_counts:
        patients_by_source.setdefault(source, set()).add(patient)
    source_count = len(patients_by_source)
    return [
        1.0 / (source_count * len(patients_by_source[source_key(row["source_dataset"])])
               * source_patient_counts[(source_key(row["source_dataset"]), row["patient_id"])])
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="MMRDR development-only manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--model-name", default="resnet50.a1_in1k")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--trainable-name-fragments", default="fc")
    parser.add_argument("--train-all-parameters", action="store_true",
                        help="Unfreeze every parameter; overrides --trainable-name-fragments.")
    parser.add_argument("--binary-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=1.0)
    parser.add_argument("--hard-boundary-weight", type=float, default=0.0)
    parser.add_argument("--positive-margin", type=float, default=2.0)
    parser.add_argument("--negative-margin", type=float, default=1.0)
    parser.add_argument("--hard-positive-weight", type=float, default=10.0)
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--hard-grade2-weight", type=float, default=2.0)
    parser.add_argument("--hard-grade1-weight", type=float, default=1.5)
    parser.add_argument("--hard-top-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--quality-gated-ungradable", action="store_true",
                        help="Train DR risk only on grades 0-4 and route grade-5 quality through a separate score.")
    parser.add_argument("--source-patient-balanced", action="store_true",
                        help="Sample sources equally, then patients equally within source, then their images.")
    args = parser.parse_args()

    set_seed(args.seed)
    rows = read_manifest(args.manifest)
    if not rows or any(row["source_split"] not in {"train", "valid"} for row in rows):
        raise SystemExit("Refusing to train: input must contain development train/valid rows only, never test rows.")
    train_rows = [row for row in rows if row["source_split"] == "train"]
    valid_rows = [row for row in rows if row["source_split"] == "valid"]
    if not train_rows or not valid_rows:
        raise SystemExit(f"Both development splits required; train={len(train_rows)} valid={len(valid_rows)}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    wrapper = GradeReferableModel(args.model_name, args.image_size, pretrained=args.init_checkpoint is None)
    model = wrapper.model
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("model_name") != args.model_name:
            raise SystemExit("Initialization checkpoint backbone does not match --model-name.")
        model.load_state_dict(initial["state_dict"])
    fragments = [] if args.train_all_parameters else [part.strip() for part in args.trainable_name_fragments.split(",") if part.strip()]
    trainable = set_trainable(model, fragments)
    model.to(device)

    train_set = FundusDataset(train_rows, wrapper.train_transform, Path.cwd())
    valid_set = FundusDataset(valid_rows, wrapper.eval_transform, Path.cwd())
    sampler = None
    if args.source_patient_balanced:
        weights = torch.as_tensor(source_patient_sampling_weights(train_rows), dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, 1.0, 1.0, 1.0, "inverse").to(device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.use_amp and device.type == "cuda"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_quality_scores = np.zeros(len(valid_rows), dtype=float)
    if args.quality_gated_ungradable:
        initial_grades, initial_scores, initial_quality_scores, initial_ids = predict_quality_gate(model, valid_loader, device)
        initial_groups = [row["eye_id"] for row in valid_rows]
        initial_selected = select_quality_gated_policy(initial_grades, initial_scores, initial_quality_scores, initial_groups)
    else:
        initial_y, initial_grades, initial_scores, initial_ids = predict(model, valid_loader, device)
        initial_selected = select_zero_miss_threshold(initial_y, initial_scores)
    initial_key = (int(initial_selected["false_negative_clearance"] == 0), float(initial_selected["specificity"]), float(initial_selected["sensitivity"]))
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best: tuple[tuple[int, float, float], dict[str, Any], dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray, list[str]] | None = (
        initial_key, initial_selected, initial_state, initial_grades.copy(), initial_scores.copy(), initial_quality_scores.copy(), list(initial_ids)
    )
    history: list[dict[str, Any]] = [{"epoch": 0, "train_loss": None, "validation_zero_miss": initial_selected}]
    initial_checkpoint = {
        "model_name": args.model_name,
        "state_dict": initial_state,
        "threshold": float(initial_selected.get("risk_threshold", initial_selected.get("threshold"))),
        "validation_metrics": initial_selected,
        "best_epoch": 0,
        "trainable_info": trainable,
        "args": vars(args) | {"manifest": str(args.manifest), "output_dir": str(args.output_dir), "artifact": str(args.artifact)},
        "test_evaluated": False,
        "status": "DEVELOPMENT_CANDIDATE_TEST_NOT_READ",
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(initial_checkpoint, args.artifact)
    print(json.dumps({"epoch": 0, "train_loss": None, "valid": initial_selected}, sort_keys=True), flush=True)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        total = 0.0
        seen = 0
        for images, grades, labels, _ in train_loader:
            images, grades, labels = images.to(device), grades.to(device), labels.to(device).float()
            if args.quality_gated_ungradable:
                labels = ((grades >= 2) & (grades <= 4)).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.use_amp and device.type == "cuda")):
                logits = model(images)
                class_loss = criterion(logits, grades)
            with torch.amp.autocast("cuda", enabled=False):
                logits32 = logits.float()
                boundary = (
                    torch.logsumexp(logits32[:, 2:5], dim=1) - torch.logsumexp(logits32[:, :2], dim=1)
                    if args.quality_gated_ungradable else referable_logit(logits32)
                )
                binary_loss = nn.functional.binary_cross_entropy_with_logits(boundary, labels)
                rank_loss = pairwise_boundary_loss(boundary, labels, args.rank_margin)
                hard_loss = hard_boundary_loss(
                    boundary,
                    grades,
                    labels.long(),
                    args.positive_margin,
                    args.negative_margin,
                    args.hard_positive_weight,
                    args.hard_negative_weight,
                    args.hard_grade2_weight,
                    args.hard_grade1_weight,
                    args.hard_top_fraction,
                )
                loss = class_loss.float() + args.binary_weight * binary_loss + args.rank_weight * rank_loss + args.hard_boundary_weight * hard_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu()) * len(grades)
            seen += len(grades)

        if args.quality_gated_ungradable:
            grades, scores, quality_scores, ids = predict_quality_gate(model, valid_loader, device)
            selected = select_quality_gated_policy(grades, scores, quality_scores, [row["eye_id"] for row in valid_rows])
        else:
            y, grades, scores, ids = predict(model, valid_loader, device)
            quality_scores = np.zeros(len(grades), dtype=float)
            selected = select_zero_miss_threshold(y, scores)
        row = {"epoch": epoch, "train_loss": total / max(1, seen), "validation_zero_miss": selected}
        history.append(row)
        key = (int(selected["false_negative_clearance"] == 0), float(selected["specificity"]), float(selected["sensitivity"]))
        if best is None or key > best[0]:
            best = (key, selected, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, grades.copy(), scores.copy(), quality_scores.copy(), list(ids))
            stale = 0
            checkpoint = {
                "model_name": args.model_name,
                "state_dict": best[2],
                "threshold": float(selected.get("risk_threshold", selected.get("threshold"))),
                "quality_threshold": selected.get("quality_threshold"),
                "validation_metrics": selected,
                "best_epoch": epoch,
                "trainable_info": trainable,
                "args": vars(args) | {"manifest": str(args.manifest), "output_dir": str(args.output_dir), "artifact": str(args.artifact)},
                "test_evaluated": False,
                "status": "DEVELOPMENT_CANDIDATE_TEST_NOT_READ",
            }
            args.artifact.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, args.artifact)
        else:
            stale += 1
        (args.output_dir / "progress.json").write_text(json.dumps({"status": "TRAINING", "history": history, "test_evaluated": False}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)
        if stale >= args.patience:
            break

    if best is None:
        raise RuntimeError("No validation checkpoint was produced")
    key, val_metrics, state, val_grades, val_scores, val_quality_scores, val_ids = best
    model.load_state_dict(state)
    labels = np.asarray([int(row["referable_dr"]) for row in valid_rows], dtype=int)
    # Predict returns validation rows in manifest order; persist scores and the locked threshold.
    prediction_path = args.output_dir / "validation_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "grade", "referable_dr", "score", "quality_score", "action"])
        writer.writeheader()
        threshold = float(val_metrics.get("risk_threshold", val_metrics.get("threshold")))
        quality_threshold = val_metrics.get("quality_threshold")
        for image_id, grade, label, score, quality_score in zip(val_ids, val_grades, labels, val_scores, val_quality_scores):
            refer = float(score) >= threshold
            if args.quality_gated_ungradable:
                refer = refer or float(quality_score) >= float(quality_threshold)
            writer.writerow({"image_id": image_id, "grade": int(grade), "referable_dr": int(label), "score": f"{float(score):.8f}", "quality_score": f"{float(quality_score):.8f}", "action": "REFER" if refer else "CLEAR"})
    summary = {
        "status": "DEVELOPMENT_CANDIDATE_TEST_NOT_READ",
        "task": "gradable referable DR = grade 2-4; grade-5 images use a separate automatic quality/referral gate" if args.quality_gated_ungradable else "binary referable DR = grade >= 2; grade 0-1 negative",
        "dataset": " + ".join(sorted({row["source_dataset"] for row in train_rows})) + "; MMRDR-UWF initialization",
        "model_name": args.model_name,
        "best_epoch": int(torch.load(args.artifact, map_location="cpu", weights_only=False)["best_epoch"]),
        "quality_gated_ungradable": bool(args.quality_gated_ungradable),
        "validation_metrics_at_quality_gated_policy": val_metrics if args.quality_gated_ungradable else None,
        "validation_metrics_at_zero_miss_threshold": metrics(labels, val_scores, float(val_metrics["threshold"])) if not args.quality_gated_ungradable else None,
        "trainable_info": trainable,
        "checkpoint": str(args.artifact),
        "test_evaluated": False,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "progress.json").write_text(
        json.dumps({"status": "COMPLETED", "best_epoch": summary["best_epoch"], "test_evaluated": False, "history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
