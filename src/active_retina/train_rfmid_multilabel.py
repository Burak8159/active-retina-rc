"""Train and evaluate a multi-label fundus classifier on the RFMiD official splits.

The disease outputs and the dataset's overall-risk output are evaluated
separately. This is a dataset-specific development experiment, not a clinical
validation or a universal retinal-disease detector.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SPLIT_PATHS = {
    "train": ("Training_Set/Training_Set/Training", "Training_Set/Training_Set/RFMiD_Training_Labels.csv"),
    "valid": ("Evaluation_Set/Evaluation_Set/Validation", "Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv"),
    "test": ("Test_Set/Test_Set/Test", "Test_Set/Test_Set/RFMiD_Testing_Labels.csv"),
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_split(root: Path, split: str) -> tuple[list[dict], list[str]]:
    image_dir_rel, labels_rel = SPLIT_PATHS[split]
    label_path = root / labels_rel
    image_dir = root / image_dir_rel
    with label_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        if not columns or columns[0].strip().lower() != "id":
            raise ValueError(f"Unexpected RFMiD label schema in {label_path}: {columns[:4]}")
        target_cols = columns[1:]
        if not target_cols:
            raise ValueError("RFMiD label table has no target columns")
        rows = []
        for row in reader:
            raw_id = row[columns[0]].strip()
            image_id = str(int(raw_id))
            image_path = image_dir / f"{image_id}.png"
            if not image_path.is_file():
                candidates = list(image_dir.glob(f"{image_id}.*"))
                if len(candidates) != 1:
                    raise FileNotFoundError(f"Expected one image for RFMiD ID {image_id} in {image_dir}")
                image_path = candidates[0]
            target = []
            for col in target_cols:
                value = int(row[col])
                if value not in (0, 1):
                    raise ValueError(f"Non-binary RFMiD value {value} for ID={image_id}, label={col}")
                target.append(value)
            rows.append({"id": image_id, "image_path": image_path, "targets": target})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate image IDs in RFMiD {split} split")
    return rows, target_cols


class RFMiDImages(Dataset):
    def __init__(self, rows: list[dict], size: int, training: bool):
        self.rows = rows
        ops = [transforms.Resize((size, size))]
        if training:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06)]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            x = self.transform(image.convert("RGB"))
        return x, torch.tensor(row["targets"], dtype=torch.float32), row["id"]


def multilabel_metrics(y_true: np.ndarray, scores: np.ndarray, names: list[str],
                       train_positive_counts: np.ndarray | None = None) -> dict:
    def rank_auc(y: np.ndarray, score: np.ndarray) -> float | None:
        positives = int(y.sum()); negatives = len(y) - positives
        if not positives or not negatives:
            return None
        order = np.argsort(score, kind="mergesort")
        sorted_scores = score[order]
        ranks = np.empty(len(y), dtype=float)
        start = 0
        while start < len(y):
            stop = start + 1
            while stop < len(y) and sorted_scores[stop] == sorted_scores[start]:
                stop += 1
            ranks[order[start:stop]] = (start + 1 + stop) / 2.0
            start = stop
        positive_rank_sum = ranks[y == 1].sum()
        return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))

    def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
        positives = int(y.sum())
        if not positives:
            return None
        order = np.argsort(-score, kind="mergesort")
        sorted_scores = score[order]
        sorted_y = y[order]
        cumulative_tp = np.cumsum(sorted_y)
        cumulative_fp = np.cumsum(1 - sorted_y)
        group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(y) - 1]
        tp = cumulative_tp[group_ends]
        fp = cumulative_fp[group_ends]
        previous_tp = np.r_[0, tp[:-1]]
        return float(np.sum((tp - previous_tp) / positives * tp / (tp + fp)))

    per_label = {}
    auc_values, ap_values, supported_accuracy = [], [], []
    prediction = scores >= 0.5
    for j, name in enumerate(names):
        y = y_true[:, j].astype(int)
        p = prediction[:, j]
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        auc = rank_auc(y, scores[:, j])
        ap = average_precision(y, scores[:, j])
        train_positives = int(train_positive_counts[j]) if train_positive_counts is not None else None
        if auc is not None and (train_positives is None or train_positives > 0):
            auc_values.append(auc)
        if ap is not None and (train_positives is None or train_positives > 0):
            ap_values.append(ap)
        tp = int(np.sum((y == 1) & p)); tn = int(np.sum((y == 0) & ~p))
        fp = int(np.sum((y == 0) & p)); fn = int(np.sum((y == 1) & ~p))
        per_label[name] = {
            "n": int(len(y)), "positive": positives, "negative": negatives,
            "train_positive": train_positives,
            "accuracy_at_0_5": float(np.mean(y == p)), "sensitivity_at_0_5": tp / positives if positives else None,
            "specificity_at_0_5": tn / negatives if negatives else None,
            "auroc": auc, "average_precision": ap,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }
        if train_positives is None or train_positives > 0:
            supported_accuracy.append(float(np.mean(y == p)))
    return {
        "n_images": int(len(y_true)),
        "macro_auroc_over_train_supported_labels_with_both_eval_classes": float(np.mean(auc_values)) if auc_values else None,
        "macro_average_precision_over_train_supported_labels_with_eval_positives": float(np.mean(ap_values)) if ap_values else None,
        "mean_label_accuracy_at_0_5": float(np.mean(y_true == prediction)),
        "mean_accuracy_over_train_supported_labels": float(np.mean(supported_accuracy)) if supported_accuracy else None,
        "subset_accuracy_at_0_5": float(np.mean(np.all(y_true == prediction, axis=1))),
        "per_label": per_label,
    }


def collect(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval(); ys, scores, ids = [], [], []
    with torch.inference_mode():
        for x, y, image_ids in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            ys.append(y.numpy())
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())
            ids.extend(image_ids)
    return np.concatenate(ys), np.concatenate(scores), ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bounded L4 run")
    root = args.data_root.resolve()
    splits, names = {}, None
    for split in SPLIT_PATHS:
        splits[split], cols = read_split(root, split)
        if names is None:
            names = cols
        elif cols != names:
            raise ValueError("RFMiD target columns differ across official splits")
    assert names is not None
    for split, rows in splits.items():
        print(json.dumps({"split": split, "n": len(rows), "label_positive_counts":
                          {name: int(sum(row["targets"][j] for row in rows)) for j, name in enumerate(names)}}), flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True); args.artifact.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    weights = None if args.no_pretrained else models.ConvNeXt_Tiny_Weights.DEFAULT
    encoder = models.convnext_tiny(weights=weights)
    width = encoder.classifier[-1].in_features
    encoder.classifier[-1] = nn.Linear(width, len(names))
    model = encoder.to(device)

    train_y = np.asarray([r["targets"] for r in splits["train"]], dtype=np.float32)
    pos = train_y.sum(axis=0); neg = len(train_y) - pos
    trainable = (pos > 0) & (neg > 0)
    trainable_idx = np.flatnonzero(trainable)
    if not len(trainable_idx):
        raise ValueError("No labels have both positive and negative examples in the training split")
    pos_weight = np.sqrt(neg[trainable] / pos[trainable]).clip(1.0, 10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=2e-4)
    loaders = {
        "train": DataLoader(RFMiDImages(splits["train"], args.size, True), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0),
        "valid": DataLoader(RFMiDImages(splits["valid"], args.size, False), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0),
        "test": DataLoader(RFMiDImages(splits["test"], args.size, False), batch_size=args.batch_size,
                           shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0),
    }
    scaler = torch.amp.GradScaler("cuda"); best_loss = float("inf"); best_epoch = 0; stale = 0; history = []
    started = time.monotonic(); max_reached = False
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y, _ in loaders["train"]:
            if time.monotonic() - started >= args.max_seconds:
                max_reached = True; break
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(x)[:, trainable_idx], y[:, trainable_idx])
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            break
        valid_y, valid_scores, _ = collect(model, loaders["valid"], device)
        valid_metrics = multilabel_metrics(valid_y, valid_scores, names, pos)
        valid_logits = np.log(np.clip(valid_scores, 1e-7, 1-1e-7) / np.clip(1-valid_scores, 1e-7, 1))
        valid_loss = float(criterion(torch.tensor(valid_logits[:, trainable_idx], device=device),
                                     torch.tensor(valid_y[:, trainable_idx], device=device)).item())
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": valid_loss,
                  "valid_macro_auroc": valid_metrics["macro_auroc_over_train_supported_labels_with_both_eval_classes"],
                  "valid_macro_ap": valid_metrics["macro_average_precision_over_train_supported_labels_with_eval_positives"]}
        history.append(record); print(json.dumps(record), flush=True)
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss; best_epoch = epoch; stale = 0
            torch.save({"state_dict": model.state_dict(), "labels": names, "epoch": epoch,
                        "architecture": "torchvision.convnext_tiny", "input_size": args.size,
                        "trainable_label_indices": trainable_idx.tolist()}, args.artifact)
        else:
            stale += 1
        if stale >= 5 or max_reached:
            break

    checkpoint = torch.load(args.artifact, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_y, test_scores, test_ids = collect(model, loaders["test"], device)
    report = {
        "status": "RFMID_OFFICIAL_SPLIT_DEVELOPMENT_ONLY_NOT_CLINICAL_VALIDATION",
        "dataset": "RFMiD public multi-label fundus mirror; source dataset descriptor reports CC BY 4.0",
        "architecture": checkpoint["architecture"], "pretrained_imagenet": not args.no_pretrained,
        "input_size": args.size, "seed": args.seed, "best_epoch_by_validation_bce": best_epoch,
        "split_image_counts": {split: len(rows) for split, rows in splits.items()},
        "training_config": {"optimizer": "AdamW", "learning_rate": 2e-5,
                            "weight_decay": 2e-4, "positive_weight": "sqrt(negative/positive), clipped to [1,10]",
                            "loss": "multi-label BCEWithLogitsLoss on labels with both classes in train",
                            "early_stopping": "validation BCE, patience 5, min_delta 1e-5",
                            "augmentation": ["horizontal flip", "rotation +/-8 degrees",
                                             "brightness +/-0.08", "contrast +/-0.08",
                                             "saturation +/-0.06"]},
        "elapsed_seconds": time.monotonic() - started, "max_seconds": args.max_seconds,
        "image_ids_available": False, "patient_disjointness_verified": False,
        "split_caveat": "Official image-level train/validation/test partitions were used; patient-level separation could not be verified from the supplied label tables.",
        "target_semantics": {"Disease_Risk": "dataset overall-risk label; not a universal retinal-disease diagnosis",
                             "other_columns": "source-provided multi-label condition indicators"},
        "history": history,
        "training_label_support": {name: {"positive": int(pos[j]), "negative": int(neg[j]),
                                           "included_in_loss": bool(trainable[j])}
                                   for j, name in enumerate(names)},
        "test_metrics": multilabel_metrics(test_y, test_scores, names, pos),
    }
    report_path = args.outdir / "run_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": str(report_path), "best_epoch": best_epoch,
                      "test_macro_auroc": report["test_metrics"]["macro_auroc_over_train_supported_labels_with_both_eval_classes"],
                      "test_macro_ap": report["test_metrics"]["macro_average_precision_over_train_supported_labels_with_eval_positives"],
                      "test_mean_label_accuracy": report["test_metrics"]["mean_label_accuracy_at_0_5"]}), flush=True)


if __name__ == "__main__":
    main()
