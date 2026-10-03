"""Fine-tune a DDR referable-DR model and evaluate autonomous clearance.

The protocol is intentionally simple and fail-closed:

1. train on DDR train only,
2. select the zero-miss threshold on DDR validation only,
3. evaluate once on DDR test,
4. report both missed-risk certification and specificity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .formal_risk_control import binomial_upper


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


class FundusDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: Any) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(row["image_path"]) as img:
            image = img.convert("RGB")
        return (
            self.transform(image),
            torch.tensor(float(row["referable_dr"]), dtype=torch.float32),
            row["image_id"],
        )


def device_for(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def build_model(model_name: str):
    import timm

    model = timm.create_model(model_name, pretrained=True, num_classes=1)
    cfg = timm.data.resolve_model_data_config(model)
    train_transform = timm.data.create_transform(**cfg, is_training=True)
    eval_transform = timm.data.create_transform(**cfg, is_training=False)
    return model, train_transform, eval_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {split: [r for r in rows if r["source_split"] == f"ddr_{split}"] for split in ("train", "valid", "test")}


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    scores: list[float] = []
    labels: list[float] = []
    ids: list[str] = []
    with torch.inference_mode():
        for images, y, image_ids in loader:
            logits = model(images.to(device)).flatten()
            probs = torch.sigmoid(logits).detach().cpu().float().numpy()
            scores.extend(float(x) for x in probs)
            labels.extend(float(x) for x in y.numpy())
            ids.extend(str(x) for x in image_ids)
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), ids


def eval_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    decided = scores >= threshold
    decided_pos = int(((y == 1) & decided).sum())
    missed = int(((y == 1) & (~decided)).sum())
    false_pos = int(((y == 0) & decided).sum())
    true_neg = int(((y == 0) & (~decided)).sum())
    ucb = binomial_upper(missed, decided_pos, 0.05) if decided_pos else float("nan")
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "decided": int(decided.sum()),
        "decided_positive": decided_pos,
        "missed_positive": missed,
        "false_positive": false_pos,
        "true_negative": true_neg,
        "specificity": float(true_neg / max(1, int((y == 0).sum()))),
        "referable_action_rate": float(decided.mean()),
        "precision_referable_action": float(decided_pos / max(1, int(decided.sum()))),
        "missed_risk_ucb95": ucb,
        "certifies_5_percent_95_confidence": bool(math.isfinite(ucb) and ucb <= 0.05),
    }


def select_threshold(y: np.ndarray, scores: np.ndarray, factors: list[float], min_positive: int) -> dict[str, Any]:
    positive_scores = scores[y == 1]
    best: dict[str, Any] | None = None
    for factor in factors:
        threshold = max(0.0, float(np.min(positive_scores)) * factor)
        cert = eval_threshold(y, scores, threshold)
        cert["threshold_name"] = f"positive_floor_x{factor:g}"
        if cert["missed_positive"] != 0 or cert["decided_positive"] < min_positive:
            continue
        if best is None or (
            cert["specificity"],
            cert["precision_referable_action"],
            -cert["referable_action_rate"],
        ) > (
            best["specificity"],
            best["precision_referable_action"],
            -best["referable_action_rate"],
        ):
            best = cert
    if best is None:
        return {"threshold": 1.01, "threshold_name": "no_valid_zero_miss_threshold"}
    return best


def write_predictions(path: Path, split: str, ids: list[str], y: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_id", "referable_dr", "score", "action"])
        if not exists:
            writer.writeheader()
        for image_id, yy, score in zip(ids, y, scores):
            writer.writerow(
                {
                    "split": split,
                    "image_id": image_id,
                    "referable_dr": int(yy),
                    "score": f"{float(score):.8f}",
                    "action": "AUTONOMOUS_REFERABLE" if score >= threshold else "AUTONOMOUS_NONREFERABLE_CANDIDATE",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--model-name", default="resnet50.a1_in1k")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-validation-positive", type=int, default=59)
    parser.add_argument("--threshold-safety-factors", default="0.9,0.75,0.5,0.25")
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(Path(args.manifest))
    by_split = split_rows(rows)
    device = device_for(args.device)
    model, train_transform, eval_transform = build_model(args.model_name)
    model.to(device)

    train_loader = DataLoader(FundusDataset(by_split["train"], train_transform), batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(FundusDataset(by_split["valid"], eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(FundusDataset(by_split["test"], eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_train = np.asarray([int(r["referable_dr"]) for r in by_split["train"]])
    pos_weight = float((y_train == 0).sum() / max(1, int((y_train == 1).sum())))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, y, _ in train_loader:
            images = images.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).flatten()
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            seen += len(y)
        y_valid, s_valid, _ = predict(model, valid_loader, device)
        selected = select_threshold(
            y_valid,
            s_valid,
            [float(x) for x in args.threshold_safety_factors.split(",") if x.strip()],
            args.min_validation_positive,
        )
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, seen), "valid_selected": selected})
        print(json.dumps(history[-1], sort_keys=True))

    y_valid, s_valid, valid_ids = predict(model, valid_loader, device)
    selected = select_threshold(
        y_valid,
        s_valid,
        [float(x) for x in args.threshold_safety_factors.split(",") if x.strip()],
        args.min_validation_positive,
    )
    threshold = float(selected["threshold"])
    valid_cert = eval_threshold(y_valid, s_valid, threshold)
    y_test, s_test, test_ids = predict(model, test_loader, device)
    test_cert = eval_threshold(y_test, s_test, threshold)

    pred_path = out / "predictions.csv"
    if pred_path.exists():
        pred_path.unlink()
    write_predictions(pred_path, "valid", valid_ids, y_valid, s_valid, threshold)
    write_predictions(pred_path, "test", test_ids, y_test, s_test, threshold)
    torch.save({"model_name": args.model_name, "state_dict": model.cpu().state_dict(), "threshold": threshold}, args.artifact)

    summary = {
        "status": (
            "GO_FINETUNED_SINGLE_IMAGE_AUTONOMY"
            if test_cert["certifies_5_percent_95_confidence"] and test_cert["specificity"] >= 0.70
            else "NO_GO_FINETUNED_SINGLE_IMAGE_AUTONOMY"
        ),
        "model_name": args.model_name,
        "artifact": args.artifact,
        "manifest": args.manifest,
        "history": history,
        "selected_threshold": selected,
        "valid_certificate": valid_cert,
        "test_certificate": test_cert,
        "predictions": str(pred_path),
        "boundary": "Fine-tuned DDR single-image autonomy; not paired-view acquisition certification.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
