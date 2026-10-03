"""Evaluate a Hugging Face DR image classifier under the 90/0 autonomy gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
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


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {split: [r for r in rows if r["source_split"] == f"ddr_{split}"] for split in ("train", "valid", "test")}


class ImageRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]], project_root: Path) -> None:
        self.rows = rows
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.rows)

    def resolve_image_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path
        candidate = self.project_root / raw_path
        if candidate.exists():
            return candidate
        normalized = raw_path.replace("\\", "/")
        marker = "/external/"
        if marker in normalized:
            candidate = self.project_root / "external" / normalized.split(marker, 1)[1]
            if candidate.exists():
                return candidate
        return path

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        with Image.open(self.resolve_image_path(row["image_path"])) as img:
            image = img.convert("RGB")
        return {
            "image": image,
            "image_id": row["image_id"],
            "grade": int(row["grade"]),
            "referable_dr": int(row["referable_dr"]),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [x["image"] for x in batch],
        "image_id": [x["image_id"] for x in batch],
        "grade": np.asarray([x["grade"] for x in batch], dtype=int),
        "referable_dr": np.asarray([x["referable_dr"] for x in batch], dtype=int),
    }


def device_for(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def infer_referable_indices(config: Any, positive_labels: str) -> list[int]:
    labels = [x.strip().lower() for x in positive_labels.split(",") if x.strip()]
    id2label = getattr(config, "id2label", None) or {}
    out: list[int] = []
    for idx, label in id2label.items():
        text = str(label).lower()
        if any(token in text for token in labels):
            out.append(int(idx))
    if out:
        return sorted(out)
    # Most DR classifiers expose 5 classes in severity order.
    n = int(getattr(config, "num_labels", 5))
    if n >= 5:
        return list(range(2, n))
    if n == 2:
        return [1]
    raise ValueError(f"Could not infer referable label indices from config labels: {id2label}")


def predict_split(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    project_root: Path,
    batch_size: int,
    device: torch.device,
    referable_indices: list[int],
    use_fp16: bool,
) -> list[dict[str, Any]]:
    loader = DataLoader(ImageRows(rows, project_root), batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval()
    out: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = processor(images=batch["images"], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.amp.autocast("cuda", enabled=bool(use_fp16 and device.type == "cuda")):
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                score = probs[:, referable_indices].sum(dim=1)
            scores = score.detach().cpu().float().numpy()
            prob_arr = probs.detach().cpu().float().numpy()
            for image_id, grade, y, s, p in zip(batch["image_id"], batch["grade"], batch["referable_dr"], scores, prob_arr):
                out.append(
                    {
                        "image_id": image_id,
                        "grade": int(grade),
                        "referable_dr": int(y),
                        "score": float(s),
                        "probs": [float(x) for x in p],
                    }
                )
    return out


def eval_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    y = np.asarray([r["referable_dr"] for r in rows], dtype=int)
    scores = np.asarray([r["score"] for r in rows], dtype=float)
    refer = scores >= threshold
    decided_pos = int(((y == 1) & refer).sum())
    misses = int(((y == 1) & (~refer)).sum())
    false_pos = int(((y == 0) & refer).sum())
    true_neg = int(((y == 0) & (~refer)).sum())
    ucb = binomial_upper(misses, decided_pos, 0.05) if decided_pos else float("nan")
    return {
        "threshold": float(threshold),
        "n": int(len(rows)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "decided_positive": decided_pos,
        "missed_positive": misses,
        "false_positive": false_pos,
        "true_negative": true_neg,
        "specificity": float(true_neg / max(1, int((y == 0).sum()))),
        "referable_action_rate": float(refer.mean()),
        "missed_risk_ucb95": ucb,
        "certifies_5_percent_95_confidence": bool(math.isfinite(ucb) and ucb <= 0.05),
    }


def select_threshold(rows: list[dict[str, Any]], factors: list[float], include_observed: bool, min_positive: int) -> dict[str, Any]:
    y = np.asarray([r["referable_dr"] for r in rows], dtype=int)
    scores = np.asarray([r["score"] for r in rows], dtype=float)
    positive_scores = scores[y == 1]
    candidates: list[tuple[str, float]] = []
    if include_observed:
        candidates += [(f"observed_{i}", float(x)) for i, x in enumerate(sorted(set(scores)))]
    if len(positive_scores):
        floor = float(np.min(positive_scores))
        candidates += [(f"positive_floor_x{f:g}", max(0.0, floor * f)) for f in factors]
    best: dict[str, Any] | None = None
    for name, threshold in candidates:
        row = eval_threshold(rows, threshold)
        row["threshold_name"] = name
        if row["missed_positive"] != 0 or row["decided_positive"] < min_positive:
            continue
        if best is None or (
            row["specificity"],
            row["decided_positive"],
            -row["referable_action_rate"],
        ) > (
            best["specificity"],
            best["decided_positive"],
            -best["referable_action_rate"],
        ):
            best = row
    return best or {"threshold": 1.01, "threshold_name": "no_valid_zero_miss_threshold"}


def write_predictions(path: Path, split: str, rows: list[dict[str, Any]], threshold: float) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "image_id", "grade", "referable_dr", "score", "action", "probs_json"],
        )
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": split,
                    "image_id": row["image_id"],
                    "grade": row["grade"],
                    "referable_dr": row["referable_dr"],
                    "score": f"{row['score']:.8f}",
                    "action": "AUTONOMOUS_REFERABLE" if row["score"] >= threshold else "AUTONOMOUS_NONREFERABLE_CANDIDATE",
                    "probs_json": json.dumps(row["probs"], separators=(",", ":")),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--positive-labels", default="2,3,4,moderate,severe,proliferative,referable")
    parser.add_argument("--min-specificity", type=float, default=0.90)
    parser.add_argument("--min-validation-positive", type=int, default=59)
    parser.add_argument("--threshold-safety-factors", default="0.999,0.99,0.95,0.9,0.8,0.7,0.5,0.25")
    parser.add_argument("--include-observed-thresholds", action="store_true")
    args = parser.parse_args()

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    project_root = Path.cwd()
    rows = read_manifest(Path(args.manifest))
    by_split = split_rows(rows)
    device = device_for(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageClassification.from_pretrained(args.model_id).to(device)
    referable_indices = infer_referable_indices(model.config, args.positive_labels)

    valid = predict_split(model, processor, by_split["valid"], project_root, args.batch_size, device, referable_indices, args.use_fp16)
    selected = select_threshold(
        valid,
        [float(x) for x in args.threshold_safety_factors.split(",") if x.strip()],
        args.include_observed_thresholds,
        args.min_validation_positive,
    )
    threshold = float(selected["threshold"])
    valid_metrics = eval_threshold(valid, threshold)
    test = predict_split(model, processor, by_split["test"], project_root, args.batch_size, device, referable_indices, args.use_fp16)
    test_metrics = eval_threshold(test, threshold)

    pred_path = out / "predictions.csv"
    if pred_path.exists():
        pred_path.unlink()
    write_predictions(pred_path, "valid", valid, threshold)
    write_predictions(pred_path, "test", test, threshold)
    summary = {
        "status": (
            "GO_HF_DR_MODEL_90_SPECIFICITY_ZERO_MISS"
            if test_metrics["missed_positive"] == 0 and test_metrics["specificity"] >= args.min_specificity
            else "NO_GO_HF_DR_MODEL_90_SPECIFICITY_ZERO_MISS"
        ),
        "model_id": args.model_id,
        "manifest": args.manifest,
        "referable_indices": referable_indices,
        "id2label": getattr(model.config, "id2label", None),
        "selected_threshold": selected,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "target_specificity": args.min_specificity,
        "predictions": str(pred_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
