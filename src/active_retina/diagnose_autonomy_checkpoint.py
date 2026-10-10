"""Audit validation score failures for an every-image autonomy checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .train_every_image_autonomy import canonical_split, referable_logit


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = []
        for row in csv.DictReader(stream):
            if canonical_split(str(row.get("source_split", ""))) != "valid":
                continue
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    if not rows:
        raise ValueError("Manifest has no validation rows.")
    return rows


class ValidationImages(Dataset):
    def __init__(self, rows: list[dict[str, Any]], root: Path, transform: Any) -> None:
        self.rows = rows
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        raw = str(row["image_path"])
        path = Path(raw)
        if not path.exists():
            normalized = raw.replace("\\", "/")
            marker = "/external/"
            path = self.root / "external" / normalized.split(marker, 1)[1] if marker in normalized else self.root / raw
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    root = Path.cwd()
    rows = read_rows(Path(args.manifest))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    import timm

    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=6)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(args.device)).eval()
    config = timm.data.resolve_model_data_config(model)
    size = int(checkpoint.get("args", {}).get("image_size", 224))
    config["input_size"] = (3, size, size)
    transform = timm.data.create_transform(**config, is_training=False)
    dataset = ValidationImages(rows, root, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    scores = np.empty(len(rows), dtype=np.float64)
    with torch.inference_mode():
        for images, indices in loader:
            logits = model(images.to(args.device)).float()
            batch_scores = referable_logit(logits).cpu().numpy().astype(np.float64)
            scores[indices.numpy()] = batch_scores

    threshold = float(checkpoint["threshold"])
    predictions = []
    for row, score in zip(rows, scores):
        predictions.append(
            {
                "source_dataset": row.get("source_dataset", "unknown"),
                "source_split": row.get("source_split", ""),
                "patient_id": row.get("patient_id", ""),
                "eye_id": row.get("eye_id", ""),
                "image_id": row.get("image_id", ""),
                "grade": int(row["grade"]),
                "referable_dr": int(row["referable_dr"]),
                "decision_score_logit": float(score),
                "threshold": threshold,
                "action": "REFER" if score >= threshold else "CLEAR",
                "image_path": row["image_path"],
            }
        )

    positives = [row for row in predictions if row["referable_dr"] == 1]
    negatives = [row for row in predictions if row["referable_dr"] == 0]
    positives.sort(key=lambda row: row["decision_score_logit"])
    negatives.sort(key=lambda row: row["decision_score_logit"], reverse=True)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_source[str(row["source_dataset"])].append(row)

    source_summary = {}
    for source, group in sorted(by_source.items()):
        pos = [row["decision_score_logit"] for row in group if row["referable_dr"] == 1]
        neg = [row["decision_score_logit"] for row in group if row["referable_dr"] == 0]
        source_summary[source] = {
            "n": len(group),
            "positive": len(pos),
            "negative": len(neg),
            "misses_at_global_threshold": sum(score < threshold for score in pos),
            "specificity_at_global_threshold": (
                sum(score < threshold for score in neg) / len(neg) if neg else None
            ),
            "minimum_positive_score": min(pos) if pos else None,
            "negative_score_percentiles": (
                {str(q): float(np.percentile(neg, q)) for q in (1, 5, 25, 50, 75, 95, 99)} if neg else {}
            ),
        }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "validation_case_scores.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    summary = {
        "status": "VALIDATION_FAILURE_AUDIT_ONLY",
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "decision_score": checkpoint.get("decision_score", "referable_logit"),
        "threshold": threshold,
        "n_validation": len(predictions),
        "positive": len(positives),
        "negative": len(negatives),
        "misses": sum(row["action"] == "CLEAR" for row in positives),
        "specificity": sum(row["action"] == "CLEAR" for row in negatives) / max(1, len(negatives)),
        "hardest_positive_cases": positives[:25],
        "highest_scoring_negative_cases": negatives[:25],
        "by_source": source_summary,
        "boundary": "Validation diagnostic only. It is not independent test or certification evidence.",
        "predictions": str(prediction_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
