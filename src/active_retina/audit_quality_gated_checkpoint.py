"""Export development-only, per-image scores for a quality-gated checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from active_retina.train_quality_gated_autonomy import (
    FundusRows,
    QualityGatedModel,
    predict,
    read_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = int(checkpoint["args"]["image_size"])
    model = QualityGatedModel(checkpoint["model_name"], False, image_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    splits = read_rows(Path(args.manifest))
    root = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    exported: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "status": "DEVELOPMENT_AUDIT_NOT_CERTIFICATION",
        "checkpoint": args.checkpoint,
        "model_name": checkpoint["model_name"],
        "validation_selection_policy": checkpoint.get("best_epoch"),
        "splits": {},
    }
    for split in ("valid", "test"):
        rows = splits[split]
        loader = DataLoader(
            FundusRows(rows, model.eval_transform, root),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        scores = predict(model, loader, device)
        split_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            item = {
                "split": split,
                "image_id": row["image_id"],
                "source_dataset": row["source_dataset"],
                "grade": int(row["grade"]),
                "referable_dr_label": int(row["referable_dr"]),
                "quality_known": int(scores["quality_known"][index]),
                "ungradable_label": int(scores["ungradable"][index]),
                "risk_score": float(scores["risk"][index]),
                "quality_score": float(scores["quality"][index]),
            }
            split_rows.append(item)
            exported.append(item)
        by_source: dict[str, Any] = {}
        for source in sorted({str(row["source_dataset"]) for row in split_rows}):
            source_rows = [r for r in split_rows if r["source_dataset"] == source]
            by_source[source] = {
                "n": len(source_rows),
                "risk_score_by_grade": {
                    str(grade): {
                        "n": sum(r["grade"] == grade for r in source_rows),
                        "min": min((r["risk_score"] for r in source_rows if r["grade"] == grade), default=None),
                        "median": float(np.median([r["risk_score"] for r in source_rows if r["grade"] == grade]))
                        if any(r["grade"] == grade for r in source_rows)
                        else None,
                        "max": max((r["risk_score"] for r in source_rows if r["grade"] == grade), default=None),
                    }
                    for grade in sorted({r["grade"] for r in source_rows})
                },
            }
        summary["splits"][split] = by_source

    with (output_dir / "case_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
