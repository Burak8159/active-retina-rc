"""Run a saved MMRDR-UWF checkpoint on the previously exposed 70.13 cohort."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .train_every_image_autonomy import FundusDataset, GradeReferableModel, predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-replay", type=Path, required=True)
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with args.reference_replay.open("r", encoding="utf-8-sig", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    with args.master_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    wanted = {row["eye_id"] for row in reference_rows}
    reference_by_id = {row["eye_id"]: row for row in reference_rows}
    if len(wanted) != len(reference_rows):
        raise SystemExit("Reference replay contains duplicate case IDs")
    rows = [dict(row) for row in manifest_rows if row["eye_id"] in wanted]
    found = {row["eye_id"] for row in rows}
    if found != wanted:
        raise SystemExit(f"Manifest is missing cases: {sorted(wanted - found)[:5]}")
    for row in rows:
        reference = reference_by_id[row["eye_id"]]
        if int(row["referable_dr"]) != int(reference["referable"]):
            raise SystemExit(f"Label mismatch for case {row['eye_id']}")
        row["source_split"] = "valid"

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = checkpoint.get("model_name")
    image_size = int(checkpoint.get("args", {}).get("image_size", 224))
    if not model_name or "state_dict" not in checkpoint:
        raise SystemExit("Checkpoint is missing model identity or state_dict")
    device = torch.device(args.device)
    wrapper = GradeReferableModel(model_name, image_size, pretrained=False)
    wrapper.model.load_state_dict(checkpoint["state_dict"], strict=True)
    wrapper.model.to(device)
    wrapper.model.eval()
    dataset = FundusDataset(rows, wrapper.eval_transform, Path.cwd())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    labels, grades, scores, image_ids = predict(wrapper.model, loader, device)
    if image_ids != [row["image_id"] for row in rows]:
        raise SystemExit("Inference output order/IDs do not match the input manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.csv"
    case_scores: dict[str, list[float]] = {}
    case_grades: dict[str, list[int]] = {}
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["eye_id", "source", "image_id", "grade", "referable", "score"])
        writer.writeheader()
        by_id = {row["image_id"]: row for row in rows}
        for image_id, grade, label, score in zip(image_ids, grades, labels, scores):
            row = by_id[image_id]
            case_scores.setdefault(row["eye_id"], []).append(float(score))
            case_grades.setdefault(row["eye_id"], []).append(int(grade))
            writer.writerow({
                "eye_id": row["eye_id"],
                "source": row["source_dataset"],
                "image_id": image_id,
                "grade": int(grade),
                "referable": int(label),
                "score": f"{float(score):.9g}",
            })
    case_path = args.output_dir / "case_predictions.csv"
    with case_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["eye_id", "source", "grade", "referable", "score_max", "n_views"])
        writer.writeheader()
        refs = reference_by_id
        for eye_id in sorted(case_scores):
            members = [row for row in rows if row["eye_id"] == eye_id]
            writer.writerow({
                "eye_id": eye_id,
                "source": members[0]["source_dataset"],
                "grade": max(case_grades[eye_id]),
                "referable": int(refs[eye_id]["referable"]),
                "score_max": f"{max(case_scores[eye_id]):.9g}",
                "n_views": len(case_scores[eye_id]),
            })
    summary = {
        "status": "EXPOSED_COHORT_INFERENCE_ONLY_THRESHOLD_NOT_SELECTED",
        "checkpoint": str(args.checkpoint),
        "model_name": model_name,
        "image_size": image_size,
        "n_scored": len(image_ids),
        "n_cases": len(case_scores),
        "case_aggregation": "maximum risk across all views belonging to the case",
        "sources": sorted({row["source_dataset"] for row in rows}),
        "official_mmrdr_test_read": False,
        "warning": "This cohort was repeatedly inspected during development. Do not interpret scores here as independent validation or clinical evidence.",
        "predictions": str(prediction_path),
        "case_predictions": str(case_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
