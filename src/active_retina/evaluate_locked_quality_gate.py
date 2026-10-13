"""Apply fixed development thresholds to a previously exposed cohort."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .train_every_image_autonomy import FundusDataset, GradeReferableModel


def summarize(
    reference_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    scores: dict[str, tuple[float, float]],
    risk_threshold: float,
    quality_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_eye: dict[str, list[tuple[float, float]]] = defaultdict(list)
    source_by_eye: dict[str, str] = {}
    for row in manifest_rows:
        eye = row["eye_id"]
        if row["image_id"] in scores:
            by_eye[eye].append(scores[row["image_id"]])
            source_by_eye[eye] = row["source_dataset"]

    predictions: list[dict[str, Any]] = []
    misses = missed_ungradable = tn = n_negative = combined_tn = combined_misses = new_tn = 0
    for row in reference_rows:
        eye = row["eye_id"]
        if eye not in by_eye:
            raise ValueError(f"No model scores for reference eye {eye}")
        risk = max(x[0] for x in by_eye[eye])
        quality = max(x[1] for x in by_eye[eye])
        candidate_action = "REFER" if risk >= risk_threshold or quality >= quality_threshold else "CLEAR"
        baseline_action = row["action"]
        combined_action = "CLEAR" if candidate_action == "CLEAR" or baseline_action == "CLEAR" else "REFER"
        referable = int(row["referable"])
        ungradable = int(row["ungradable"])
        if ungradable:
            missed_ungradable += int(candidate_action == "CLEAR")
            combined_misses += int(combined_action == "CLEAR")
        elif referable:
            misses += int(candidate_action == "CLEAR")
            combined_misses += int(combined_action == "CLEAR")
        else:
            n_negative += 1
            tn += int(candidate_action == "CLEAR")
            combined_tn += int(combined_action == "CLEAR")
            new_tn += int(baseline_action == "REFER" and candidate_action == "CLEAR")
        predictions.append({
            "eye_id": eye,
            "source": source_by_eye[eye],
            "grade": int(row["grade"]),
            "referable": referable,
            "ungradable": ungradable,
            "risk": risk,
            "quality": quality,
            "candidate_action": candidate_action,
            "baseline_action": baseline_action,
            "combined_action": combined_action,
        })
    target = int(math.ceil(0.75 * n_negative))
    summary = {
        "status": "EXPOSED_REFERENCE_TRANSFER_DIAGNOSTIC_NOT_CERTIFICATION",
        "n_reference_cases": len(reference_rows),
        "n_negative": n_negative,
        "frozen_thresholds_from_patient_disjoint_development_validation": {
            "risk": risk_threshold,
            "quality": quality_threshold,
        },
        "candidate_standalone": {
            "true_negatives": tn,
            "specificity": tn / max(1, n_negative),
            "missed_referable": misses,
            "missed_ungradable": missed_ungradable,
            "action_coverage": 1.0,
        },
        "frozen_baseline_plus_candidate": {
            "true_negatives": combined_tn,
            "specificity": combined_tn / max(1, n_negative),
            "additional_true_negatives": new_tn,
            "misses_after_combination": combined_misses,
            "action_coverage": 1.0,
            "true_negatives_required_for_75_percent": target,
            "gap_to_75_percent": max(0, target - combined_tn),
        },
        "warning": (
            "Thresholds were fixed from a patient-disjoint development split and not selected on this cohort. "
            "However, this reference cohort has been repeatedly inspected during prior development and is not an "
            "untouched certification set."
        ),
    }
    return predictions, summary


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--risk-threshold", type=float, required=True)
    parser.add_argument("--quality-threshold", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    reference = read_csv(args.reference)
    manifest = read_csv(args.manifest)
    eyes = {row["eye_id"] for row in reference}
    rows = [dict(row) for row in manifest if row["eye_id"] in eyes]
    if {row["eye_id"] for row in rows} != eyes:
        raise SystemExit("Master manifest does not contain every reference eye")
    for row in rows:
        row["source_split"] = "valid"

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = checkpoint.get("model_name")
    image_size = int(checkpoint.get("args", {}).get("image_size", 224))
    if not model_name or "state_dict" not in checkpoint:
        raise SystemExit("Checkpoint lacks model identity or state_dict")
    device = torch.device(args.device)
    wrapper = GradeReferableModel(model_name, image_size, pretrained=False)
    wrapper.model.load_state_dict(checkpoint["state_dict"], strict=True)
    wrapper.model.to(device).eval()
    dataset = FundusDataset(rows, wrapper.eval_transform, Path.cwd())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    score_by_image: dict[str, tuple[float, float]] = {}
    with torch.inference_mode():
        for images, _, _, image_ids in loader:
            logits = wrapper.model(images.to(device)).float()
            risk = torch.logsumexp(logits[:, 2:5], dim=1) - torch.logsumexp(logits[:, :2], dim=1)
            quality = logits[:, 5] - torch.logsumexp(logits[:, :5], dim=1)
            for image_id, risk_score, quality_score in zip(image_ids, risk.cpu().tolist(), quality.cpu().tolist()):
                score_by_image[str(image_id)] = (float(risk_score), float(quality_score))

    predictions, summary = summarize(
        reference, rows, score_by_image, args.risk_threshold, args.quality_threshold
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    summary.update({"checkpoint": str(args.checkpoint), "best_epoch": checkpoint.get("best_epoch"), "test_evaluated": False})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
