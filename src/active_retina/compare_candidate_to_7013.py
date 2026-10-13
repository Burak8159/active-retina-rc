"""Compare a development candidate with the frozen 70.13 exposed reference."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate(
    reference_path: Path,
    manifest_path: Path,
    predictions_path: Path,
    risk_column: str = "score_referable_logodds",
) -> dict[str, Any]:
    reference = read_csv(reference_path)
    manifest = read_csv(manifest_path)
    predictions = read_csv(predictions_path)
    eye_by_image = {row["image_id"]: row["eye_id"] for row in manifest}
    source_by_eye = {row["eye_id"]: row["source_dataset"] for row in manifest}
    pred_by_eye: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        eye = eye_by_image.get(row["image_id"])
        if eye is not None:
            pred_by_eye[eye].append(row)

    aligned = []
    missing = []
    for row in reference:
        cases = pred_by_eye.get(row["eye_id"])
        if not cases:
            missing.append(row["eye_id"])
            continue
        # The rule is deliberately conservative across an eye's fields.
        risk = max(float(case.get(risk_column, case.get("score"))) for case in cases)
        quality = max(float(case.get("quality_score", case.get("score_ungradable_margin", "-inf"))) for case in cases)
        aligned.append({**row, "candidate_risk": risk, "candidate_quality": quality})
    if missing:
        raise ValueError(f"Candidate predictions missing for {len(missing)} reference eyes; first={missing[:3]}")

    positive_risk = [r["candidate_risk"] for r in aligned if int(r["referable"]) == 1 and int(r["ungradable"]) == 0]
    ungradable_quality = [r["candidate_quality"] for r in aligned if int(r["ungradable"]) == 1]
    if not positive_risk or not ungradable_quality:
        raise ValueError("Reference must include gradable positives and ungradable cases")
    risk_threshold = min(positive_risk)
    quality_threshold = min(ungradable_quality)
    source_thresholds: dict[str, tuple[float, float]] = {}
    for source in sorted({row["source"] for row in aligned}):
        source_rows = [row for row in aligned if row["source"] == source]
        source_positive = [row["candidate_risk"] for row in source_rows if int(row["referable"]) == 1 and int(row["ungradable"]) == 0]
        source_ungradable = [row["candidate_quality"] for row in source_rows if int(row["ungradable"]) == 1]
        source_thresholds[source] = (
            min(source_positive) if source_positive else float("inf"),
            min(source_ungradable) if source_ungradable else float("inf"),
        )

    tn_base = tn_candidate = tn_combined = n_negative = 0
    missed = missed_quality = newly_cleared = 0
    source_tn = source_combined_tn = source_negative = source_missed = source_missed_quality = 0
    for row in aligned:
        label = int(row["referable"])
        ungradable = int(row["ungradable"])
        base_clear = row["action"] == "CLEAR"
        candidate_refer = row["candidate_risk"] >= risk_threshold or row["candidate_quality"] >= quality_threshold
        candidate_clear = not candidate_refer
        if ungradable:
            missed_quality += int(candidate_clear)
        elif label:
            missed += int(candidate_clear)
        else:
            n_negative += 1
            tn_base += int(base_clear)
            tn_candidate += int(candidate_clear)
            tn_combined += int(base_clear or candidate_clear)
            newly_cleared += int((not base_clear) and candidate_clear)

        src_risk, src_quality = source_thresholds[row["source"]]
        source_candidate_clear = not (row["candidate_risk"] >= src_risk or row["candidate_quality"] >= src_quality)
        if ungradable:
            source_missed_quality += int(source_candidate_clear)
        elif label:
            source_missed += int(source_candidate_clear)
        else:
            source_negative += 1
            source_tn += int(source_candidate_clear)
            source_combined_tn += int(base_clear or source_candidate_clear)

    denominator = max(1, n_negative)
    source_denominator = max(1, source_negative)
    target_true_negatives = int(np.ceil(0.75 * denominator))
    combined_gap = max(0, target_true_negatives - tn_combined)
    source_combined_gap = max(0, target_true_negatives - source_combined_tn)
    return {
        "status": "POSTHOC_EXPOSED_COHORT_DIAGNOSTIC_NOT_VALIDATION",
        "n_reference_cases": len(reference),
        "n_negative": n_negative,
        "thresholds_selected_on_reference_labels": {
            "risk_column": risk_column,
            "risk": risk_threshold,
            "quality": quality_threshold,
        },
        "source_thresholds_selected_on_reference_labels": {
            source: {"risk": risk, "quality": quality}
            for source, (risk, quality) in source_thresholds.items()
        },
        "frozen_reference": {
            "true_negatives": tn_base,
            "specificity": tn_base / denominator,
        },
        "candidate_standalone": {
            "true_negatives": tn_candidate,
            "specificity": tn_candidate / denominator,
            "missed_referable": missed,
            "missed_ungradable": missed_quality,
        },
        "reference_plus_candidate_clearance": {
            "true_negatives": tn_combined,
            "specificity": tn_combined / denominator,
            "additional_true_negatives": newly_cleared,
            "missed_referable": 0,
            "missed_ungradable": 0,
            "cases_needed_from_frozen_reference": max(0, target_true_negatives - tn_base),
            "additional_true_negatives_needed_from_this_result": combined_gap,
            "crosses_75_percent": tn_combined / denominator >= 0.75,
        },
        "source_specific_threshold_candidate": {
            "true_negatives": source_tn,
            "specificity": source_tn / source_denominator,
            "missed_referable": source_missed,
            "missed_ungradable": source_missed_quality,
        },
        "reference_plus_source_specific_clearance": {
            "true_negatives": source_combined_tn,
            "specificity": source_combined_tn / source_denominator,
            "additional_true_negatives": source_combined_tn - tn_base,
            "missed_referable": 0,
            "missed_ungradable": 0,
            "cases_needed_from_frozen_reference": max(0, target_true_negatives - tn_base),
            "additional_true_negatives_needed_from_this_result": source_combined_gap,
            "crosses_75_percent": source_combined_tn / source_denominator >= 0.75,
        },
        "warning": (
            "Thresholds were selected using the same inspected reference labels. This result cannot validate future-case "
            "performance, a frozen model, clinical safety, or deployment. The original 70.13 reference is unchanged."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risk-column", default="score_referable_logodds")
    args = parser.parse_args()
    result = evaluate(args.reference, args.manifest, args.predictions, args.risk_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
