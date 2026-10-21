#!/usr/bin/env python3
"""Post-hoc diagnostic summary of a frozen MMRDR test prediction export.

This intentionally does not select or recommend a deployment threshold. The
official test labels have already been inspected; outputs are descriptive only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(path: Path, expected_hash: str) -> dict[str, Any]:
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise ValueError("test prediction export hash differs from the frozen run card")

    rows: list[tuple[float, int, int]] = []
    thresholds: set[float] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"grade", "referable_dr", "score", "threshold", "action"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"prediction export must contain {sorted(required)}")
        for row in reader:
            grade = int(row["grade"])
            label = int(row["referable_dr"])
            score = float(row["score"])
            threshold = float(row["threshold"])
            if grade not in range(5) or label != int(grade >= 2):
                raise ValueError("invalid grade/referable label")
            if not math.isfinite(score) or not math.isfinite(threshold):
                raise ValueError("scores and thresholds must be finite")
            if row["action"] != ("REFER" if score >= threshold else "CLEAR"):
                raise ValueError("stored action disagrees with frozen score threshold")
            rows.append((score, label, grade))
            thresholds.add(threshold)

    if len(thresholds) != 1:
        raise ValueError("prediction export must contain exactly one frozen threshold")
    frozen_threshold = next(iter(thresholds))

    positives = sum(label for _, label, _ in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        raise ValueError("both classes are required")

    # Mann-Whitney AUC with average ranks for ties.
    ranked = sorted((score, label) for score, label, _ in rows)
    positive_rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ranked[i:j])
        i = j
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    descending = sorted(rows, key=lambda item: item[0], reverse=True)
    roc_steps = [{"false_positive_rate": 0.0, "true_positive_rate": 0.0}]
    operating_steps = []
    cumulative_tp = cumulative_fp = 0
    ap = 0.0
    i = 0
    while i < len(descending):
        j = i + 1
        while j < len(descending) and descending[j][0] == descending[i][0]:
            j += 1
        group_tp = sum(label for _, label, _ in descending[i:j])
        group_fp = j - i - group_tp
        cumulative_tp += group_tp
        cumulative_fp += group_fp
        sensitivity = cumulative_tp / positives
        specificity = (negatives - cumulative_fp) / negatives
        ap += (group_tp / positives) * (cumulative_tp / (cumulative_tp + cumulative_fp))
        roc_steps.append({
            "false_positive_rate": cumulative_fp / negatives,
            "true_positive_rate": sensitivity,
        })
        operating_steps.append({
            "threshold": descending[i][0],
            "sensitivity": sensitivity,
            "specificity": specificity,
        })
        i = j

    # Fixed-FPR grid keeps the public figure aggregate-only instead of exposing
    # one threshold point per image. Exact AUROC above still uses every score.
    roc_display = []
    step_index = 0
    for grid_index in range(101):
        fpr = grid_index / 100
        while (step_index + 1 < len(roc_steps)
               and roc_steps[step_index + 1]["false_positive_rate"] <= fpr + 1e-12):
            step_index += 1
        roc_display.append({
            "false_positive_rate": fpr,
            "true_positive_rate": roc_steps[step_index]["true_positive_rate"],
        })

    # Descriptive upper envelope only. Thresholds are evaluated after test-label
    # inspection and must not be used as candidate operating points.
    frontier = []
    for target_sensitivity in (1.00, 0.99, 0.98, 0.95, 0.90, 0.80):
        best = next(
            step for step in operating_steps
            if step["sensitivity"] + 1e-12 >= target_sensitivity
        )
        frontier.append({"minimum_sensitivity": target_sensitivity, **best})

    grade_summary = []
    for grade in range(5):
        subset = [(score, label) for score, label, g in rows if g == grade]
        if subset:
            grade_summary.append({
                "grade": grade,
                "n": len(subset),
                "refer_rate_at_frozen_threshold": sum(
                    1 for score, _ in subset if score >= frozen_threshold
                ) / len(subset),
            })

    frozen_tp = sum(label for score, label, _ in rows if score >= frozen_threshold)
    frozen_fp = sum(1 - label for score, label, _ in rows if score >= frozen_threshold)
    frozen_fn = positives - frozen_tp
    frozen_tn = negatives - frozen_fp
    return {
        "status": "POST_HOC_TEST_SCORE_DIAGNOSTIC_NOT_VALIDATION_OR_THRESHOLD_SELECTION",
        "export_sha256": observed_hash,
        "n": len(rows),
        "positives": positives,
        "negatives": negatives,
        "roc_auc": auc,
        "average_precision": ap,
        "roc_display_grid": roc_display,
        "roc_display_grid_note": (
            "101-point fixed-FPR empirical ROC step envelope for visualization only; "
            "exact AUROC is computed from all image scores. No identifiers or score thresholds are included."
        ),
        "frozen_threshold": frozen_threshold,
        "frozen_operating_point": {
            "true_positive": frozen_tp,
            "false_negative": frozen_fn,
            "true_negative": frozen_tn,
            "false_positive": frozen_fp,
            "sensitivity": frozen_tp / positives,
            "specificity": frozen_tn / negatives,
            "referral_rate": (frozen_tp + frozen_fp) / len(rows),
        },
        "grade_stratified": grade_summary,
        "sensitivity_specificity_frontier": frontier,
        "scope": "Threshold-free ranking and retrospective test-label frontier computed after the official test was inspected. No threshold is recommended; these results must not be used to repair or select the model.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-export", type=Path, required=True)
    parser.add_argument("--run-card", type=Path, default=Path("paper/reproducibility/mmrdr_uwf_official_test_run_card.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--roc-display-csv", type=Path,
                        help="Optional aggregate-only 101-point ROC display grid CSV")
    args = parser.parse_args()
    card = json.loads(args.run_card.read_text(encoding="utf-8"))
    expected_hash = card["provenance_hashes"]["test_prediction_export_local_only"]["sha256"]
    report = analyze(args.prediction_export, expected_hash)
    if args.roc_display_csv:
        args.roc_display_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.roc_display_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("false_positive_rate", "true_positive_rate")
            )
            writer.writeheader()
            # The fixed-FPR grid starts at the first empirical step; include
            # the origin so the plotted ROC trace has its conventional start.
            writer.writerow({"false_positive_rate": 0.0, "true_positive_rate": 0.0})
            writer.writerows(report["roc_display_grid"])
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
