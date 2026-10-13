"""Expose the post-hoc zero-miss margin frontier for nested OOF folds.

This deliberately uses held-out labels to select source-specific margins. Its
output is an exploratory diagnostic only and must never be reported as a
cross-validated candidate or a deployable operating point.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from active_retina.crossfit_expert_policy_calibration import grouped_folds
from active_retina.nested_crossfit_expert_weights import score


SOURCE_ORDER = ("DDR", "DeepDRiD", "DRTiD")
MARGIN_GRID = (0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03,
               0.04, 0.05, 0.075, 0.1, 0.15, 0.2)


def run(prediction_path: Path, nested_summary_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(prediction_path)
    summary = json.loads(nested_summary_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    manifest = pd.read_csv(root / "results/autonomy_route_three_dataset_manifest_20261011_exactdedup.csv")
    manifest = manifest[manifest["source_dataset"].isin(
        ["ddr_hf_specificity300", "deepdrid_paired", "drtid_paired"])]
    manifest = manifest.drop_duplicates("eye_id").set_index("eye_id")
    frame["patient_id"] = frame["eye_id"].map(manifest["patient_id"])
    if frame["patient_id"].isna().any():
        raise ValueError("Some prediction cases lack patient IDs")

    fold_cache: list[dict[str, Any]] = []
    for record in summary["fold_results"]:
        source = record["source"]
        group = frame[frame["source"] == source].reset_index(drop=True)
        test_idx = grouped_folds(group, summary["outer_folds"], 20261013)[record["fold"] - 1]
        train_idx = np.setdiff1d(np.arange(len(group)), test_idx, assume_unique=True)
        train, test = group.iloc[train_idx], group.iloc[test_idx]
        weights = tuple(record["weights"])
        train_scores = score(source, train, train, weights)
        test_scores = score(source, train, test, weights)
        required_train = train["referable"].to_numpy(bool) | train["ungradable"].to_numpy(bool)
        if not required_train.any():
            raise ValueError(f"No calibration positives for {source} fold {record['fold']}")
        cut = float(train_scores[required_train].min() - record["selected_margin"])
        required_test = test["referable"].to_numpy(bool) | test["ungradable"].to_numpy(bool)
        fold_cache.append({"source": source, "gap": test_scores - cut,
                           "required": required_test})

    best: tuple[int, tuple[float, ...], int] | None = None
    candidates = []
    for margins in itertools.product(MARGIN_GRID, repeat=len(SOURCE_ORDER)):
        tn = negatives = misses = 0
        for fold in fold_cache:
            margin = margins[SOURCE_ORDER.index(fold["source"])]
            referred = fold["gap"] >= -margin
            required = fold["required"]
            negatives_mask = ~required
            tn += int((negatives_mask & ~referred).sum())
            negatives += int(negatives_mask.sum())
            misses += int((required & ~referred).sum())
        candidates.append({"additional_margins": dict(zip(SOURCE_ORDER, margins)),
                           "true_negatives": tn, "negative_denominator": negatives,
                           "specificity_percent": 100.0 * tn / negatives,
                           "misses": misses})
        if misses == 0 and (best is None or (tn, -sum(margins)) > (best[0], -best[2])):
            best = (tn, margins, sum(margins))

    return {
        "status": "POSTHOC_DIAGNOSTIC_ONLY_NOT_A_VALID_CROSS_VALIDATED_CANDIDATE",
        "warning": "Source-specific additional margins are selected using held-out fold labels. This is a descriptive frontier, not an unbiased performance estimate, frozen threshold, deployment artifact, or clinical evidence.",
        "cases": int(len(frame)),
        "action_coverage_percent": 100.0,
        "margin_grid": list(MARGIN_GRID),
        "best_zero_miss_point_on_inspected_folds": next(
            (candidate for candidate in candidates
             if best and candidate["true_negatives"] == best[0]
             and tuple(candidate["additional_margins"][s] for s in SOURCE_ORDER) == best[1]), None),
        "candidate_count": len(candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--nested-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.predictions, args.nested_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["best_zero_miss_point_on_inspected_folds"], sort_keys=True))


if __name__ == "__main__":
    main()
