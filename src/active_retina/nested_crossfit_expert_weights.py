"""Nested patient-grouped evaluation of fixed score-blend weights.

This is a development diagnostic: component models and this cohort have been
inspected previously. It prevents the outer test patients from selecting the
blend weights or score threshold.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from active_retina.crossfit_expert_policy_calibration import grouped_folds, rank_against
from active_retina.exposed_expert_policy import FEATURES


def weight_grid(n_features: int, denominator: int = 10) -> list[tuple[float, ...]]:
    def compositions(total: int, parts: int):
        if parts == 1:
            yield (total,)
        else:
            for first in range(total + 1):
                for rest in compositions(total - first, parts - 1):
                    yield (first, *rest)
    return [tuple(value / denominator for value in row)
            for row in compositions(denominator, n_features)]


def score(source: str, reference: pd.DataFrame, cases: pd.DataFrame,
          weights: tuple[float, ...]) -> np.ndarray:
    columns = FEATURES[source]
    ranked = [rank_against(reference[col].to_numpy(), cases[col].to_numpy())
              for col in columns]
    return np.sum(np.asarray(ranked) * np.asarray(weights)[:, None], axis=0)


def evaluate_weights(source: str, train: pd.DataFrame, test: pd.DataFrame,
                     weights: tuple[float, ...], margin: float = 0.0) -> dict[str, Any]:
    scores_train = score(source, train, train, weights)
    scores_test = score(source, train, test, weights)
    required_train = train["referable"].to_numpy(bool) | train["ungradable"].to_numpy(bool)
    if not required_train.any():
        raise ValueError(f"No required-positive cases in {source} calibration fold")
    threshold = float(scores_train[required_train].min() - margin)
    referred = scores_test >= threshold
    required_test = test["referable"].to_numpy(bool) | test["ungradable"].to_numpy(bool)
    negatives = ~required_test
    return {
        "true_negatives": int((negatives & ~referred).sum()),
        "negatives": int(negatives.sum()),
        "misses": int((required_test & ~referred).sum()),
        "threshold": threshold,
        "margin": margin,
    }


def select_weights(source: str, frame: pd.DataFrame, inner_folds: int,
                   seed: int) -> tuple[tuple[float, ...], dict[str, int]]:
    folds = grouped_folds(frame, inner_folds, seed)
    all_indices = np.arange(len(frame))
    candidates = weight_grid(len(FEATURES[source]))
    evaluated: list[tuple[tuple[int, int, float], tuple[float, ...], float, dict[str, int]]] = []
    for weights in candidates:
        for margin in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2):
            tn = negatives = misses = 0
            for test_idx in folds:
                train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=True)
                result = evaluate_weights(source, frame.iloc[train_idx], frame.iloc[test_idx],
                                          weights, margin)
                tn += result["true_negatives"]
                negatives += result["negatives"]
                misses += result["misses"]
            # Enforce zero misses first; among ties, maximize specificity.
            key = (int(misses == 0), -misses, tn / max(1, negatives))
            evaluated.append((key, weights, margin, {"inner_true_negatives": tn,
                                                      "inner_negatives": negatives,
                                                      "inner_misses": misses}))
    _, weights, margin, summary = max(evaluated, key=lambda row: row[0])
    summary["selected_margin"] = margin
    return weights, summary


def run(csv_path: Path, outer_folds: int = 5, inner_folds: int = 4) -> dict[str, Any]:
    frame = pd.read_csv(csv_path)
    root = Path(__file__).resolve().parents[2]
    manifest = pd.read_csv(root / "results/autonomy_route_three_dataset_manifest_20261011_exactdedup.csv")
    manifest = manifest[manifest["source_dataset"].isin(
        ["ddr_hf_specificity300", "deepdrid_paired", "drtid_paired"])]
    consistency = manifest.groupby("eye_id")[["patient_id", "source_dataset", "referable_dr"]].nunique()
    if (consistency > 1).any().any():
        raise ValueError("Master manifest has conflicting patient/source/label rows for an eye")
    lookup = manifest.drop_duplicates("eye_id").set_index("eye_id")
    frame["patient_id"] = frame["eye_id"].map(lookup["patient_id"])
    source_names = {"DDR": "ddr_hf_specificity300", "DeepDRiD": "deepdrid_paired",
                    "DRTiD": "drtid_paired"}
    if frame["patient_id"].isna().any():
        raise ValueError("Some replay cases lack patient IDs")
    if not frame["source"].map(source_names).equals(frame["eye_id"].map(lookup["source_dataset"])):
        raise ValueError("Replay source labels disagree with the master manifest")
    if not np.array_equal(frame["referable"].to_numpy(int),
                          frame["eye_id"].map(lookup["referable_dr"]).to_numpy(int)):
        raise ValueError("Replay labels disagree with the master manifest")

    records: list[dict[str, Any]] = []
    for source in ("DDR", "DeepDRiD", "DRTiD"):
        group = frame[frame["source"] == source].reset_index(drop=True)
        outer = grouped_folds(group, outer_folds, 20261013)
        all_indices = np.arange(len(group))
        for fold, test_idx in enumerate(outer, 1):
            train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=True)
            train, test = group.iloc[train_idx], group.iloc[test_idx]
            weights, inner = select_weights(source, train, inner_folds,
                                            seed=20261013 + fold)
            margin = inner["selected_margin"]
            outer_result = evaluate_weights(source, train, test, weights, margin)
            records.append({
                "source": source, "fold": fold,
                "calibration_patients": int(train["patient_id"].nunique()),
                "heldout_patients": int(test["patient_id"].nunique()),
                "patient_disjoint": not bool(set(train["patient_id"]) & set(test["patient_id"])),
                "weights": list(weights), **inner, **outer_result,
            })
    total_tn = sum(row["true_negatives"] for row in records)
    total_neg = sum(row["negatives"] for row in records)
    total_misses = sum(row["misses"] for row in records)
    return {
        "status": "NESTED_CROSSFIT_DEVELOPMENT_ONLY",
        "caveat": "The component-model family and cohort were previously inspected. Nested patient folds prevent this run's outer patients from selecting blend weights or thresholds, but this is not independent confirmation.",
        "outer_folds": outer_folds, "inner_folds": inner_folds,
        "patient_disjoint": all(row["patient_disjoint"] for row in records),
        "cases": int(len(frame)), "gradable_negatives": total_neg,
        "true_negatives": total_tn,
        "specificity_percent": 100.0 * total_tn / total_neg,
        "misses_referable_or_ungradable": total_misses,
        "action_coverage_percent": 100.0,
        "reference_same_cohort_70_13": {"specificity_percent": 70.12987012987013,
                                        "true_negatives": 270, "misses": 0},
        "fold_results": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    args = parser.parse_args()
    result = run(args.predictions, args.outer_folds, args.inner_folds)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "patient_disjoint", "specificity_percent", "true_negatives",
        "gradable_negatives", "misses_referable_or_ungradable", "action_coverage_percent")},
        sort_keys=True))


if __name__ == "__main__":
    main()
