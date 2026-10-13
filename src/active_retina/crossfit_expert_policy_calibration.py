"""Cross-fit fixed per-source calibration of the 70.13% expert-score recipe.

Base models and this validation cohort were previously inspected; this is a
development diagnostic, not independent confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from active_retina.exposed_expert_policy import FEATURES


WEIGHTS = {
    "DDR": (0.5, 0.1, 0.4),
    "DeepDRiD": (0.2, 0.4, 0.4),
    "DRTiD": (0.72, 0.08, 0.2),
}


def rank_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    if reference.size == 0 or not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("Calibration references and scores must be finite and nonempty")
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return ((left + 1) + right) / (2.0 * len(reference))


def combined_score(source: str, reference: pd.DataFrame, cases: pd.DataFrame) -> np.ndarray:
    names = FEATURES[source]
    ranked = [rank_against(reference[name].to_numpy(), cases[name].to_numpy()) for name in names]
    if source == "DRTiD":
        return 0.8 * (0.9 * ranked[0] + 0.1 * ranked[1]) + 0.2 * ranked[2]
    return sum(weight * score for weight, score in zip(WEIGHTS[source], ranked))


def fit_threshold(
    source: str, calibration: pd.DataFrame, scores: np.ndarray
) -> tuple[float, float]:
    required = calibration["referable"].to_numpy(dtype=bool) | calibration["ungradable"].to_numpy(dtype=bool)
    quality_values = calibration["quality"].to_numpy(dtype=float) if source == "DDR" else None
    q_candidates = np.unique(quality_values) if source == "DDR" else np.asarray([np.nan])
    if source == "DDR":
        q_candidates = np.append(q_candidates, np.nextafter(np.max(quality_values), np.inf))
    best: tuple[int, float, float] | None = None
    negative = ~required
    for q_threshold in q_candidates:
        forced = quality_values >= q_threshold if source == "DDR" else np.zeros(len(calibration), dtype=bool)
        remaining_positive = required & ~forced
        risk_threshold = float(scores[remaining_positive].min()) if remaining_positive.any() else float("inf")
        refer = forced | (scores >= risk_threshold)
        if np.any(required & ~refer):
            raise AssertionError("Calibration threshold missed a required calibration case")
        tn = int((negative & ~refer).sum())
        # Prefer the less aggressive quality gate if specificity ties.
        key = (tn, float(q_threshold))
        if best is None or key > (best[0], best[1]):
            best = (tn, float(q_threshold), risk_threshold)
    assert best is not None
    return best[1], best[2]


def grouped_folds(frame: pd.DataFrame, folds: int, seed: int) -> list[np.ndarray]:
    """Assign whole patients to folds, stratifying by any required-positive eye."""
    by_patient = frame.groupby("patient_id", sort=True)
    strata: dict[bool, list[str]] = {False: [], True: []}
    for patient, rows in by_patient:
        required = rows["referable"].astype(bool) | rows["ungradable"].astype(bool)
        strata[bool(required.any())].append(str(patient))
    if min(map(len, strata.values())) < folds:
        raise ValueError("Not enough positive and negative patients for requested grouped folds")
    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    for label, patients in strata.items():
        shuffled = np.asarray(patients, dtype=object)
        rng.shuffle(shuffled)
        for i, patient in enumerate(shuffled):
            assignment[str(patient)] = i % folds
    return [
        np.flatnonzero(frame["patient_id"].astype(str).map(assignment).to_numpy() == fold)
        for fold in range(folds)
    ]


def run(csv_path: Path, folds: int = 5) -> dict[str, Any]:
    frame = pd.read_csv(csv_path)
    manifest = pd.read_csv(
        Path(__file__).resolve().parents[2]
        / "results/autonomy_route_three_dataset_manifest_20261011_exactdedup.csv"
    )
    source_names = {
        "DDR": "ddr_hf_specificity300",
        "DeepDRiD": "deepdrid_paired",
        "DRTiD": "drtid_paired",
    }
    manifest = manifest[manifest["source_dataset"].isin(source_names.values())]
    consistency = manifest.groupby("eye_id")[["patient_id", "source_dataset", "referable_dr"]].nunique()
    if (consistency > 1).any().any():
        raise ValueError("Master manifest has conflicting patient/source/label rows for an eye")
    manifest = manifest.drop_duplicates("eye_id")
    patient_lookup = manifest.set_index("eye_id")[["patient_id", "source_dataset", "referable_dr"]]
    mapped = frame["eye_id"].map(patient_lookup["patient_id"])
    if mapped.isna().any():
        raise ValueError(f"{int(mapped.isna().sum())} replay eyes are absent from the patient manifest")
    frame["patient_id"] = mapped.astype(str)
    expected_source = frame["source"].map(source_names)
    actual_source = frame["eye_id"].map(patient_lookup["source_dataset"])
    if expected_source.isna().any() or not expected_source.equals(actual_source):
        raise ValueError("Replay source labels do not match the master manifest")
    manifest_labels = frame["eye_id"].map(patient_lookup["referable_dr"]).astype(int)
    if not np.array_equal(manifest_labels.to_numpy(), frame["referable"].astype(int).to_numpy()):
        raise ValueError("Replay labels do not match the master manifest")
    actions: dict[tuple[str, str], str] = {}
    fold_records: list[dict[str, Any]] = []
    for source in ("DDR", "DeepDRiD", "DRTiD"):
        group = frame[frame["source"] == source].reset_index(drop=True)
        fold_rows = grouped_folds(group, folds, 20261013)
        all_indices = np.arange(len(group))
        for fold, test_idx in enumerate(fold_rows, 1):
            cal_idx = np.setdiff1d(all_indices, test_idx, assume_unique=True)
            calibration, test = group.iloc[cal_idx], group.iloc[test_idx]
            references = calibration
            cal_scores = combined_score(source, references, calibration)
            test_scores = combined_score(source, references, test)
            q_threshold, risk_threshold = fit_threshold(source, calibration, cal_scores)
            if source == "DDR":
                test_forced = test["quality"].to_numpy(dtype=float) >= q_threshold
            else:
                test_forced = np.zeros(len(test), dtype=bool)
            referred = test_forced | (test_scores >= risk_threshold)
            required = test["referable"].to_numpy(dtype=bool) | test["ungradable"].to_numpy(dtype=bool)
            negative = ~required
            missed = int((required & ~referred).sum())
            tn = int((negative & ~referred).sum())
            fold_records.append({
                "source": source,
                "fold": fold,
                "calibration_cases": int(len(calibration)),
                "heldout_cases": int(len(test)),
                "heldout_negatives": int(negative.sum()),
                "true_negatives": tn,
                "misses_referable_or_ungradable": missed,
                "specificity": tn / int(negative.sum()),
                "patient_disjoint": not bool(set(calibration["patient_id"]) & set(test["patient_id"])),
                "risk_threshold": risk_threshold,
            "quality_threshold": None if np.isnan(q_threshold) else q_threshold,
            })
            for row_index, decision in zip(test.index, referred):
                actions[(source, str(group.loc[row_index, "eye_id"]))] = "REFER" if decision else "CLEAR"

    total_negatives = sum(row["heldout_negatives"] for row in fold_records)
    total_tn = sum(row["true_negatives"] for row in fold_records)
    total_misses = sum(row["misses_referable_or_ungradable"] for row in fold_records)
    return {
        "status": "CROSSFIT_POLICY_CALIBRATION_DEVELOPMENT_ONLY",
        "base_model_selection_caveat": "The component model family and validation cohort were previously inspected; patient-grouped folds only isolate calibration fitting within this cohort.",
        "reference_replay_specificity": 70.12987012987013,
        "reference_replay_true_negatives": 270,
        "crossfit": {
            "folds": folds,
            "patient_disjoint": all(row["patient_disjoint"] for row in fold_records),
            "true_negatives": total_tn,
            "negative_denominator": total_negatives,
            "specificity_percent": 100.0 * total_tn / total_negatives,
            "misses_referable_or_ungradable": total_misses,
            "binary_action_coverage_percent": 100.0 * len(actions) / len(frame),
        },
        "fold_results": fold_records,
        "frozen_policy_thresholds_are_not_reused_across_folds": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    result = run(args.predictions, args.folds)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["crossfit"], sort_keys=True))


if __name__ == "__main__":
    main()
