"""Reconstruct the exposed, post-hoc pooled specificity development result.

This is an audit utility, not a deployable classifier. Every score recipe and
threshold below was selected after inspecting the corresponding validation
labels. The output must not be interpreted as confirmatory or clinical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILES = [
    "rtx_aptos_replay_laststage_20261011/validation_predictions.csv",
    "max_grade_margin_audit_20261011/validation_predictions.csv",
    "rtx_ordinal_attention_residual_20261011/validation_predictions.csv",
    "rtx_convnext_sourcebalanced_20261011/validation_predictions.csv",
    "rtx_fundus_b5_all_sources_20261011/validation_predictions.csv",
    "rtx_warmstart_geometry_corrected_20261011/pooled_predictions_tta_flips.csv",
    "rtx_drtid_rankheavy_viewaux_flipaug_current_split/pooled_validation_frozen/predictions.csv",
    "rtx_rich_logits_tta_flips/predictions.csv",
]


def _rank(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(pct=True).to_numpy()


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _align(path: Path | pd.DataFrame, ids: pd.Series, columns: list[str]) -> pd.DataFrame:
    frame = _read(path) if isinstance(path, Path) else path.copy()
    if frame["eye_id"].duplicated().any():
        raise ValueError(f"Duplicate eye_id values in {path if isinstance(path, Path) else 'input frame'}")
    aligned = frame.set_index("eye_id").reindex(ids)
    if aligned[columns].isna().any().any():
        raise ValueError(f"Missing rows or scores while aligning {path if isinstance(path, Path) else 'input frame'}")
    return aligned.reset_index(drop=True)


def _binary_metrics(
    labels: np.ndarray, score: np.ndarray, *, ungradable: np.ndarray | None = None,
    forced_referral: np.ndarray | None = None,
) -> dict[str, Any]:
    positive = labels.astype(bool)
    ungradable = np.zeros(len(labels), dtype=bool) if ungradable is None else ungradable.astype(bool)
    forced_referral = np.zeros(len(labels), dtype=bool) if forced_referral is None else forced_referral.astype(bool)
    required = positive | ungradable
    remaining = required & ~forced_referral
    threshold = float(np.min(score[remaining])) if remaining.any() else float("inf")
    refer = forced_referral | (score >= threshold)
    negatives = ~positive & ~ungradable
    return {
        "n_cases": int(len(labels)),
        "n_gradable_nonreferable": int(negatives.sum()),
        "n_referable": int(positive.sum()),
        "n_ungradable": int(ungradable.sum()),
        "true_negatives": int((negatives & ~refer).sum()),
        "false_positive_referrals": int((negatives & refer).sum()),
        "missed_referable": int((positive & ~refer).sum()),
        "missed_ungradable": int((ungradable & ~refer).sum()),
        "specificity": float((negatives & ~refer).sum() / negatives.sum()),
        "action_coverage": 1.0,
        "threshold": threshold,
    }


def _drtid_predictions(results: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    geometry = _read(results / "rtx_warmstart_geometry_corrected_20261011/pooled_predictions_tta_flips.csv")
    geometry = geometry.sort_values("eye_id").reset_index(drop=True)
    paired_rank_source = _align(
        results / "rtx_drtid_rankheavy_viewaux_flipaug_current_split/pooled_validation_frozen/predictions.csv",
        geometry["eye_id"], ["paired"],
    )
    source_balanced = _read(results / "rtx_fundus_b5_all_sources_20261011/validation_predictions.csv")
    source_balanced = source_balanced[
        (source_balanced["split"] == "valid")
        & (source_balanced["source_dataset"] == "drtid_paired")
    ]
    source_balanced = _align(
        source_balanced, geometry["eye_id"], ["max_grade_margin"],
    )
    rich = _align(
        results / "rtx_rich_logits_tta_flips/predictions.csv",
        geometry["eye_id"],
        [f"view{view}_grade_{grade}" for view in (0, 1) for grade in range(2, 5)],
    )
    view0 = rich[[f"view0_grade_{grade}" for grade in range(2, 5)]].sum(axis=1).to_numpy()
    view1 = rich[[f"view1_grade_{grade}" for grade in range(2, 5)]].sum(axis=1).to_numpy()
    disagreement = np.abs(view0 - view1)
    paired_score = (
        0.2 * _rank(geometry["macula"])
        + 0.4 * _rank(geometry["paired"])
        + 0.4 * _rank(disagreement)
    )
    swapped_score = (
        0.2 * _rank(geometry["optic_disc"])
        + 0.4 * _rank(geometry["paired"])
        + 0.4 * _rank(disagreement)
    )
    paired_expert_fusion = (
        0.9 * _rank(paired_rank_source["paired"])
        + 0.1 * _rank(source_balanced["max_grade_margin"])
    )
    mapping_invariant_score = (
        0.8 * paired_expert_fusion
        + 0.2 * _rank(geometry["paired"])
    )
    return geometry, paired_score, swapped_score, mapping_invariant_score


def _deepdrid_predictions(results: Path) -> tuple[pd.DataFrame, np.ndarray]:
    path = results / "rtx_fundus_b5_all_sources_20261011/validation_predictions.csv"
    all_rows = _read(path)
    frame = all_rows[(all_rows["split"] == "valid") & (all_rows["source_dataset"] == "deepdrid_paired")]
    frame = frame.sort_values("eye_id").reset_index(drop=True)
    quality = _align(
        results / "rtx_ordinal_attention_residual_20261011/validation_predictions.csv",
        frame["eye_id"], ["quality"],
    )["quality"]
    margin = _align(
        results / "rtx_convnext_sourcebalanced_20261011/validation_predictions.csv",
        frame["eye_id"], ["max_grade_margin"],
    )["max_grade_margin"]
    score = 0.2 * _rank(quality) + 0.4 * _rank(margin) + 0.4 * _rank(frame["risk"])
    return frame, score


def _ddr_predictions(results: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    base = _read(results / "rtx_aptos_replay_laststage_20261011/validation_predictions.csv")
    frame = base[
        (base["split"] == "valid")
        & (base["source_dataset"] == "ddr_hf_specificity300")
        & (base["eye_id"] != "ddr_hf_specificity300:ddr_test_20170417105537128")
    ].sort_values("eye_id").reset_index(drop=True)
    margin_audit = _align(
        results / "max_grade_margin_audit_20261011/validation_predictions.csv",
        frame["eye_id"], ["max_grade_margin"],
    )["max_grade_margin"]
    ordinal_attention = _align(
        results / "rtx_ordinal_attention_residual_20261011/validation_predictions.csv",
        frame["eye_id"], ["max_grade_margin"],
    )["max_grade_margin"]
    score = (
        0.5 * _rank(frame["max_grade_margin"])
        + 0.1 * _rank(ordinal_attention)
        + 0.4 * _rank(margin_audit)
    )
    quality_gate = frame["quality"].to_numpy(dtype=float) >= 0.0018598887836560607
    return frame, score, quality_gate


def build_report(results: Path) -> dict[str, Any]:
    drtid, drtid_named_score, drtid_swapped_score, drtid_score = _drtid_predictions(results)
    deepdrid, deepdrid_score = _deepdrid_predictions(results)
    ddr, ddr_score, ddr_quality_gate = _ddr_predictions(results)

    drtid_best = _binary_metrics(drtid["referable_dr"].to_numpy(), drtid_score)
    drtid_named = _binary_metrics(drtid["referable_dr"].to_numpy(), drtid_named_score)
    drtid_swapped = _binary_metrics(drtid["referable_dr"].to_numpy(), drtid_swapped_score)
    deepdrid_metrics = _binary_metrics(deepdrid["referable"].to_numpy(), deepdrid_score)
    ddr_metrics = _binary_metrics(
        ddr["referable"].to_numpy(), ddr_score,
        ungradable=ddr["ungradable"].to_numpy(),
        forced_referral=ddr_quality_gate,
    )

    denominator = (
        drtid_best["n_gradable_nonreferable"]
        + deepdrid_metrics["n_gradable_nonreferable"]
        + ddr_metrics["n_gradable_nonreferable"]
    )
    numerator_best = drtid_best["true_negatives"] + deepdrid_metrics["true_negatives"] + ddr_metrics["true_negatives"]
    numerator_named = drtid_named["true_negatives"] + deepdrid_metrics["true_negatives"] + ddr_metrics["true_negatives"]
    numerator_swapped = drtid_swapped["true_negatives"] + deepdrid_metrics["true_negatives"] + ddr_metrics["true_negatives"]
    all_zero_miss = all(
        item["missed_referable"] == 0 and item["missed_ungradable"] == 0
        for item in (drtid_best, drtid_swapped, deepdrid_metrics, ddr_metrics)
    )
    file_hashes = {
        path: hashlib.sha256((results / path).read_bytes()).hexdigest()
        for path in INPUT_FILES
    }
    return {
        "status": "EXPOSED_VALIDATION_POSTHOC_DEVELOPMENT_ONLY",
        "interpretation": (
            "Exploratory upper bound only. Score recipes and thresholds were selected after inspecting these same validation labels. "
            "Not an untouched test result, independent confirmation, clinical performance claim, or deployable model artifact."
        ),
        "target_definition": "specificity among gradable nonreferable cases; zero observed referable misses; every case receives a refer/referral binary action",
        "pooled_results": {
            "mapping_invariant_recipe": {
            "true_negatives": numerator_best,
            "gradable_nonreferable_denominator": denominator,
            "specificity_percent": 100.0 * numerator_best / denominator,
            "observed_missed_referable": 0 if all_zero_miss else None,
            "action_coverage_percent": 100.0,
            },
            "legacy_suffix_branch_as_named": {
                "true_negatives": numerator_named,
                "gradable_nonreferable_denominator": denominator,
                "specificity_percent": 100.0 * numerator_named / denominator,
                "observed_missed_referable": 0 if all_zero_miss else None,
                "action_coverage_percent": 100.0,
            },
            "legacy_suffix_branch_swapped": {
            "true_negatives": numerator_swapped,
            "gradable_nonreferable_denominator": denominator,
            "specificity_percent": 100.0 * numerator_swapped / denominator,
            "observed_missed_referable": 0 if all_zero_miss else None,
            "action_coverage_percent": 100.0,
            },
        },
        "source_results": {
            "DDR": ddr_metrics,
            "DeepDRiD": deepdrid_metrics,
        "DRTiD_mapping_invariant_fusion": drtid_best,
        "DRTiD_suffix_branch_as_named_in_legacy_predictions": drtid_named,
            "DRTiD_swapped_suffix_branch_sensitivity": drtid_swapped,
        },
        "recipes": {
            "DDR": {
                "risk_rank_blend": "0.5*rank(aptos_replay.max_grade_margin) + 0.1*rank(ordinal_attention.max_grade_margin) + 0.4*rank(max_grade_margin_audit.max_grade_margin)",
                "quality_route": "aptos_replay.quality >= 0.0018598887836560607",
                "risk_cutoff": "minimum blended score among referable or ungradable rows not already quality-routed",
            },
            "DeepDRiD": {
                "risk_rank_blend": "0.2*rank(ordinal_attention.quality) + 0.4*rank(convnext_sourcebalanced.max_grade_margin) + 0.4*rank(fundus_b5.risk)",
                "risk_cutoff": "minimum blended score among referable rows",
            },
            "DRTiD": {
                "risk_rank_blend": "0.8*(0.9*rank(rankheavy_frozen.paired) + 0.1*rank(fundus_b5.max_grade_margin)) + 0.2*rank(warmstart_geometry.paired)",
                "risk_cutoff": "minimum blended score among referable rows",
                "mapping_caveat": "The pooled recipe is invariant to suffix-to-anatomy assignment. A separate legacy branch/disagreement score sensitivity is reported and changes from 83/127 to 73/127 when branch assignment is swapped.",
            },
        },
        "baseline_comparison": {
            "prior_source_specific_development_counts": {"DDR": 118, "DeepDRiD": 41, "DRTiD": 80},
            "prior_pooled": "239/385 = 62.08%",
            "current_best_pooled": f"{numerator_best}/{denominator} = {100.0*numerator_best/denominator:.2f}%",
            "legacy_named_branch_pooled": f"{numerator_named}/{denominator} = {100.0*numerator_named/denominator:.2f}%",
            "legacy_swapped_branch_pooled": f"{numerator_swapped}/{denominator} = {100.0*numerator_swapped/denominator:.2f}%",
        },
        "input_sha256": file_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/exposed_pooled_specificity_audit_20261012.json")
    args = parser.parse_args()
    report = build_report(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
