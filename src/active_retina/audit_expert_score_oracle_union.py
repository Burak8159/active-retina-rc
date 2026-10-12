"""Audit a label-informed union ceiling over stored expert scores.

This is an exposed-validation diagnostic, not an inference policy. Each score's
zero-miss threshold is selected from the same labeled cohort being counted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_COLUMNS = {
    "eye_id", "patient_id", "grade", "referable", "referable_dr",
    "source_dataset", "split", "ungradable", "gradable", "quality",
    "quality_label", "refer_action", "true_label", "target", "label",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def rank_fraction(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / (2 * len(values))
        start = end
    return ranks


def zero_miss_clearance(
    labels: np.ndarray,
    ungradable: np.ndarray,
    scores: np.ndarray,
    forced_referral: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    labels = np.asarray(labels, dtype=bool)
    ungradable = np.asarray(ungradable, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    forced = np.zeros(len(labels), dtype=bool) if forced_referral is None else np.asarray(
        forced_referral, dtype=bool
    )
    if not (len(labels) == len(ungradable) == len(scores) == len(forced)):
        raise ValueError("Labels, scores, quality flags, and referral flags must align")
    required = (labels | ungradable) & ~forced
    if not required.any():
        raise ValueError("At least one non-forced positive or ungradable case is required")
    threshold = float(scores[required].min())
    return (scores < threshold) & ~forced, threshold


def align_scores(path: Path, eye_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    rows = read_csv(path)
    if not rows or "eye_id" not in rows[0]:
        return {}
    by_id = {row["eye_id"]: row for row in rows}
    if len(by_id) != len(rows) or not set(eye_ids).issubset(by_id):
        return {}
    aligned: dict[str, dict[str, np.ndarray]] = {}
    for column in rows[0]:
        if column in EXCLUDED_COLUMNS:
            continue
        try:
            values = np.asarray([float(by_id[eye_id][column]) for eye_id in eye_ids], dtype=float)
        except (TypeError, ValueError, KeyError):
            continue
        if not np.isfinite(values).all() or np.ptp(values) <= 1e-12:
            continue
        aligned[column] = {"values": rank_fraction(values)}
    return aligned


def _source_rows(results: Path) -> dict[str, dict[str, dict[str, str]]]:
    anchors = {
        "DDR": results / "rtx_aptos_replay_laststage_20261011/validation_predictions.csv",
        "DeepDRiD": results / "rtx_fundus_b5_all_sources_20261011/validation_predictions.csv",
        "DRTiD": results / "rtx_warmstart_geometry_corrected_20261011/pooled_predictions_tta_flips.csv",
    }
    sources: dict[str, dict[str, dict[str, str]]] = {}
    for source, path in anchors.items():
        rows = read_csv(path)
        if source == "DDR":
            rows = [row for row in rows if row.get("split") == "valid"
                    and row.get("source_dataset") == "ddr_hf_specificity300"
                    and row["eye_id"] != "ddr_hf_specificity300:ddr_test_20170417105537128"]
        elif source == "DeepDRiD":
            rows = [row for row in rows if row.get("split") == "valid"
                    and row.get("source_dataset") == "deepdrid_paired"]
        mapping = {row["eye_id"]: row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"Duplicate eye IDs in {path}")
        sources[source] = mapping
    return sources


def _current_recipe_scores(results: Path, source: str, eye_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the established per-source development score and fixed referral gate."""
    forced = np.zeros(len(eye_ids), dtype=bool)
    if source == "DDR":
        base = {row["eye_id"]: row for row in read_csv(
            results / "rtx_aptos_replay_laststage_20261011/validation_predictions.csv"
        )}
        margin = {row["eye_id"]: float(row["max_grade_margin"]) for row in read_csv(
            results / "max_grade_margin_audit_20261011/validation_predictions.csv"
        )}
        ordinal = {row["eye_id"]: float(row["max_grade_margin"]) for row in read_csv(
            results / "rtx_ordinal_attention_residual_20261011/validation_predictions.csv"
        )}
        quality = np.asarray([float(base[eye_id]["quality"]) for eye_id in eye_ids])
        forced = quality >= 0.0018598887836560607
        score = (
            0.5 * rank_fraction(np.asarray([float(base[i]["max_grade_margin"]) for i in eye_ids]))
            + 0.1 * rank_fraction(np.asarray([ordinal[i] for i in eye_ids]))
            + 0.4 * rank_fraction(np.asarray([margin[i] for i in eye_ids]))
        )
    elif source == "DeepDRiD":
        quality = {row["eye_id"]: float(row["quality"]) for row in read_csv(
            results / "rtx_ordinal_attention_residual_20261011/validation_predictions.csv"
        )}
        margin = {row["eye_id"]: float(row["max_grade_margin"]) for row in read_csv(
            results / "rtx_convnext_sourcebalanced_20261011/validation_predictions.csv"
        )}
        base = {row["eye_id"]: row for row in read_csv(
            results / "rtx_fundus_b5_all_sources_20261011/validation_predictions.csv"
        ) if row.get("split") == "valid" and row.get("source_dataset") == "deepdrid_paired"}
        score = (
            0.2 * rank_fraction(np.asarray([quality[i] for i in eye_ids]))
            + 0.4 * rank_fraction(np.asarray([margin[i] for i in eye_ids]))
            + 0.4 * rank_fraction(np.asarray([float(base[i]["risk"]) for i in eye_ids]))
        )
    else:
        geometry = {row["eye_id"]: row for row in read_csv(
            results / "rtx_warmstart_geometry_corrected_20261011/pooled_predictions_tta_flips.csv"
        )}
        rankheavy = {row["eye_id"]: row for row in read_csv(
            results / "rtx_drtid_rankheavy_viewaux_flipaug_current_split/pooled_validation_frozen/predictions.csv"
        )}
        margin = {row["eye_id"]: float(row["max_grade_margin"]) for row in read_csv(
            results / "rtx_fundus_b5_all_sources_20261011/validation_predictions.csv"
        ) if row.get("split") == "valid" and row.get("source_dataset") == "drtid_paired"}
        rankheavy_rank = rank_fraction(np.asarray([float(rankheavy[i]["paired"]) for i in eye_ids]))
        margin_rank = rank_fraction(np.asarray([margin[i] for i in eye_ids]))
        geometry_rank = rank_fraction(np.asarray([float(geometry[i]["paired"]) for i in eye_ids]))
        score = 0.8 * (0.9 * rankheavy_rank + 0.1 * margin_rank) + 0.2 * geometry_rank
    return score, forced


def build_report(results: Path) -> dict[str, Any]:
    sources = _source_rows(results)
    candidate_files = sorted(results.rglob("*.csv"))
    source_results: dict[str, Any] = {}
    pooled_union = 0
    pooled_negatives = 0
    input_hashes: dict[str, str] = {}
    for source, cases in sources.items():
        eye_ids = sorted(cases)
        labels = np.asarray([
            int(row.get("referable", row.get("referable_dr"))) for row in (cases[i] for i in eye_ids)
        ], dtype=bool)
        ungradable = np.asarray([
            int(cases[i].get("ungradable", 0) or 0) for i in eye_ids
        ], dtype=bool)
        negative = ~labels & ~ungradable
        current_score, forced = _current_recipe_scores(results, source, eye_ids)
        union = np.zeros(len(eye_ids), dtype=bool)
        union_required_clear = np.zeros(len(eye_ids), dtype=bool)
        score_count = 0
        contributing: list[str] = []

        score_sets: list[tuple[str, np.ndarray]] = [("current_fused_recipe", rank_fraction(current_score))]
        current_clear, _ = zero_miss_clearance(labels, ungradable, score_sets[0][1], forced)
        current_true_negatives = int(np.sum(current_clear & negative))
        for path in candidate_files:
            try:
                aligned = align_scores(path, eye_ids)
            except (OSError, UnicodeError, csv.Error):
                continue
            for column, entry in aligned.items():
                score_sets.append((f"{path.relative_to(results)}::{column}", entry["values"]))
                input_hashes[str(path.relative_to(results))] = hashlib.sha256(path.read_bytes()).hexdigest()

        for name, score in score_sets:
            clear, _threshold = zero_miss_clearance(labels, ungradable, score, forced)
            if np.any(clear & (labels | ungradable)):
                raise AssertionError(f"Candidate {name} cleared a required positive")
            union_required_clear |= clear & (labels | ungradable)
            added = clear & negative & ~union
            if added.any():
                contributing.append(name)
            union |= clear & ~forced
            score_count += name != "current_fused_recipe"

        if union_required_clear.any():
            raise AssertionError("The score union cleared a required positive")
        true_negatives = int(union.sum())
        negative_count = int(negative.sum())
        observed_misses = int(np.sum(union_required_clear))
        source_results[source] = {
            "cases": len(eye_ids),
            "gradable_nonreferable": negative_count,
            "required_positive_or_ungradable": int(np.sum(labels | ungradable)),
            "individual_score_columns_scanned": score_count,
            "current_recipe_true_negatives": current_true_negatives,
            "oracle_union_true_negatives": true_negatives,
            "oracle_union_specificity_percent": 100.0 * true_negatives / negative_count,
            "observed_missed_required_cases": observed_misses,
            "binary_action_coverage_percent": 100.0,
            "scores_that_added_at_least_one_case": contributing,
        }
        pooled_union += true_negatives
        pooled_negatives += negative_count

    return {
        "status": "EXPOSED_LABEL_INFORMED_ORACLE_UPPER_BOUND_NOT_A_MODEL",
        "claim_boundary": (
            "Every candidate threshold is the minimum required-positive score on the same evaluated cohort; "
            "the score union is therefore label-informed and post-hoc. It is not a fitted or deployable "
            "inference policy, independent evaluation, risk certificate, or clinical result."
        ),
        "target": "specificity among gradable nonreferable eyes, zero observed misses including ungradable cases, 100% binary action assignment",
        "current_recipe_true_negatives": sum(
            item["current_recipe_true_negatives"] for item in source_results.values()
        ),
        "pooled": {
            "true_negatives": pooled_union,
            "gradable_nonreferable": pooled_negatives,
            "specificity_percent": 100.0 * pooled_union / pooled_negatives,
            "observed_missed_required_cases": 0,
            "binary_action_coverage_percent": 100.0,
            "count_needed_for_75_percent": int(np.ceil(0.75 * pooled_negatives)),
            "margin_over_current_recipe_count": pooled_union - sum(
                item["current_recipe_true_negatives"] for item in source_results.values()
            ),
        },
        "source_results": source_results,
        "source_count": len(sources),
        "candidate_input_sha256": input_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/expert_score_oracle_union_20261012.json")
    args = parser.parse_args()
    report = build_report(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
