#!/usr/bin/env python3
"""Recompute and audit the exposed 70.13% score-level replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
REPLAY_RELATIVE = Path("results/deployable_exposed_baseline_7013_20261013_with_components")
SUMMARY_RELATIVES = {
    "replay": REPLAY_RELATIVE / "summary.json",
    "crossfit": Path("results/crossfit_expert_calibration_20261013_corrected/summary.json"),
    "nested": Path("results/nested_crossfit_expert_weights_20261013/summary.json"),
    "nested_margin": Path("results/nested_crossfit_expert_weights_margin_20261013/summary.json"),
    "candidate_transfer": Path("results/clean_candidate_on_exposed981_20261013/summary.json"),
}
MMRDR_RUN_CARD = Path("paper/reproducibility/mmrdr_uwf_official_test_run_card.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total**2)
    ) / denominator
    return [center - radius, center + radius]


def audit_mmrdr_aggregate(root: Path) -> dict[str, Any]:
    """Cross-check tracked MMRDR aggregates; this does not run image inference."""
    root = root.resolve()
    run_card_path = root / MMRDR_RUN_CARD
    run_card = read_json(run_card_path)
    validation = run_card["validation_result"]
    trace_meta = run_card["validation_selection_trace"]
    history_path = root / validation["source_summary_path"]
    trace_path = root / trace_meta["path"]
    history = read_json(history_path)

    if sha256(history_path) != validation["source_summary_sha256"]:
        raise AssertionError("MMRDR validation summary hash disagrees with the run card")
    if sha256(trace_path) != trace_meta["sha256"]:
        raise AssertionError("MMRDR validation trace hash disagrees with the run card")

    epochs = history.get("history")
    if not isinstance(epochs, list) or len(epochs) != trace_meta["recorded_epochs"]:
        raise AssertionError("MMRDR history length disagrees with the run card")
    with trace_path.open(newline="", encoding="utf-8") as stream:
        trace_rows = list(csv.DictReader(stream))
    if len(trace_rows) != len(epochs):
        raise AssertionError("MMRDR trace row count disagrees with source history")

    zero_miss_epochs = []
    selected_history_row = None
    for expected_epoch, (item, trace_row) in enumerate(zip(epochs, trace_rows), start=1):
        metrics = item["validation_zero_miss"]
        if item.get("epoch") != expected_epoch or int(trace_row["epoch"]) != expected_epoch:
            raise AssertionError("MMRDR history and trace epochs must be consecutive and aligned")
        for name in (
            "n", "positive", "negative", "true_positive", "false_negative_clearance",
            "true_negative", "false_positive_referral",
        ):
            if int(trace_row[name]) != metrics[name]:
                raise AssertionError(f"MMRDR validation trace mismatch at epoch {expected_epoch}: {name}")
        for name in ("train_loss", "threshold", "sensitivity", "specificity", "refer_rate"):
            expected = item["train_loss"] if name == "train_loss" else metrics[name]
            if not math.isclose(float(trace_row[name]), float(expected), rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError(f"MMRDR validation trace mismatch at epoch {expected_epoch}: {name}")
        if metrics["positive"] != metrics["true_positive"] + metrics["false_negative_clearance"]:
            raise AssertionError(f"MMRDR validation positive counts do not reconcile at epoch {expected_epoch}")
        if metrics["negative"] != metrics["true_negative"] + metrics["false_positive_referral"]:
            raise AssertionError(f"MMRDR validation negative counts do not reconcile at epoch {expected_epoch}")
        if metrics["n"] != metrics["positive"] + metrics["negative"]:
            raise AssertionError(f"MMRDR validation total does not reconcile at epoch {expected_epoch}")
        if metrics["false_negative_clearance"] == 0:
            zero_miss_epochs.append(item)
        if expected_epoch == validation["selected_epoch"]:
            selected_history_row = item

    if len(zero_miss_epochs) != trace_meta["zero_miss_epochs"]:
        raise AssertionError("MMRDR zero-miss epoch count disagrees with the run card")
    zero_miss_specificities = [
        item["validation_zero_miss"]["specificity"] for item in zero_miss_epochs
    ]
    recorded_range = trace_meta["specificity_range"]
    observed_range = [min(zero_miss_specificities), max(zero_miss_specificities)]
    if any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(observed_range, recorded_range)):
        raise AssertionError("MMRDR zero-miss specificity range disagrees with the run card")
    if selected_history_row is None or history.get("best_epoch") != validation["selected_epoch"]:
        raise AssertionError("MMRDR selected epoch disagrees with source history")
    selected = selected_history_row["validation_zero_miss"]
    best_true_negative = max(item["validation_zero_miss"]["true_negative"] for item in zero_miss_epochs)
    if selected["false_negative_clearance"] != 0 or selected["true_negative"] != best_true_negative:
        raise AssertionError("MMRDR selected epoch is not the maximum-specificity zero-miss epoch")
    if selected["true_negative"] != trace_meta["selected_true_negative"] or selected["negative"] != trace_meta["selected_nonreferable"]:
        raise AssertionError("MMRDR selected validation counts disagree with the trace metadata")
    for key, metric_key in (
        ("true_positive", "true_positive"),
        ("false_negative", "false_negative_clearance"),
        ("true_negative", "true_negative"),
        ("false_positive", "false_positive_referral"),
    ):
        if validation[key] != selected[metric_key]:
            raise AssertionError(f"MMRDR selected validation result disagrees for {key}")
    if not math.isclose(validation["threshold"], selected["threshold"], abs_tol=1e-12):
        raise AssertionError("MMRDR validation threshold disagrees with selected history")
    if not math.isclose(
        validation["sensitivity"],
        validation["true_positive"] / (validation["true_positive"] + validation["false_negative"]),
        abs_tol=1e-12,
    ):
        raise AssertionError("MMRDR selected validation sensitivity does not reconcile")
    if not math.isclose(
        validation["specificity"],
        validation["true_negative"] / (validation["true_negative"] + validation["false_positive"]),
        abs_tol=1e-12,
    ):
        raise AssertionError("MMRDR selected validation specificity does not reconcile")

    test = run_card["official_test_result"]
    test_rows = run_card["dataset"]["official_test_rows"]
    tp, fn = test["true_positive"], test["false_negative"]
    tn, fp = test["true_negative"], test["false_positive"]
    if tp + fn + tn + fp != test_rows:
        raise AssertionError("MMRDR official-test confusion counts do not sum to manifest rows")
    if not math.isclose(test["sensitivity"], tp / (tp + fn), abs_tol=1e-12):
        raise AssertionError("MMRDR official-test sensitivity does not reconcile")
    if not math.isclose(test["specificity"], tn / (tn + fp), abs_tol=1e-12):
        raise AssertionError("MMRDR official-test specificity does not reconcile")
    ci = _wilson_interval(tn, tn + fp)
    if any(
        not math.isclose(observed, expected, abs_tol=1e-12)
        for observed, expected in zip(ci, test["specificity_wilson_95ci"])
    ):
        raise AssertionError("MMRDR official-test Wilson interval does not reconcile")
    sensitivity_ci = _wilson_interval(tp, tp + fn)
    if any(
        not math.isclose(observed, expected, abs_tol=1e-12)
        for observed, expected in zip(sensitivity_ci, test["sensitivity_wilson_95ci"])
    ):
        raise AssertionError("MMRDR official-test sensitivity Wilson interval does not reconcile")
    grade_actions = run_card["official_test_grade_actions"]
    if [item["grade"] for item in grade_actions] != list(range(5)):
        raise AssertionError("MMRDR grade-action table must contain grades 0 through 4 once")
    if sum(item["images"] for item in grade_actions) != test_rows:
        raise AssertionError("MMRDR grade-action counts do not sum to official-test rows")
    if sum(item["referred"] for item in grade_actions) != tp + fp or sum(item["cleared"] for item in grade_actions) != tn + fn:
        raise AssertionError("MMRDR grade-action totals do not reconcile with confusion counts")
    if sum(item["images"] for item in grade_actions if item["grade"] < 2) != tn + fp:
        raise AssertionError("MMRDR grades 0-1 do not reconcile with the nonreferable denominator")
    if sum(item["images"] for item in grade_actions if item["grade"] >= 2) != tp + fn:
        raise AssertionError("MMRDR grades 2-4 do not reconcile with the referable denominator")
    if test["clinical_certification"] or test["binary_action_assignment"] != 1.0:
        raise AssertionError("MMRDR certification/coverage boundary changed unexpectedly")

    return {
        "status": "PASS_MMRDR_AGGREGATE_AND_TRACE_INTEGRITY_NOT_INFERENCE_REPRODUCTION",
        "validation": {
            "epochs": len(epochs),
            "zero_miss_epochs": len(zero_miss_epochs),
            "selected_epoch": validation["selected_epoch"],
            "selected_true_negative": selected["true_negative"],
            "selected_nonreferable": selected["negative"],
            "specificity": selected["specificity"],
        },
        "official_test": {
            "rows": test_rows,
            "true_positive": tp,
            "false_negative": fn,
            "true_negative": tn,
            "false_positive": fp,
            "specificity": test["specificity"],
            "specificity_wilson_95ci": ci,
            "sensitivity_wilson_95ci": sensitivity_ci,
            "clinical_certification": False,
        },
        "sha256": {
            "run_card": sha256(run_card_path),
            "validation_summary": sha256(history_path),
            "validation_trace": sha256(trace_path),
        },
        "scope": "cross-checks tracked aggregate history, trace, run-card metrics, arithmetic, and hashes; does not run image inference, verify patient-level independence, or establish clinical validation",
    }


def audit_summary(root: Path) -> dict[str, Any]:
    """Check tracked aggregate records without claiming score-level replay."""
    root = root.resolve()
    required = {name: root / path for name, path in SUMMARY_RELATIVES.items()}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required tracked summary is missing:\n  " + "\n  ".join(missing))

    summaries = {name: read_json(path) for name, path in required.items()}
    replay = summaries["replay"]["same_cohort_replay_metrics"]
    observed = {
        "replay_cases": replay["n_cases"],
        "replay_gradable_negatives": replay["n_gradable_nonreferable"],
        "replay_true_negatives": replay["true_negatives"],
        "replay_referable_misses": replay["missed_referable"],
        "replay_ungradable_misses": replay["missed_ungradable"],
        "crossfit_TN": summaries["crossfit"]["crossfit"]["true_negatives"],
        "crossfit_misses": summaries["crossfit"]["crossfit"]["misses_referable_or_ungradable"],
        "nested_TN": summaries["nested"]["true_negatives"],
        "nested_misses": summaries["nested"]["misses_referable_or_ungradable"],
        "nested_margin_TN": summaries["nested_margin"]["true_negatives"],
        "nested_margin_misses": summaries["nested_margin"]["misses_referable_or_ungradable"],
        "transfer_union_TN": summaries["candidate_transfer"]["frozen_baseline_plus_candidate"]["true_negatives"],
        "transfer_union_misses": summaries["candidate_transfer"]["frozen_baseline_plus_candidate"]["misses_after_combination"],
    }
    expected = {
        "replay_cases": 981,
        "replay_gradable_negatives": 385,
        "replay_true_negatives": 270,
        "replay_referable_misses": 0,
        "replay_ungradable_misses": 0,
        "crossfit_TN": 268,
        "crossfit_misses": 3,
        "nested_TN": 248,
        "nested_misses": 5,
        "nested_margin_TN": 253,
        "nested_margin_misses": 6,
        "transfer_union_TN": 273,
        "transfer_union_misses": 2,
    }
    mismatches = {
        name: {"observed": observed[name], "expected": value}
        for name, value in expected.items()
        if observed[name] != value
    }
    if mismatches:
        raise AssertionError(json.dumps(mismatches, indent=2, sort_keys=True))

    mmrdr = audit_mmrdr_aggregate(root)
    return {
        "status": "PASS_TRACKED_AGGREGATE_INTEGRITY_ONLY_NOT_REPLAY",
        "scope": "checks tracked aggregate summaries and the MMRDR run-card/trace for internal consistency; does not execute the score policy, run image inference, validate row-level inputs, or establish independent clinical performance",
        "specificity": {"true_negatives": 270, "gradable_nonreferable": 385, "percent": 100 * 270 / 385},
        "headline_checks": {
            name: {"observed": observed[name], "expected": value}
            for name, value in expected.items()
        },
        "mmrdr_aggregate_audit": mmrdr,
        "sha256": {name: sha256(path) for name, path in required.items()},
    }


def audit_full_replay(root: Path) -> dict[str, Any]:
    root = root.resolve()
    replay_dir = root / REPLAY_RELATIVE
    policy_path = replay_dir / "policy.json"
    rows_path = replay_dir / "validation_replay.csv"
    required = {"policy": policy_path, "score_rows": rows_path}
    required.update({name: root / path for name, path in SUMMARY_RELATIVES.items()})
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot perform the local evidence audit; missing required files:\n  "
            + "\n  ".join(missing)
            + "\nObtain the source-derived score rows and saved summaries under their "
            "original dataset terms, then rerun this command."
        )

    sys.path.insert(0, str(root / "src"))
    from active_retina.exposed_expert_policy import FEATURES, score_case

    policy = read_json(policy_path)
    summaries = {name: read_json(root / path) for name, path in SUMMARY_RELATIVES.items()}
    replay = summaries["replay"]["same_cohort_replay_metrics"]
    source_summary = replay["per_source"]
    counts = {
        source: {
            "cases": 0,
            "negatives": 0,
            "true_negatives": 0,
            "missed_referable": 0,
            "missed_ungradable": 0,
            "assigned_actions": 0,
        }
        for source in FEATURES
    }
    rows_seen = 0
    with rows_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows_seen += 1
            source = row["source"]
            names = FEATURES[source]
            feature_names = names + (("quality",) if source == "DDR" else ())
            features = {name: float(row[name]) for name in feature_names}
            output = score_case(policy, source, features)
            if output["action"] != row["action"]:
                raise AssertionError(f"Action mismatch at input row {rows_seen}")
            if not np.isclose(float(output["risk_score"]), float(row["risk_score"]), rtol=0, atol=1e-12):
                raise AssertionError(f"Risk-score mismatch at input row {rows_seen}")

            result = counts[source]
            result["cases"] += 1
            result["assigned_actions"] += output["action"] in {"CLEAR", "REFER"}
            referable = row["referable"] == "1"
            ungradable = row["ungradable"] == "1"
            if not referable and not ungradable:
                result["negatives"] += 1
                result["true_negatives"] += output["action"] == "CLEAR"
            elif referable and output["action"] == "CLEAR":
                result["missed_referable"] += 1
            elif ungradable and output["action"] == "CLEAR":
                result["missed_ungradable"] += 1

    expected_source = {
        "DDR": {"cases": 625, "gradable_nonreferable": 206, "true_negatives": 134},
        "DeepDRiD": {"cases": 106, "gradable_nonreferable": 52, "true_negatives": 51},
        "DRTiD": {"cases": 250, "gradable_nonreferable": 127, "true_negatives": 85},
    }
    for source, expected in expected_source.items():
        got = counts[source]
        saved = source_summary[source]
        assert got["cases"] == expected["cases"] == saved["cases"]
        assert got["negatives"] == expected["gradable_nonreferable"] == saved["gradable_nonreferable"]
        assert got["true_negatives"] == expected["true_negatives"] == saved["true_negatives"]
        assert got["missed_referable"] == saved["missed_referable"] == 0
        assert got["missed_ungradable"] == saved["missed_ungradable"] == 0

    headline = {
        "replay_cases": (replay["n_cases"], 981),
        "replay_gradable_negatives": (replay["n_gradable_nonreferable"], 385),
        "replay_true_negatives": (replay["true_negatives"], 270),
        "replay_referable_misses": (replay["missed_referable"], 0),
        "replay_ungradable_misses": (replay["missed_ungradable"], 0),
        "action_assignment_fraction": (replay["action_coverage"], 1.0),
        "crossfit_TN": (summaries["crossfit"]["crossfit"]["true_negatives"], 268),
        "crossfit_misses": (summaries["crossfit"]["crossfit"]["misses_referable_or_ungradable"], 3),
        "nested_TN": (summaries["nested"]["true_negatives"], 248),
        "nested_misses": (summaries["nested"]["misses_referable_or_ungradable"], 5),
        "nested_margin_TN": (summaries["nested_margin"]["true_negatives"], 253),
        "nested_margin_misses": (summaries["nested_margin"]["misses_referable_or_ungradable"], 6),
        "transfer_union_TN": (
            summaries["candidate_transfer"]["frozen_baseline_plus_candidate"]["true_negatives"],
            273,
        ),
        "transfer_union_misses": (
            summaries["candidate_transfer"]["frozen_baseline_plus_candidate"]["misses_after_combination"],
            2,
        ),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in headline.items()
        if observed != expected
    }
    if mismatches:
        raise AssertionError(json.dumps(mismatches, indent=2, sort_keys=True))

    return {
        "status": "PASS_SCORE_LEVEL_REPLAY_AND_HEADLINE_COUNTS",
        "scope": "deterministic replay of supplied component scores; not image inference, model training, or external validation",
        "score_rows_recomputed": rows_seen,
        "specificity": {"true_negatives": 270, "gradable_nonreferable": 385, "percent": 100 * 270 / 385},
        "replay_missed_referable": 0,
        "replay_missed_ungradable": 0,
        "action_assignment_fraction_on_supplied_rows": 1.0,
        "headline_checks": {name: {"observed": value, "expected": expected} for name, (value, expected) in headline.items()},
        "per_source_recomputed": counts,
        "sha256": {name: sha256(path) for name, path in required.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="ActiveRetina repository root")
    parser.add_argument("--output", type=Path, help="Optional path for the machine-readable JSON report")
    parser.add_argument(
        "--mode",
        choices=("summary", "full-replay"),
        default="summary",
        help="summary checks tracked aggregate records only; full-replay requires the local score rows and policy",
    )
    args = parser.parse_args()
    report = audit_summary(args.root) if args.mode == "summary" else audit_full_replay(args.root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
