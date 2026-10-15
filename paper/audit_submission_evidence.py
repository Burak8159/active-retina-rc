#!/usr/bin/env python3
"""Recompute and audit the exposed 70.13% score-level replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

    return {
        "status": "PASS_TRACKED_AGGREGATE_INTEGRITY_ONLY_NOT_REPLAY",
        "scope": "checks arithmetic consistency of tracked summary files only; does not execute the policy or validate underlying row-level data",
        "specificity": {"true_negatives": 270, "gradable_nonreferable": 385, "percent": 100 * 270 / 385},
        "headline_checks": {
            name: {"observed": observed[name], "expected": value}
            for name, value in expected.items()
        },
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
