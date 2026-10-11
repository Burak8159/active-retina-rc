"""Find an optimistic, validation-only ceiling across saved model predictions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_retina.train_quality_gated_autonomy import evaluate_policy, select_policy


SCORE_COLUMNS = ("risk", "ordinal_risk", "max_grade_margin", "grade_boundary_margin")


def evaluate_rows(
    rows: list[dict[str, str]], score_column: str, excluded_eye_ids: set[str] | None = None
) -> dict[str, Any]:
    excluded_eye_ids = excluded_eye_ids or set()
    usable = [
        row for row in rows
        if row.get("split") in {"valid", "validation"}
        and row.get("eye_id") not in excluded_eye_ids
    ]
    if not usable:
        raise ValueError("No validation prediction rows remain")
    pred = {
        "grade": np.asarray([int(row["grade"]) for row in usable]),
        "quality_known": np.asarray([int(row["quality_known"]) for row in usable]),
        "ungradable": np.asarray([int(row["ungradable"]) for row in usable]),
        "risk": np.asarray([float(row[score_column]) for row in usable]),
        "quality": np.asarray([float(row["quality"]) for row in usable]),
    }
    policy = select_policy(pred)
    metrics = evaluate_policy(pred, policy)
    return {"policy": policy, "metrics": metrics}


def audit(
    prediction_root: Path, excluded_eye_ids: set[str] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for path in sorted(prediction_root.rglob("validation_predictions.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row.get("source_dataset"):
                by_source[row["source_dataset"]].append(row)
        for source, source_rows in by_source.items():
            for score_column in SCORE_COLUMNS:
                if any(row.get(score_column) in (None, "") for row in source_rows if row.get("split") in {"valid", "validation"}):
                    continue
                try:
                    result = evaluate_rows(source_rows, score_column, excluded_eye_ids)
                except (KeyError, ValueError):
                    continue
                candidates.append({
                    "source": source,
                    "run": path.parent.name,
                    "score_column": score_column,
                    "prediction_file": str(path),
                    **result,
                })

    best_by_source: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        source = candidate["source"]
        score = candidate["metrics"]["specificity_gradable_nonreferable"]
        current = best_by_source.get(source)
        if current is None or score > current["metrics"]["specificity_gradable_nonreferable"]:
            best_by_source[source] = candidate
    if not best_by_source:
        raise ValueError("No eligible checkpoint/score candidates found")

    keys = (
        "gradable_nonreferable", "false_referrals_on_gradable_nonreferable",
        "gradable_referable_dr", "missed_referable_dr", "known_ungradable",
        "missed_known_ungradable", "n",
    )
    aggregate = {key: sum(item["metrics"][key] for item in best_by_source.values()) for key in keys}
    aggregate["specificity_gradable_nonreferable"] = (
        1.0 - aggregate["false_referrals_on_gradable_nonreferable"] / aggregate["gradable_nonreferable"]
    )
    aggregate["case_action_coverage"] = 1.0
    summary = {
        "status": "VALIDATION_ONLY_OPTIMISTIC_CHECKPOINT_CEILING",
        "candidate_count": len(candidates),
        "excluded_eye_ids": sorted(excluded_eye_ids or set()),
        "score_columns": list(SCORE_COLUMNS),
        "best_by_source": best_by_source,
        "aggregate": aggregate,
        "selection_boundary": (
            "Each source's checkpoint, score column, and policy are selected and evaluated on the same repeatedly inspected validation set. "
            "This is an optimistic development upper bound, not an independent performance estimate. "
            "Ungradable routing is computed from the model quality score, not hard-coded from the grade label."
        ),
    }
    return summary, candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", default="results")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--candidates-output", required=True)
    parser.add_argument("--exclude-eye-id", action="append", default=[])
    args = parser.parse_args()
    summary_path, candidates_path = Path(args.summary_output), Path(args.candidates_output)
    if summary_path.exists() or candidates_path.exists():
        raise SystemExit("Refusing to overwrite existing audit outputs")
    summary, candidates = audit(Path(args.prediction_root), set(args.exclude_eye_id))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("source", "run", "score_column", "specificity", "true_negatives", "negatives", "positives", "misses", "known_ungradable_missed", "policy")
    with candidates_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            metrics = item["metrics"]
            writer.writerow({
                "source": item["source"], "run": item["run"], "score_column": item["score_column"],
                "specificity": metrics["specificity_gradable_nonreferable"],
                "true_negatives": metrics["gradable_nonreferable"] - metrics["false_referrals_on_gradable_nonreferable"],
                "negatives": metrics["gradable_nonreferable"], "positives": metrics["gradable_referable_dr"],
                "misses": metrics["missed_referable_dr"], "known_ungradable_missed": metrics["missed_known_ungradable"],
                "policy": json.dumps(item["policy"], sort_keys=True),
            })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
