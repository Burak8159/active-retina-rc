"""Explain the exposed validation gap between the frozen recipe and score oracle.

This is a label-informed diagnostic over a repeatedly inspected cohort, not a
trained selector or an inference policy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audit_expert_score_oracle_union import (
    EXCLUDED_COLUMNS,
    _current_recipe_scores,
    _source_rows,
    align_scores,
    read_csv,
    rank_fraction,
    zero_miss_clearance,
)


def analyze(results: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = _source_rows(results)
    candidates = sorted(results.rglob("*.csv"))
    case_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"status": "EXPOSED_LABEL_INFORMED_DIAGNOSTIC_ONLY", "sources": {}}

    for source, cases in sources.items():
        eye_ids = sorted(cases)
        labels = np.asarray([int(cases[e].get("referable", cases[e].get("referable_dr"))) for e in eye_ids], dtype=bool)
        ungradable = np.asarray([int(cases[e].get("ungradable", 0) or 0) for e in eye_ids], dtype=bool)
        negatives = ~labels & ~ungradable
        current_score, forced = _current_recipe_scores(results, source, eye_ids)
        current_clear, current_threshold = zero_miss_clearance(
            labels, ungradable, rank_fraction(current_score), forced
        )

        candidate_clears: list[tuple[str, np.ndarray]] = []
        for path in candidates:
            try:
                aligned = align_scores(path, eye_ids)
            except (OSError, UnicodeError, csv.Error):
                continue
            for column, entry in aligned.items():
                score = entry["values"]
                clear, threshold = zero_miss_clearance(labels, ungradable, score, forced)
                if np.any(clear & (labels | ungradable)):
                    raise AssertionError(f"{path.name}:{column} cleared a required case")
                candidate_clears.append((f"{path.relative_to(results)}::{column}", clear))

        oracle = current_clear.copy()
        for _, clear in candidate_clears:
            oracle |= clear & ~forced

        additions = np.flatnonzero(oracle & negatives & ~current_clear)
        summary["sources"][source] = {
            "negative_count": int(negatives.sum()),
            "current_true_negatives": int((current_clear & negatives).sum()),
            "oracle_true_negatives": int((oracle & negatives).sum()),
            "oracle_only_additions": int(len(additions)),
            "current_threshold_label_derived": current_threshold,
            "candidate_score_count": len(candidate_clears),
        }
        for idx in additions:
            contributors = [name for name, clear in candidate_clears if clear[idx]]
            row = cases[eye_ids[idx]]
            case_rows.append({
                "source_dataset": source,
                "eye_id": eye_ids[idx],
                "patient_id": row.get("patient_id", ""),
                "grade": int(row.get("grade", -1)),
                "current_recipe_clears": False,
                "oracle_clears": True,
                "contributing_score_count": len(contributors),
                "contributing_scores": " | ".join(contributors),
            })

    summary["oracle_only_additions_pooled"] = len(case_rows)
    summary["interpretation"] = (
        "Each listed case is an exposed validation negative cleared by at least one score whose threshold "
        "was chosen from the same cohort's positive/ungradable labels. The list diagnoses complementarity; "
        "it does not show that a deployable selector can identify these cases."
    )
    return case_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results/oracle_gap_cases_20261013")
    args = parser.parse_args()
    rows, summary = analyze(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fields = ["source_dataset", "eye_id", "patient_id", "grade", "current_recipe_clears",
              "oracle_clears", "contributing_score_count", "contributing_scores"]
    with (args.output_dir / "oracle_only_negative_cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
