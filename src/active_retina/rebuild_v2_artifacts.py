"""Generate ActiveRetina Flagship Rebuild V2 development artifacts.

This script intentionally uses only existing development predictions. It does
not promote the manuscript to preprint or journal readiness.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .write_fast_paper_pack import write_csv, write_json


BRANCHES = [
    ("suffix2_initial", Path("results/deepdrid_train_only_development_predictions.csv"), "macula", "macula+optic_disc"),
    ("suffix1_initial", Path("results/deepdrid_swapped_train_only_development_predictions.csv"), "macula", "macula+optic_disc"),
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def auc_binary(labels: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum = sum(i + 1 for i, (_, y) in enumerate(pairs) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def brier(labels: list[int], scores: list[float]) -> float:
    if not labels:
        return float("nan")
    return mean((y - s) ** 2 for y, s in zip(labels, scores))


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(r["referable_dr"]) for r in rows]
    scores = [float(r["referable_score"]) for r in rows]
    preds = [1 if s >= 0.5 else 0 for s in scores]
    misses = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    false_pos = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    positives = sum(labels)
    negatives = len(labels) - positives
    return {
        "n": len(rows),
        "referable_n": positives,
        "mean_score": mean(scores) if scores else float("nan"),
        "auroc": auc_binary(labels, scores),
        "brier": brier(labels, scores),
        "miss_rate_at_0_5": misses / positives if positives else float("nan"),
        "false_positive_rate_at_0_5": false_pos / negatives if negatives else float("nan"),
    }


def grade_stratified(rows: list[dict[str, Any]], branch: str) -> list[dict[str, Any]]:
    out = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "validation":
            continue
        grouped[(row["view_set"], row["grade"])].append(row)
    for (view_set, grade), items in sorted(grouped.items()):
        out.append({"branch": branch, "view_set": view_set, "grade": grade, **summarize_group(items)})
    return out


def paired_delta(rows: list[dict[str, Any]], branch: str, first_view: str, paired_view: str) -> list[dict[str, Any]]:
    by_eye: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["split"] == "validation":
            by_eye[row["eye_id"]][row["view_set"]] = row
    out = []
    for eye_id, views in by_eye.items():
        if first_view not in views or paired_view not in views:
            continue
        first = views[first_view]
        paired = views[paired_view]
        out.append(
            {
                "branch": branch,
                "eye_id": eye_id,
                "patient_id": first["patient_id"],
                "grade": first["grade"],
                "referable_dr": first["referable_dr"],
                "first_score": float(first["referable_score"]),
                "paired_score": float(paired["referable_score"]),
                "paired_minus_first": float(paired["referable_score"]) - float(first["referable_score"]),
            }
        )
    return out


def summarize_delta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["branch"], row["grade"])].append(row)
    out = []
    for (branch, grade), items in sorted(grouped.items()):
        deltas = [r["paired_minus_first"] for r in items]
        out.append(
            {
                "branch": branch,
                "grade": grade,
                "n": len(items),
                "mean_paired_minus_first": mean(deltas),
                "median_paired_minus_first": sorted(deltas)[len(deltas) // 2],
                "fraction_increased_score": sum(1 for d in deltas if d > 0) / len(deltas) if deltas else float("nan"),
            }
        )
    return out


def main() -> None:
    grade_rows = []
    delta_rows = []
    branch_summaries = []
    for branch, path, first_view, paired_view in BRANCHES:
        rows = read_rows(path)
        grade_rows.extend(grade_stratified(rows, branch))
        deltas = paired_delta(rows, branch, first_view, paired_view)
        delta_rows.extend(deltas)
        validation = [r for r in rows if r["split"] == "validation"]
        branch_summaries.append(
            {
                "branch": branch,
                "prediction_file": str(path),
                "validation_rows": len(validation),
                "available_view_sets": sorted(set(r["view_set"] for r in validation)),
                "grade_counts_validation": dict(sorted({g: sum(1 for r in validation if r["grade"] == g and r["view_set"] == first_view) for g in set(r["grade"] for r in validation)}.items())),
                "five_class_secondary_status": "BLOCKED_CLASS_PROBABILITIES_MISSING",
                "five_class_secondary_reason": "Existing prediction files contain referable-DR scores and grade labels, but not five-class class probabilities or predicted grades.",
            }
        )

    delta_summary = summarize_delta(delta_rows)
    write_csv(Path("results/rebuild_v2_grade_stratified_referable.csv"), grade_rows)
    write_csv(Path("results/rebuild_v2_paired_score_deltas.csv"), delta_rows)
    write_csv(Path("results/rebuild_v2_paired_delta_by_grade.csv"), delta_summary)
    write_json(
        Path("results/rebuild_v2_status.json"),
        {
            "decision": "METHOD_DEVELOPMENT_CONTINUE",
            "source_of_truth": "docs/flagship_rebuild_v2.md",
            "publication_readiness": "NOT_READY_FOR_ARXIV_OR_IEEE",
            "branch_summaries": branch_summaries,
            "artifacts": [
                "results/rebuild_v2_grade_stratified_referable.csv",
                "results/rebuild_v2_paired_score_deltas.csv",
                "results/rebuild_v2_paired_delta_by_grade.csv",
            ],
            "next_required_model_output": "five-class predicted probabilities or predicted grades for each view set and split",
        },
    )


if __name__ == "__main__":
    main()
