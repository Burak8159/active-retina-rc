"""Reproduce exposed-validation rank-blend diagnostics for the DRTiD model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def average_rank_fraction(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks / max(1, len(values) - 1)


def zero_miss_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    threshold = float(np.min(scores[labels == 1]))
    negatives = labels == 0
    positives = labels == 1
    return {
        "threshold": threshold,
        "true_negatives": int(np.sum(scores[negatives] < threshold)),
        "negatives": int(np.sum(negatives)),
        "misses": int(np.sum(scores[positives] < threshold)),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-predictions", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--ceiling-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    new_rows = read_csv(Path(args.new_predictions))
    old_rows = [row for row in read_csv(Path(args.baseline_predictions))
                if row.get("source_dataset") == "drtid_paired"]
    new_by_id = {row["eye_id"]: row for row in new_rows}
    old_by_id = {row["eye_id"]: row for row in old_rows}
    if len(new_by_id) != len(new_rows) or len(old_by_id) != len(old_rows):
        raise ValueError("Duplicate eye IDs in one of the prediction files")
    if set(new_by_id) != set(old_by_id):
        raise ValueError("New model and baseline do not contain identical DRTiD eyes")
    eyes = sorted(new_by_id)
    labels = np.asarray([int(new_by_id[eye]["referable_dr"]) for eye in eyes], dtype=np.int64)
    for eye in eyes:
        old_label = int(old_by_id[eye].get("referable", old_by_id[eye].get("referable_dr", -1)))
        if old_label != int(new_by_id[eye]["referable_dr"]):
            raise ValueError(f"Label mismatch for eye {eye}")

    new_scores = {
        name: np.asarray([float(new_by_id[eye][name]) for eye in eyes])
        for name in ("paired", "max_view", "mean_view")
    }
    old_scores = {
        name: np.asarray([float(old_by_id[eye][name]) for eye in eyes])
        for name in ("risk", "ordinal_risk", "max_grade_margin", "grade_boundary_margin")
    }
    ceiling = json.loads(Path(args.ceiling_json).read_text(encoding="utf-8"))
    source_results = ceiling["best_by_source"]
    other_tn = sum(
        int(item["metrics"]["gradable_nonreferable"]
            - item["metrics"]["false_referrals_on_gradable_nonreferable"])
        for source, item in source_results.items() if source != "drtid_paired"
    )
    pooled_negatives = int(ceiling["aggregate"]["gradable_nonreferable"])
    rows = []
    for new_name, new in new_scores.items():
        new_rank = average_rank_fraction(new)
        for old_name, old in old_scores.items():
            old_rank = average_rank_fraction(old)
            for weight in np.linspace(0, 1, 101):
                score = weight * new_rank + (1 - weight) * old_rank
                metrics = zero_miss_metrics(labels, score)
                pooled_tn = other_tn + int(metrics["true_negatives"])
                rows.append({
                    "new_score": new_name,
                    "baseline_score": old_name,
                    "new_score_weight": float(weight),
                    **metrics,
                    "pooled_true_negatives": pooled_tn,
                    "pooled_negatives": pooled_negatives,
                    "pooled_specificity": pooled_tn / pooled_negatives,
                })
    rows.sort(key=lambda row: (
        int(row["true_negatives"]),
        float(row["new_score_weight"]),
    ), reverse=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "EXPOSED_VALIDATION_BLEND_SEARCH_NOT_CONFIRMATORY",
        "boundary": "101 weights per score pair were evaluated on the previously inspected DRTiD validation cohort. The maximum is an optimistic exploratory upper bound, not a frozen or cross-fitted policy.",
        "n": len(eyes),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "pairs_tested": len(rows),
        "best_on_this_validation": rows[0],
        "baseline_drtid_tn": int(source_results["drtid_paired"]["metrics"]["gradable_nonreferable"]
                                  - source_results["drtid_paired"]["metrics"]["false_referrals_on_gradable_nonreferable"]),
        "other_source_tn": other_tn,
        "required_pooled_tn_for_65_percent": int(np.ceil(0.65 * pooled_negatives)),
        "additional_tn_needed_after_best": max(
            0, int(np.ceil(0.65 * pooled_negatives)) - int(rows[0]["pooled_true_negatives"])
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "rank_blend_grid.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
