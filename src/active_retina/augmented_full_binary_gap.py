"""Test full-coverage binary autonomy with the augmented signal family.

This is a falsification script for the user's hardest target: every image gets
a binary nonreferable/referable diagnosis, with zero false negatives and high
specificity. Thresholds are selected on validation and applied once to test.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from active_retina.autonomy_extra_signal_search import read_extra_rows
from active_retina.autonomy_zero_miss_90_search import ranknorm, read_model_rows


def confusion(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = scores >= threshold
    pos = y == 1
    neg = y == 0
    tp = int((pos & pred).sum())
    fn = int((pos & ~pred).sum())
    tn = int((neg & ~pred).sum())
    fp = int((neg & pred).sum())
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": float(tp / max(1, int(pos.sum()))),
        "specificity": float(tn / max(1, int(neg.sum()))),
    }


def select_zero_miss_threshold(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    pos_scores = scores[y == 1]
    if len(pos_scores) == 0:
        return {"status": "NO_POSITIVES"}
    threshold = float(np.min(pos_scores)) - 1e-12
    return confusion(y, scores, threshold)


def select_best_validated(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [
        row for row in rows
        if "valid_specificity" in row and int(row["valid_fn"]) == 0
    ]
    if not feasible:
        return None
    return max(feasible, key=lambda row: (float(row["valid_specificity"]), -int(row["valid_fn"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-predictions", required=True)
    parser.add_argument("--extra-name", default="clementp_efficientnet_b5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-weight-candidates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()

    models, base_ids, model_rows = read_model_rows()
    model_rows[args.extra_name] = read_extra_rows(Path(args.extra_predictions), args.extra_name)
    models = sorted(model_rows)
    image_ids = sorted(set(base_ids).intersection(model_rows[args.extra_name]))
    splits = [model_rows[models[0]][image_id]["split"] for image_id in image_ids]
    y = np.asarray([model_rows[models[0]][image_id]["referable_dr"] for image_id in image_ids], dtype=int)
    matrix = np.asarray([[model_rows[m][image_id]["score"] for m in models] for image_id in image_ids], dtype=float)
    candidates: dict[str, np.ndarray] = {
        **{m: matrix[:, idx] for idx, m in enumerate(models)},
        "mean_augmented": matrix.mean(axis=1),
        "median_augmented": np.median(matrix, axis=1),
        "max_augmented": matrix.max(axis=1),
        "min_augmented": matrix.min(axis=1),
        "top2_mean_augmented": np.sort(matrix, axis=1)[:, -2:].mean(axis=1),
        "top3_mean_augmented": np.sort(matrix, axis=1)[:, -3:].mean(axis=1),
    }
    rng = np.random.default_rng(args.seed)
    for idx in range(args.random_weight_candidates):
        weights = rng.dirichlet(np.ones(len(models)))
        candidates[f"random_weight_{idx:04d}"] = np.sum(matrix * weights[None, :], axis=1)
    for name, values in list(candidates.items()):
        candidates[f"{name}_ranknorm"] = ranknorm(values, splits)

    valid_mask = np.asarray([s == "valid" for s in splits])
    test_mask = np.asarray([s == "test" for s in splits])
    rows = []
    for name, scores in candidates.items():
        selected = select_zero_miss_threshold(y[valid_mask], scores[valid_mask])
        if "status" in selected:
            rows.append({"candidate": name, **selected})
            continue
        test = confusion(y[test_mask], scores[test_mask], float(selected["threshold"]))
        test_gate = "DIAGNOSTIC_PASS" if test["fn"] == 0 and test["specificity"] >= 0.90 else "DIAGNOSTIC_FAIL"
        row = {
            "candidate": name,
            "status": "DEVELOPMENT_ONLY",
            "test_gate_diagnostic": test_gate,
            **{f"valid_{k}": v for k, v in selected.items()},
            **{f"test_{k}": v for k, v in test.items()},
        }
        rows.append(row)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "full_binary_gap_table.csv", rows)
    best = select_best_validated(rows) or {"status": "NO_VALIDATION_CANDIDATE"}
    summary = {
        "status": "DEVELOPMENT_ONLY_TEST_EXPOSED",
        "selection_split": "validation",
        "test_interpretation": "Diagnostic only; this dataset split has been inspected in prior candidate searches and cannot establish confirmation.",
        "best": best,
        "n_candidates": len(rows),
        "table": str(out / "full_binary_gap_table.csv"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
