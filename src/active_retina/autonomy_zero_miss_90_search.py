"""Search existing DDR scores for a true zero-miss / 90% specificity policy.

The earlier selective search used >=99% validation sensitivity. This script
uses the stricter gate the project actually needs: zero referable clearances on
validation, then reports the frozen policy once on test.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from active_retina.autonomy_failure_analysis import DDR_MODELS, write_csv


def read_model_rows() -> tuple[list[str], list[str], dict[str, dict[str, dict[str, Any]]]]:
    model_rows = {}
    for model, rel in DDR_MODELS.items():
        path = Path(rel)
        if not path.exists():
            continue
        rows = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                score = row.get("referable_probability", row.get("score"))
                rows[row["image_id"]] = {
                    "split": row["split"],
                    "image_id": row["image_id"],
                    "referable_dr": int(row["referable_dr"]),
                    "score": float(score),
                }
        model_rows[model] = rows
    common_ids = sorted(set.intersection(*(set(rows) for rows in model_rows.values())))
    return sorted(model_rows), common_ids, model_rows


def ranknorm(values: np.ndarray, splits: list[str]) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for split in sorted(set(splits)):
        idx = np.asarray([s == split for s in splits])
        vals = values[idx]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty_like(vals, dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, len(vals), endpoint=True)
        out[idx] = ranks
    return out


def metrics(y: np.ndarray, scores: np.ndarray, low: float, high: float) -> dict[str, Any]:
    clear = scores <= low
    refer = scores >= high
    auto = clear | refer
    pos = y == 1
    neg = y == 0
    false_clear = int((pos & clear).sum())
    true_clear = int((neg & clear).sum())
    false_refer = int((neg & refer).sum())
    true_refer = int((pos & refer).sum())
    return {
        "low": float(low),
        "high": float(high),
        "coverage": float(auto.mean()),
        "abstention": float((~auto).mean()),
        "false_negative_clearances": false_clear,
        "true_negative_clearances": true_clear,
        "false_positive_referrals": false_refer,
        "true_positive_referrals": true_refer,
        "sensitivity_all_cases_clear_is_negative": float(1.0 - false_clear / max(1, int(pos.sum()))),
        "specificity_among_auto_decisions": float(true_clear / max(1, true_clear + false_refer)),
        "negative_clearance_rate": float(true_clear / max(1, int(neg.sum()))),
        "referable_auto_rate": float((pos & refer).sum() / max(1, int(pos.sum()))),
        "nonreferable_refer_rate": float(false_refer / max(1, int(neg.sum()))),
    }


def select_policy(valid_y: np.ndarray, valid_s: np.ndarray, min_spec: float) -> dict[str, Any]:
    # Low threshold must be below the lowest positive validation score to ensure
    # zero validation false-negative clearance. We test every useful low from
    # observed scores plus a no-clear sentinel. High controls false referrals.
    scores = np.unique(np.quantile(valid_s, np.linspace(0, 1, 161))).tolist()
    scores = sorted(set([float(np.min(valid_s)) - 1e-9, *[float(x) for x in scores], float(np.max(valid_s)) + 1e-9]))
    best = None
    for low in scores:
        for high in scores:
            if low >= high:
                continue
            row = metrics(valid_y, valid_s, low, high)
            if row["false_negative_clearances"] != 0:
                continue
            if row["specificity_among_auto_decisions"] < min_spec:
                continue
            # Prioritize actual negative clearance, then total coverage, then
            # fewer false referrals. This blocks degenerate "refer everyone"
            # rows from looking successful.
            key = (
                row["negative_clearance_rate"],
                row["coverage"],
                -row["false_positive_referrals"],
                row["referable_auto_rate"],
            )
            if best is None or key > best[0]:
                best = (key, row)
    return best[1] if best else {"status": "NO_VALID_ZERO_MISS_90_POLICY"}


def select_best_validated(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [
        row for row in rows
        if "valid_negative_clearance_rate" in row
        and int(row["valid_false_negative_clearances"]) == 0
        and float(row["valid_specificity_among_auto_decisions"]) >= 0.90
        and float(row["valid_negative_clearance_rate"]) > 0
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (
            float(row["valid_negative_clearance_rate"]),
            float(row["valid_coverage"]),
            float(row["valid_specificity_among_auto_decisions"]),
        ),
    )


def main() -> None:
    models, image_ids, model_rows = read_model_rows()
    splits = [model_rows[models[0]][i]["split"] for i in image_ids]
    y = np.asarray([model_rows[models[0]][i]["referable_dr"] for i in image_ids], dtype=int)
    matrix = np.asarray([[model_rows[m][i]["score"] for m in models] for i in image_ids], dtype=float)
    candidates = {
        **{m: matrix[:, idx] for idx, m in enumerate(models)},
        "mean_all": matrix.mean(axis=1),
        "max_all": matrix.max(axis=1),
        "median_all": np.median(matrix, axis=1),
        "top2_mean_all": np.sort(matrix, axis=1)[:, -2:].mean(axis=1),
        "min_all": matrix.min(axis=1),
    }
    for name, values in list(candidates.items()):
        candidates[f"{name}_ranknorm"] = ranknorm(values, splits)

    valid_mask = np.asarray([s == "valid" for s in splits])
    test_mask = np.asarray([s == "test" for s in splits])
    rows = []
    min_specs = [0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 1.0]
    for min_spec in min_specs:
      for name, scores in candidates.items():
        selected = select_policy(y[valid_mask], scores[valid_mask], min_spec=min_spec)
        if "status" in selected:
            rows.append({"candidate": name, "min_validation_specificity": min_spec, "status": selected["status"]})
            continue
        test = metrics(y[test_mask], scores[test_mask], selected["low"], selected["high"])
        test_gate = (
            "DIAGNOSTIC_PASS"
            if test["false_negative_clearances"] == 0 and test["specificity_among_auto_decisions"] >= 0.90 and test["negative_clearance_rate"] > 0
            else "DIAGNOSTIC_FAIL"
        )
        rows.append(
            {
                "candidate": name,
                "min_validation_specificity": min_spec,
                "status": "DEVELOPMENT_ONLY",
                "test_gate_diagnostic": test_gate,
                **{f"valid_{k}": v for k, v in selected.items()},
                **{f"test_{k}": v for k, v in test.items()},
            }
        )
    write_csv(Path("results/autonomy_zero_miss_90_search.csv"), rows)
    best = select_best_validated(rows) or {"status": "NO_VALIDATION_CANDIDATE"}
    summary = {
        "status": "DEVELOPMENT_ONLY_TEST_EXPOSED",
        "n_candidates": len(rows),
        "n_validation_feasible_policies": sum(
            1 for r in rows if "valid_false_negative_clearances" in r
            and int(r["valid_false_negative_clearances"]) == 0
            and float(r["valid_specificity_among_auto_decisions"]) >= 0.90
            and float(r["valid_negative_clearance_rate"]) > 0
        ),
        "best": best,
        "table": "results/autonomy_zero_miss_90_search.csv",
        "selection_split": "validation",
        "test_interpretation": "Diagnostic only; this dataset split has been inspected in prior candidate searches and cannot establish confirmation.",
        "selection_rule": "validation-only candidate selection; zero-miss and >=90% validation-specificity policies rank by validation negative-clearance rate, then coverage",
    }
    Path("results/autonomy_zero_miss_90_search_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = [
        "# Zero-Miss / 90% Specificity Search\n\n",
        "This search tightens the gate from 99% sensitivity to exactly zero validation false-negative clearances, then applies the selected policy once to test.\n\n",
        "Status: **DEVELOPMENT ONLY; the test split is exposed and this is not a certificate.**\n\n",
        f"Best candidate: `{best.get('candidate')}`\n\n",
        f"Test false-negative clearances: **{best.get('test_false_negative_clearances')}**\n\n",
        f"Test specificity among automatic decisions: **{100 * float(best.get('test_specificity_among_auto_decisions', 0.0)):.1f}%**\n\n",
        f"Test negative-clearance rate: **{100 * float(best.get('test_negative_clearance_rate', 0.0)):.1f}%**\n\n",
        f"Test coverage: **{100 * float(best.get('test_coverage', 0.0)):.1f}%**\n",
    ]
    Path("docs/autonomy_zero_miss_90_search_report.md").write_text("".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
