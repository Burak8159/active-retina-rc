"""Expanded positive-patient development analysis.

This intentionally does not create a certification cohort. It combines existing
patient-disjoint DeepDRiD cross-validation predictions with the inspected online
cohort candidate predictions to answer a narrower development question:

If more positive patients are available, can a zero-miss operating mode remain
plausible while retaining nonzero specificity?
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .formal_risk_control import binomial_upper


FULL_CV = Path("results/deepdrid_full_cv_predictions.csv")
ONLINE_CANDIDATES = Path("results/model_upgrade_harness_predictions.csv")
OUTPUT = Path("results/expanded_cohort_development_summary.json")
TABLE = Path("results/expanded_cohort_development_table.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def deepdrid_cv_rows(branch: str = "suffix2_initial") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_csv(FULL_CV):
        if row["branch"] != branch:
            continue
        key = row["eye_id"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "source": "deepdrid_full_cv_crossfit",
                "patient_id": f"deepdrid_cv_{row['patient_id']}",
                "eye_id": f"deepdrid_cv_{row['eye_id']}",
                "referable_dr": int(row["referable_dr"]),
                "score": float(row["two_view_score"]),
                "prediction_scope": "crossfit_or_fold_heldout",
            }
        )
    return rows


def online_candidate_rows(candidate: str = "logistic_c025_balanced") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(ONLINE_CANDIDATES):
        if row["candidate"] != candidate:
            continue
        rows.append(
            {
                "source": "deepdrid_online_inspected_candidate",
                "patient_id": f"deepdrid_online_{row['patient_id']}",
                "eye_id": f"deepdrid_online_{row['eye_id']}",
                "referable_dr": int(row["referable_dr"]),
                "score": float(row["referable_probability"]),
                "prediction_scope": "trained_on_full_development_evaluated_on_inspected_online",
            }
        )
    return rows


def patient_stats(rows: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        score = row["score"]
        if score <= low:
            pred = 0
            deferred = False
        elif score >= high:
            pred = 1
            deferred = False
        else:
            pred = None
            deferred = True
        by_patient[row["patient_id"]].append({**row, "prediction": pred, "deferred": deferred})

    positive = 0
    negative = 0
    decided_positive = 0
    decided_negative = 0
    missed = 0
    false_positive = 0
    decided_patients = 0
    for patient_rows in by_patient.values():
        is_positive = any(r["referable_dr"] == 1 for r in patient_rows)
        positive += int(is_positive)
        negative += int(not is_positive)
        decided_rows = [r for r in patient_rows if not r["deferred"]]
        decided_patients += int(bool(decided_rows))
        if is_positive:
            decided_ref = [r for r in patient_rows if r["referable_dr"] == 1 and not r["deferred"]]
            if decided_ref:
                decided_positive += 1
                missed += int(any(r["prediction"] == 0 for r in decided_ref))
        elif decided_rows:
            decided_negative += 1
            false_positive += int(any(r["prediction"] == 1 for r in decided_rows))

    ucb = binomial_upper(missed, decided_positive, 0.05) if decided_positive else float("nan")
    specificity = 1.0 - false_positive / decided_negative if decided_negative else float("nan")
    return {
        "low": low,
        "high": high,
        "n_patients": len(by_patient),
        "positive_patients": positive,
        "negative_patients": negative,
        "decided_positive_patients": decided_positive,
        "decided_negative_patients": decided_negative,
        "misses": missed,
        "false_positive_patients": false_positive,
        "specificity_decided_negative_patients": specificity,
        "coverage_patients": decided_patients / len(by_patient) if by_patient else float("nan"),
        "patient_missed_risk": missed / decided_positive if decided_positive else float("nan"),
        "patient_missed_risk_ucb_95": ucb,
        "certifies_5_percent_95_confidence_if_prospective": math.isfinite(ucb) and ucb <= 0.05,
    }


def search(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = np.asarray([r["score"] for r in rows], dtype=float)
    grid = sorted(set(float(x) for x in np.quantile(scores, np.linspace(0.0, 1.0, 201))))
    grid = sorted(set([0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0] + grid))
    out: list[dict[str, Any]] = []
    for low in grid:
        for high in grid:
            if low >= high:
                continue
            out.append(patient_stats(rows, low, high))
    return out


def best(candidates: list[dict[str, Any]], min_specificity: float) -> dict[str, Any] | None:
    viable = [
        c
        for c in candidates
        if c["misses"] == 0
        and c["decided_negative_patients"] > 0
        and c["specificity_decided_negative_patients"] >= min_specificity
    ]
    return max(
        viable,
        key=lambda c: (
            c["certifies_5_percent_95_confidence_if_prospective"],
            c["decided_positive_patients"],
            c["coverage_patients"],
            c["specificity_decided_negative_patients"],
        ),
        default=None,
    )


def cohort_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    return {
        "eyes": len(rows),
        "patients": len(by_patient),
        "positive_patients": sum(1 for rs in by_patient.values() if any(r["referable_dr"] == 1 for r in rs)),
        "referable_eyes": sum(int(r["referable_dr"]) for r in rows),
    }


def main() -> None:
    cv_rows = deepdrid_cv_rows()
    online_rows = online_candidate_rows()
    combined = cv_rows + online_rows
    candidates = search(combined)
    table = []
    for min_spec in [0.5, 0.67, 0.8, 0.9]:
        row = best(candidates, min_spec)
        if row:
            table.append({"minimum_specificity": min_spec, **row})
    write_csv(TABLE, table)
    summary = {
        "status": "EXPANDED_DEVELOPMENT_COHORT_ANALYZED_NOT_CERTIFICATION",
        "warning": (
            "This combines cross-validated development predictions and inspected online predictions. "
            "It increases evidence size for stress testing only; it is not an untouched product certificate."
        ),
        "sources": {
            "deepdrid_full_cv_crossfit": cohort_summary(cv_rows),
            "deepdrid_online_inspected_candidate": cohort_summary(online_rows),
            "combined": cohort_summary(combined),
        },
        "table": str(TABLE),
        "best_zero_miss_by_specificity": table,
        "interpretation": (
            "If a row has certifies_5_percent_95_confidence_if_prospective=true, the sample-size math is satisfied "
            "on this expanded development pool, but a real claim still requires freezing the artifact and repeating once "
            "on an untouched cohort."
        ),
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
