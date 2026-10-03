"""Search certification-style operating modes on frozen prediction files.

This is an analysis tool. If run on the online certification cohort, results are
post-hoc and cannot be called a prospective certificate. They are useful for
diagnosing whether a narrower autonomous mode is even plausible.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .formal_risk_control import binomial_upper


ONLINE = Path("results/deepdrid_online_certification_predictions.csv")
OUTPUT = Path("results/certification_mode_search_online.json")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            row["referable_probability"] = float(row["referable_probability"])
            rows.append(row)
        return rows


def patient_stats(rows: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    decisions = []
    for row in rows:
        score = row["referable_probability"]
        if score <= low:
            pred = 0
            deferred = False
        elif score >= high:
            pred = 1
            deferred = False
        else:
            pred = None
            deferred = True
        decisions.append({**row, "prediction": pred, "deferred": deferred})
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        by_patient[row["patient_id"]].append(row)
    positive = 0
    decided_positive = 0
    missed = 0
    decided_patients = 0
    for patient_rows in by_patient.values():
        if any(r["referable_dr"] == 1 for r in patient_rows):
            positive += 1
        if any(not r["deferred"] for r in patient_rows):
            decided_patients += 1
        decided_ref = [r for r in patient_rows if r["referable_dr"] == 1 and not r["deferred"]]
        if decided_ref:
            decided_positive += 1
            if any(r["prediction"] == 0 for r in decided_ref):
                missed += 1
    ucb = binomial_upper(missed, decided_positive, 0.05) if decided_positive else float("nan")
    return {
        "low": low,
        "high": high,
        "positive_patients": positive,
        "decided_positive_patients": decided_positive,
        "misses": missed,
        "patient_missed_risk": missed / decided_positive if decided_positive else float("nan"),
        "patient_missed_risk_ucb_95": ucb,
        "certifies_5_percent_95_confidence": math.isfinite(ucb) and ucb <= 0.05,
        "coverage_patients": decided_patients / len(by_patient) if by_patient else float("nan"),
        "defer_patients": 1.0 - (decided_patients / len(by_patient) if by_patient else float("nan")),
        "n_patients": len(by_patient),
    }


def main() -> None:
    rows = read_csv(ONLINE)
    scores = sorted(set(round(r["referable_probability"], 6) for r in rows))
    candidates = []
    for low in scores:
        for high in scores:
            if low >= high:
                continue
            candidates.append(patient_stats(rows, low, high))
    certified = [c for c in candidates if c["certifies_5_percent_95_confidence"]]
    zero_miss = [c for c in candidates if c["misses"] == 0]
    result = {
        "status": "POST_HOC_ONLINE_MODE_SEARCH_NOT_A_PROSPECTIVE_CERTIFICATE",
        "n_candidates": len(candidates),
        "any_certified_post_hoc": bool(certified),
        "best_certified_by_coverage": max(certified, key=lambda c: c["coverage_patients"], default=None),
        "best_zero_miss_by_decided_positive": max(zero_miss, key=lambda c: c["decided_positive_patients"], default=None),
        "top_zero_miss": sorted(zero_miss, key=lambda c: (-c["decided_positive_patients"], -c["coverage_patients"]))[:10],
        "interpretation": (
            "If best_zero_miss_by_decided_positive has fewer than 59 decided positive patients, "
            "then even post-hoc thresholding cannot meet the 95%/5% zero-miss certificate on this cohort."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
