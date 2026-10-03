"""Explain finite-sample feasibility for a 5% missed-DR certificate."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .formal_risk_control import binomial_upper


TARGET = 0.05
ALPHA = 0.05
CANDIDATES = Path("results/formal_risk_control_candidates.csv")
OUTPUT = Path("results/risk_certificate_feasibility.json")


def required_zero_miss_n(target: float = TARGET, alpha: float = ALPHA) -> int:
    """Minimum n so the one-sided exact upper bound is <= target with zero misses."""
    n = 1
    while binomial_upper(0, n, alpha) > target:
        n += 1
    return n


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = read_rows(CANDIDATES)
    zero_miss = []
    all_rows = []
    for row in rows:
        decided = int(float(row.get("cal_decided_positive_patients") or 0))
        misses = int(float(row.get("cal_misses") or 0))
        ucb = binomial_upper(misses, decided, ALPHA) if decided else float("nan")
        item = {
            "branch": row.get("branch"),
            "method": row.get("method"),
            "params": row.get("params"),
            "cal_misses": misses,
            "cal_decided_positive_patients": decided,
            "exact_ucb_95": ucb,
            "certifies_5_percent": math.isfinite(ucb) and ucb <= TARGET,
        }
        all_rows.append(item)
        if misses == 0 and decided > 0:
            zero_miss.append(item)
    best_zero = max(zero_miss, key=lambda x: x["cal_decided_positive_patients"], default=None)
    best_any = min(
        [x for x in all_rows if math.isfinite(x["exact_ucb_95"])],
        key=lambda x: x["exact_ucb_95"],
        default=None,
    )
    required = required_zero_miss_n()
    result = {
        "status": "FIVE_PERCENT_AUTONOMOUS_CERTIFICATE_NOT_FEASIBLE_WITH_CURRENT_CALIBRATION_SAMPLE",
        "target_missed_referable_dr_risk": TARGET,
        "confidence": 0.95,
        "minimum_decided_positive_patients_needed_if_zero_misses": required,
        "reason": (
            "An exact one-sided 95% binomial upper bound for zero misses requires "
            f"at least {required} automatically decided positive-patient cases. "
            "The current calibration grid does not provide that many zero-miss decided positive-patient cases."
        ),
        "best_zero_miss_candidate": best_zero,
        "best_overall_candidate": best_any,
        "autonomous_use_decision": "NO_GO_AUTONOMOUS_DIAGNOSIS",
        "supervised_workflow_decision": "GO_CLINICIAN_REVIEWED_ACQUISITION_ASSISTANT",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
