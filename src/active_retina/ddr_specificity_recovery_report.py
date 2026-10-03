"""Summarize DDR specificity-recovery autonomy runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: float | int | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.1f}%"


def decision(summary: dict[str, Any], min_specificity: float) -> str:
    cert = summary.get("test_certificate", {})
    if not cert.get("certifies_5_percent_95_confidence", False):
        return "NO_GO_MISSED_RISK"
    if float(cert.get("specificity", 0.0)) < min_specificity:
        return "NO_GO_SPECIFICITY"
    return "GO_FULLER_SINGLE_IMAGE_AUTONOMY"


def decision_rank(label: str) -> int:
    if label == "GO_FULLER_SINGLE_IMAGE_AUTONOMY":
        return 3
    if label == "NO_GO_SPECIFICITY":
        return 2
    if label == "NO_GO_MISSED_RISK":
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-specificity", type=float, default=0.70)
    args = parser.parse_args()

    rows = []
    for item in args.summaries:
        path = Path(item)
        summary = load_json(path)
        cert = summary.get("test_certificate", {})
        best = summary.get("best_candidate", {})
        rows.append(
            {
                "run": str(path),
                "status": summary.get("status"),
                "specificity_recovery_decision": decision(summary, args.min_specificity),
                "candidate": best.get("candidate"),
                "threshold": cert.get("threshold"),
                "test_positive": cert.get("positive"),
                "missed_positive": cert.get("missed_positive"),
                "ucb95": cert.get("missed_risk_ucb95"),
                "specificity": cert.get("specificity"),
                "precision_referable_action": cert.get("precision_referable_action"),
                "referable_action_rate": cert.get("referable_action_rate"),
                "test_auroc": best.get("test_auroc"),
                "test_average_precision": best.get("test_average_precision"),
            }
        )

    best = sorted(
        rows,
        key=lambda row: (
            decision_rank(str(row["specificity_recovery_decision"])),
            float(row["specificity"] or 0.0),
            -float(row["missed_positive"] or 10**9),
        ),
        reverse=True,
    )[0]
    report = {
        "status": best["specificity_recovery_decision"],
        "min_specificity_target": args.min_specificity,
        "best_run": best,
        "runs": rows,
        "interpretation": (
            "Fuller single-image autonomy requires both the finite-sample zero-miss certificate "
            "and clinically useful specificity. A zero-miss refer-all solution remains a safety "
            "stress-test pass but fails product usefulness."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
