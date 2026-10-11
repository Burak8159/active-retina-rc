"""Validation-only audit of known-source-specific zero-miss binary policies."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_retina.train_quality_gated_autonomy import evaluate_policy, select_policy


def source_policy_audit(rows: list[dict[str, str]], score_key: str = "ordinal_risk") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if score_key not in {"risk", "ordinal_risk", "max_grade_margin", "grade_boundary_margin"}:
        raise ValueError(f"Unsupported score key: {score_key}")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "valid":
            continue
        groups[row["source_dataset"]].append(row)
    if not groups:
        raise ValueError("No validation rows found")

    policies: dict[str, Any] = {}
    output_rows: list[dict[str, Any]] = []
    total_neg = total_clear = total_fn = total_missed_quality = total_cases = 0
    for source in sorted(groups):
        subset = groups[source]
        pred = {
            "grade": np.asarray([int(row["grade"]) for row in subset]),
            "risk": np.asarray([float(row[score_key]) for row in subset]),
            "quality": np.asarray([float(row["quality"]) for row in subset]),
            "quality_known": np.asarray([int(row["quality_known"]) for row in subset]),
            "ungradable": np.asarray([int(row["ungradable"]) for row in subset]),
        }
        policy = select_policy(pred)
        metrics = evaluate_policy(pred, policy)
        policies[source] = {"policy": policy, "metrics": metrics}
        total_neg += metrics["gradable_nonreferable"]
        total_clear += metrics["gradable_nonreferable"] - metrics["false_referrals_on_gradable_nonreferable"]
        total_fn += metrics["missed_referable_dr"]
        total_missed_quality += metrics["missed_known_ungradable"]
        total_cases += metrics["n"]
        for row in subset:
            referred = (
                float(row[score_key]) >= float(policy["risk_threshold"])
                or float(row["quality"]) >= float(policy["quality_threshold"])
            )
            output_rows.append({**row, "source_policy_action": "REFER" if referred else "CLEAR"})

    summary = {
        "status": "DEVELOPMENT_ONLY_SOURCE_CONDITIONED_POLICY_AUDIT",
        "score_key": score_key,
        "n_sources": len(groups),
        "n_cases": total_cases,
        "gradable_nonreferable_cleared": total_clear,
        "gradable_nonreferable_total": total_neg,
        "specificity_gradable_nonreferable": total_clear / max(1, total_neg),
        "missed_referable_dr": total_fn,
        "missed_known_ungradable": total_missed_quality,
        "case_action_coverage": 1.0,
        "source_policies": policies,
        "claim_boundary": "Each source threshold was selected and scored on the same repeatedly inspected validation set. Diagnostic only; not independent confirmation and not evidence of 100% population coverage.",
    }
    return summary, output_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--score-key", default="ordinal_risk")
    args = parser.parse_args()
    with Path(args.predictions).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary, output_rows = source_policy_audit(rows, args.score_key)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path = Path(args.predictions_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "source_policies"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
