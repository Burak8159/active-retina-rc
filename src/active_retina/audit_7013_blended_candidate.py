"""Reproduce a post-hoc, same-cohort rank blend beside the frozen 70.13 baseline."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


WEIGHTS = {
    "DDR": {
        "risk_score": 0.360,
        "aptos_margin": 0.268,
        "ordinal_attention_margin": 0.100,
        "margin_audit_margin": 0.271,
    },
    "DeepDRiD": {
        "risk_score": 0.693,
        "ordinal_quality": 0.018,
        "sourcebalanced_margin": 0.220,
        "fundus_risk": 0.068,
    },
    "DRTiD": {
        "risk_score": 0.032,
        "rankheavy_pair": 0.584,
        "fundus_margin": 0.076,
        "geometry_pair": 0.308,
    },
}


def run(input_path: Path, output_dir: Path) -> dict:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Input replay is empty")
    fields = set(rows[0])
    for source, weights in WEIGHTS.items():
        if not set(weights).issubset(fields):
            raise ValueError(f"Missing required component score(s) for {source}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    if set(grouped) != set(WEIGHTS):
        raise ValueError(f"Unexpected source set: {sorted(grouped)}")

    predictions: list[dict[str, object]] = []
    per_source: dict[str, dict[str, object]] = {}
    baseline_tn = candidate_tn = total_negative = total_misses = missed_ungradable = 0
    for source, source_rows in grouped.items():
        weights = WEIGHTS[source]
        ranks: dict[str, np.ndarray] = {}
        for column in weights:
            values = np.asarray([float(row[column]) for row in source_rows], dtype=np.float64)
            ranks[column] = rankdata(values, method="average") / (len(values) + 1.0)
        scores = sum(weights[name] * ranks[name] for name in weights)
        positive_scores = [
            float(scores[i]) for i, row in enumerate(source_rows)
            if int(row["referable"]) == 1 and int(row["ungradable"]) == 0
        ]
        if not positive_scores:
            raise ValueError(f"No gradable referable cases in {source}")
        threshold = min(positive_scores)
        neg = tn_base = tn_candidate = fn = n_pos = n_ungradable = 0
        for i, row in enumerate(source_rows):
            ungradable = int(row["ungradable"]) == 1
            referable = int(row["referable"]) == 1
            # Preserve the existing quality gate: ungradable cases are referred.
            action = "REFER" if ungradable or float(scores[i]) >= threshold else "CLEAR"
            if ungradable:
                n_ungradable += 1
                missed_ungradable += int(action != "REFER")
            elif referable:
                n_pos += 1
                fn += int(action == "CLEAR")
            else:
                neg += 1
                tn_candidate += int(action == "CLEAR")
                tn_base += int(row["action"] == "CLEAR")
            predictions.append({
                "source": source,
                "eye_id": row["eye_id"],
                "grade": int(row["grade"]),
                "referable": int(row["referable"]),
                "ungradable": int(ungradable),
                "candidate_score": float(scores[i]),
                "candidate_threshold": threshold,
                "baseline_action": row["action"],
                "candidate_action": action,
            })
        baseline_tn += tn_base
        candidate_tn += tn_candidate
        total_negative += neg
        total_misses += fn
        per_source[source] = {
            "gradable_negatives": neg,
            "candidate_true_negatives": tn_candidate,
            "candidate_specificity": tn_candidate / max(1, neg),
            "gradable_referable": n_pos,
            "candidate_missed_referable": fn,
            "ungradable_cases": n_ungradable,
            "candidate_missed_ungradable": 0,
            "posthoc_threshold": threshold,
            "weights": weights,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    summary = {
        "status": "POSTHOC_SAME_COHORT_DEVELOPMENT_ONLY_NOT_VALIDATED",
        "n_cases": len(rows),
        "baseline": {
            "true_negatives": baseline_tn,
            "gradable_negatives": total_negative,
            "specificity": baseline_tn / max(1, total_negative),
            "missed_referable": 0,
            "missed_ungradable": 0,
        },
        "candidate": {
            "true_negatives": candidate_tn,
            "gradable_negatives": total_negative,
            "specificity": candidate_tn / max(1, total_negative),
            "missed_referable": total_misses,
            "missed_ungradable": missed_ungradable,
            "delta_true_negatives": candidate_tn - baseline_tn,
        },
        "per_source": per_source,
        "provenance_warning": (
            "Candidate weights and per-source zero-miss thresholds were selected after inspecting this same 981-case cohort. "
            "The apparent lift is an optimistic, exposed-cohort diagnostic, not independent validation, a deployable frozen model, "
            "or clinical/autonomous evidence. The 70.13 baseline remains unchanged."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "weights.json").write_text(json.dumps(WEIGHTS, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.input, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
