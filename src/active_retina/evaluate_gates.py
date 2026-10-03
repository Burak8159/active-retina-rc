from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .common import FIELDS, all_nonempty_subsets, read_yaml_like, write_json


def auc_binary(y_true: list[int], score: list[float]) -> float:
    pairs = sorted(zip(score, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = sum(i + 1 for i, (_, y) in enumerate(pairs) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(y_true: list[int], score: list[float]) -> float:
    if not y_true or sum(y_true) == 0:
        return float("nan")
    pairs = sorted(zip(score, y_true), key=lambda x: x[0], reverse=True)
    tp = 0
    precisions = []
    for i, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            tp += 1
            precisions.append(tp / i)
    return sum(precisions) / sum(y_true)


def brier_score(y_true: list[int], score: list[float]) -> float:
    if not y_true:
        return float("nan")
    return mean((float(y) - float(s)) ** 2 for y, s in zip(y_true, score))


def sensitivity_specificity(y_true: list[int], score: list[float], threshold: float = 0.5) -> tuple[float, float]:
    tp = sum(1 for y, s in zip(y_true, score) if y == 1 and s >= threshold)
    fn = sum(1 for y, s in zip(y_true, score) if y == 1 and s < threshold)
    tn = sum(1 for y, s in zip(y_true, score) if y == 0 and s < threshold)
    fp = sum(1 for y, s in zip(y_true, score) if y == 0 and s >= threshold)
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return sens, spec


def metrics_for(values_by_patient: dict[str, tuple[int, float]]) -> dict[str, float]:
    patients = sorted(values_by_patient)
    y_true = [values_by_patient[p][0] for p in patients]
    score = [values_by_patient[p][1] for p in patients]
    sens, spec = sensitivity_specificity(y_true, score)
    return {
        "auroc": auc_binary(y_true, score),
        "auprc": average_precision(y_true, score),
        "brier": brier_score(y_true, score),
        "sensitivity_at_0_5": sens,
        "specificity_at_0_5": spec,
        "n_patients": len(patients),
    }


def bootstrap_ci(values_by_patient: dict[str, tuple[int, float]], n: int, seed: int, metric: str = "auroc") -> tuple[float, float, float]:
    patients = sorted(values_by_patient)
    observed = metrics_for(values_by_patient)[metric]
    rng = random.Random(seed)
    reps = []
    for _ in range(n):
        sampled = [rng.choice(patients) for _ in patients]
        reps.append(metrics_for({f"{i}_{p}": values_by_patient[p] for i, p in enumerate(sampled)})[metric])
    reps = sorted(x for x in reps if not math.isnan(x))
    if not reps:
        return observed, float("nan"), float("nan")
    lo = reps[int(0.025 * (len(reps) - 1))]
    hi = reps[int(0.975 * (len(reps) - 1))]
    return observed, lo, hi


def load_predictions(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row.get("eye_id") or row["patient_id"]
            data[sample_id][row["field"]] = {
                "grade": int(row["grade"]),
                "referable_score": float(row["referable_score"]),
            }
    return data


def subset_score(fields: tuple[str, ...], field_rows: dict[str, dict[str, float]]) -> float:
    return mean(field_rows[f]["referable_score"] for f in fields if f in field_rows)


def label_optimal_score(label: int, scores: list[float]) -> float:
    return max(scores) if label == 1 else min(scores)


def greedy_sequence_score(start: tuple[str, ...], budget: int, fields: tuple[str, ...], label: int, field_rows: dict[str, dict[str, float]]) -> tuple[tuple[str, ...], float]:
    acquired = list(start)
    while len(acquired) < budget:
        candidates = [f for f in fields if f not in acquired and f in field_rows]
        if not candidates:
            break
        scored = []
        for f in candidates:
            subset = tuple(acquired + [f])
            scored.append((subset_score(subset, field_rows), f))
        next_field = max(scored)[1] if label == 1 else min(scored)[1]
        acquired.append(next_field)
    final = tuple(acquired)
    return final, subset_score(final, field_rows)


def nonmyopic_sequence_score(start: tuple[str, ...], budget: int, fields: tuple[str, ...], label: int, field_rows: dict[str, dict[str, float]]) -> tuple[tuple[str, ...], float]:
    from itertools import combinations

    remaining = [f for f in fields if f not in start and f in field_rows]
    need = budget - len(start)
    if need <= 0:
        return start, subset_score(start, field_rows)
    candidates = [tuple(start + combo) for combo in combinations(remaining, need)]
    scored = [(subset_score(s, field_rows), s) for s in candidates]
    score, subset = max(scored) if label == 1 else min(scored)
    return subset, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", default="results/sealed_field_predictions.csv")
    args = parser.parse_args()
    cfg = read_yaml_like(Path(args.config))
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit("Missing sealed predictions. Train/import predictions before A1/A2 evaluation.")

    preds = load_predictions(pred_path)
    threshold = int(cfg["dataset"]["referable_threshold"])
    fields = tuple(cfg["dataset"].get("field_order", FIELDS))
    first_field = cfg["evaluation"].get("fixed_first_field", fields[0])
    subsets = all_nonempty_subsets(fields)
    rows = []
    for subset in subsets:
        values = {}
        for patient, by_field in preds.items():
            if all(f in by_field for f in subset):
                grade = next(iter(by_field.values()))["grade"]
                values[patient] = (1 if grade >= threshold else 0, subset_score(subset, by_field))
        metric_values = metrics_for(values)
        obs, lo, hi = bootstrap_ci(values, int(cfg["evaluation"]["bootstrap"]["n"]), int(cfg["evaluation"]["bootstrap"]["seed"]))
        rows.append(
            {
                "subset": "+".join(subset),
                "budget": len(subset),
                "fixed_first": first_field in subset,
                "auroc_ci_low": lo,
                "auroc_ci_high": hi,
                **metric_values,
            }
        )

    best_fixed = {}
    best_fixed_first = {}
    for budget in range(1, len(fields) + 1):
        candidates = [r for r in rows if r["budget"] == budget]
        best_fixed[str(budget)] = max(candidates, key=lambda r: r["auroc"])
        fixed_first_candidates = [r for r in candidates if r["fixed_first"]]
        if fixed_first_candidates:
            best_fixed_first[str(budget)] = max(fixed_first_candidates, key=lambda r: r["auroc"])

    oracle_by_budget = {}
    for budget in range(1, len(fields) + 1):
        patient_values = {}
        budget_subsets = [s for s in subsets if len(s) == budget]
        for patient, by_field in preds.items():
            if not all(f in by_field for f in fields):
                continue
            grade = next(iter(by_field.values()))["grade"]
            label = 1 if grade >= threshold else 0
            scores = [subset_score(s, by_field) for s in budget_subsets]
            oracle_score = label_optimal_score(label, scores)
            patient_values[patient] = (label, oracle_score)
        obs, lo, hi = bootstrap_ci(patient_values, int(cfg["evaluation"]["bootstrap"]["n"]), int(cfg["evaluation"]["bootstrap"]["seed"]))
        oracle_by_budget[str(budget)] = {"auroc": obs, "ci_low": lo, "ci_high": hi, **metrics_for(patient_values)}

    planning = {}
    start = (first_field,)
    for budget in range(2, len(fields) + 1):
        greedy_values = {}
        nonmyopic_values = {}
        sequence_counts = {"greedy": defaultdict(int), "nonmyopic": defaultdict(int)}
        for patient, by_field in preds.items():
            if not all(f in by_field for f in fields):
                continue
            grade = next(iter(by_field.values()))["grade"]
            label = 1 if grade >= threshold else 0
            greedy_subset, greedy_score = greedy_sequence_score(start, budget, fields, label, by_field)
            nonmyopic_subset, nonmyopic_score = nonmyopic_sequence_score(start, budget, fields, label, by_field)
            greedy_values[patient] = (label, greedy_score)
            nonmyopic_values[patient] = (label, nonmyopic_score)
            sequence_counts["greedy"]["+".join(greedy_subset)] += 1
            sequence_counts["nonmyopic"]["+".join(nonmyopic_subset)] += 1
        greedy_metrics = metrics_for(greedy_values)
        nonmyopic_metrics = metrics_for(nonmyopic_values)
        planning[str(budget)] = {
            "greedy": greedy_metrics,
            "nonmyopic": nonmyopic_metrics,
            "nonmyopic_minus_greedy_auroc": nonmyopic_metrics["auroc"] - greedy_metrics["auroc"],
            "sequence_counts": {k: dict(v) for k, v in sequence_counts.items()},
        }

    thresholds = cfg["evaluation"]["decision_thresholds"]
    full_budget = str(len(fields))
    multiview_gap = best_fixed[full_budget]["auroc"] - best_fixed["1"]["auroc"]
    individualization_gaps = {
        str(k): oracle_by_budget[str(k)]["auroc"] - best_fixed[str(k)]["auroc"]
        for k in range(1, len(fields) + 1)
        if str(k) in oracle_by_budget and str(k) in best_fixed
    }
    max_individualization_gap = max(individualization_gaps.values())
    max_planning_gap = max((v["nonmyopic_minus_greedy_auroc"] for v in planning.values()), default=float("nan"))

    if multiview_gap < thresholds["min_four_view_minus_best_one_view_auroc"]:
        decision = "NO-GO"
        decision_reason = "F1 failed: full multi-field performance does not materially exceed best one-field performance."
    elif max_individualization_gap < thresholds["min_patient_oracle_minus_best_fixed_same_budget_auroc"]:
        decision = "NO-GO"
        decision_reason = "F2 failed: hindsight patient-specific subsets do not materially beat fixed subsets."
    elif max_planning_gap >= thresholds["min_nonmyopic_minus_greedy_expected_utility"]:
        decision = "GO-CCG"
        decision_reason = "F1/F2 pass and F3 shows a material non-myopic planning gap."
    else:
        decision = "GO-MYOPIC"
        decision_reason = "F1/F2 pass but F3 does not show material non-myopic value over greedy selection."

    output = Path(cfg["evaluation"].get("output", "results/A1_A2_metrics.json"))
    write_json(
        output,
        {
            "status": "A0_A6_EVALUATED_FROM_PROVIDED_PREDICTIONS",
            "fixed_first_field": first_field,
            "field_order": fields,
            "subset_metrics": rows,
            "best_fixed": best_fixed,
            "best_fixed_with_first_field": best_fixed_first,
            "hindsight_oracle": oracle_by_budget,
            "individualization_gaps": individualization_gaps,
            "planning_oracle": planning,
            "gate_summary": {
                "F1_multiview_gap_auroc": multiview_gap,
                "F2_max_individualization_gap_auroc": max_individualization_gap,
                "F3_max_nonmyopic_minus_greedy_auroc": max_planning_gap,
            },
            "decision": decision,
            "decision_reason": decision_reason,
            "warning": "This decision is valid only for the supplied prediction file. Fixture predictions are infrastructure checks, not empirical paper evidence.",
        },
    )


if __name__ == "__main__":
    main()
