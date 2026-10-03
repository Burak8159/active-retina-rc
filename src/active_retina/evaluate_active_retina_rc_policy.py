from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .common import read_yaml_like, write_json
from .evaluate_gates import auc_binary, average_precision, brier_score


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "patient_id": row["patient_id"],
                    "eye_id": row["eye_id"],
                    "split": row.get("split", "unknown"),
                    "view_set": row["view_set"],
                    "grade": int(row["grade"]),
                    "referable_dr": int(row["referable_dr"]),
                    "referable_score": float(row["referable_score"]),
                    "fold_id": row["fold_id"],
                    "prediction_scope": row["prediction_scope"],
                }
            )
    return rows


def pair_predictions(rows: list[dict[str, Any]], first_view: str, second_view: str) -> dict[str, dict[str, Any]]:
    both_key = "+".join(sorted((first_view, second_view)))
    by_eye: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        by_eye[row["eye_id"]][row["view_set"]] = row
    paired = {}
    for eye_id, pred_by_set in by_eye.items():
        if first_view in pred_by_set and both_key in pred_by_set:
            first = pred_by_set[first_view]
            both = pred_by_set[both_key]
            paired[eye_id] = {
                "patient_id": first["patient_id"],
                "eye_id": eye_id,
                "split": first["split"],
                "grade": first["grade"],
                "referable_dr": first["referable_dr"],
                "first_score": first["referable_score"],
                "two_view_score": both["referable_score"],
                "first_scope": first["prediction_scope"],
                "two_view_scope": both["prediction_scope"],
            }
    return paired


def policy_decision(first_score: float, two_view_score: float, stop_low: float, stop_high: float, final_threshold: float) -> dict[str, Any]:
    if first_score <= stop_low:
        return {"action": "STOP", "images": 1, "score": first_score, "prediction": 0, "deferred": 0}
    if first_score >= stop_high:
        return {"action": "STOP", "images": 1, "score": first_score, "prediction": 1, "deferred": 0}
    if two_view_score >= final_threshold:
        return {"action": "ACQUIRE_SECOND_VIEW", "images": 2, "score": two_view_score, "prediction": 1, "deferred": 0}
    return {"action": "DEFER", "images": 2, "score": two_view_score, "prediction": None, "deferred": 1}


def evaluate_policy(examples: list[dict[str, Any]], stop_low: float, stop_high: float, final_threshold: float) -> dict[str, Any]:
    rows = []
    for ex in examples:
        decision = policy_decision(ex["first_score"], ex["two_view_score"], stop_low, stop_high, final_threshold)
        rows.append({**ex, **decision})

    decided = [r for r in rows if not r["deferred"]]
    referable = [r for r in rows if r["referable_dr"] == 1]
    decided_referable = [r for r in decided if r["referable_dr"] == 1]
    missed = [r for r in decided_referable if r["prediction"] == 0]
    false_pos = [r for r in decided if r["referable_dr"] == 0 and r["prediction"] == 1]
    nonref = [r for r in decided if r["referable_dr"] == 0]

    scores = [r["score"] for r in decided]
    labels = [r["referable_dr"] for r in decided]

    return {
        "n_total_eyes": len(rows),
        "n_decided_eyes": len(decided),
        "n_deferred_eyes": len(rows) - len(decided),
        "defer_rate": (len(rows) - len(decided)) / len(rows) if rows else float("nan"),
        "mean_images": mean(r["images"] for r in rows) if rows else float("nan"),
        "image_savings_vs_always_two_view": 1.0 - (mean(r["images"] for r in rows) / 2.0) if rows else float("nan"),
        "missed_referable_dr_risk": len(missed) / len(decided_referable) if decided_referable else float("nan"),
        "missed_referable_dr_fraction_all_referable": len(missed) / len(referable) if referable else float("nan"),
        "referable_decision_coverage": len(decided_referable) / len(referable) if referable else float("nan"),
        "sensitivity_among_decided": 1.0 - (len(missed) / len(decided_referable)) if decided_referable else float("nan"),
        "specificity_among_decided": 1.0 - (len(false_pos) / len(nonref)) if nonref else float("nan"),
        "auroc_among_decided": auc_binary(labels, scores) if len(set(labels)) == 2 else float("nan"),
        "auprc_among_decided": average_precision(labels, scores) if labels else float("nan"),
        "brier_among_decided": brier_score(labels, scores) if labels else float("nan"),
        "rows": rows,
    }


def bootstrap_metric(
    examples: list[dict[str, Any]],
    stop_low: float,
    stop_high: float,
    final_threshold: float,
    metric: str,
    n: int,
    seed: int,
) -> dict[str, float]:
    observed = evaluate_policy(examples, stop_low, stop_high, final_threshold)[metric]
    if not examples:
        return {"observed": observed, "ci_low": float("nan"), "ci_high": float("nan")}
    rng = random.Random(seed)
    clusters = defaultdict(list)
    for example in examples:
        clusters[example["patient_id"]].append(example)
    patients = sorted(clusters)
    reps = []
    for _ in range(n):
        sample = [eye for _ in patients for eye in clusters[rng.choice(patients)]]
        val = evaluate_policy(sample, stop_low, stop_high, final_threshold)[metric]
        if not math.isnan(val):
            reps.append(val)
    reps.sort()
    if not reps:
        return {"observed": observed, "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "observed": observed,
        "ci_low": reps[int(0.025 * (len(reps) - 1))],
        "ci_high": reps[int(0.975 * (len(reps) - 1))],
    }


def candidate_grid(scores: list[float], max_points: int) -> list[float]:
    if not scores:
        return [0.5]
    values = sorted(set(round(s, 6) for s in scores))
    if len(values) <= max_points:
        return values
    return [values[int(i * (len(values) - 1) / (max_points - 1))] for i in range(max_points)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--output")
    parser.add_argument("--max-grid-points", type=int, default=31)
    parser.add_argument("--min-specificity", type=float, default=0.50)
    parser.add_argument("--include-test", action="store_true", help="Explicit final evaluation only")
    args = parser.parse_args()

    cfg = read_yaml_like(Path(args.config))
    first_view = cfg["evaluation"]["fixed_first_view"]
    second_view = cfg["evaluation"]["second_view"]
    pred_path = Path(args.predictions or cfg["evaluation"]["prediction_input"])
    output = Path(args.output or str(Path(cfg["evaluation"]["output"]).with_name(Path(cfg["evaluation"]["output"]).stem + "_policy.json")))

    if not pred_path.exists():
        write_json(
            output,
            {
                "status": "BLOCKED_PREDICTIONS_MISSING",
                "required_prediction_file": str(pred_path),
                "reason": "Risk-controlled policy evaluation requires first-view and two-view predictions.",
            },
        )
        return

    rows = read_predictions(pred_path)
    allowed = {"crossfit", "heldout_development", "heldout_test"}
    if any(row["prediction_scope"] not in allowed for row in rows):
        raise SystemExit("Unsupported prediction scope; in-sample predictions are prohibited")
    paired = pair_predictions(rows, first_view, second_view)
    examples = list(paired.values())
    calibration = [x for x in examples if x["split"] == "calibration"]
    validation = [x for x in examples if x["split"] == "validation"]
    sealed_test = [x for x in examples if x["split"] == "sealed_test"] if args.include_test else []
    if not calibration or not validation:
        raise SystemExit("Policy evaluation requires calibration and validation predictions.")

    first_scores = [x["first_score"] for x in calibration]
    two_scores = [x["two_view_score"] for x in calibration]
    lows = candidate_grid(first_scores, args.max_grid_points)
    highs = candidate_grid(first_scores, args.max_grid_points)
    finals = candidate_grid(two_scores, args.max_grid_points)

    raw_targets = cfg.get("evaluation", {}).get("risk_control", {}).get("candidate_target_risks", [0.01, 0.02, 0.05, 0.1])
    if isinstance(raw_targets, dict):
        raw_targets = [0.01, 0.02, 0.05, 0.1]
    targets = [float(x) for x in raw_targets]

    n_boot = int(cfg["evaluation"]["bootstrap"]["n"])
    seed = int(cfg["evaluation"]["bootstrap"]["seed"])
    selected = []
    for target in targets:
        candidates = []
        for stop_low in lows:
            for stop_high in highs:
                if stop_low >= stop_high:
                    continue
                for final_threshold in finals:
                    metrics = evaluate_policy(calibration, stop_low, stop_high, final_threshold)
                    if (
                        metrics["missed_referable_dr_risk"] <= target
                        and (
                            math.isfinite(metrics["specificity_among_decided"])
                            and metrics["specificity_among_decided"] >= args.min_specificity
                        )
                    ):
                        candidates.append(
                            {
                                "stop_low": stop_low,
                                "stop_high": stop_high,
                                "final_threshold": final_threshold,
                                "calibration": {k: v for k, v in metrics.items() if k != "rows"},
                            }
                        )
        if not candidates:
            selected.append(
                {
                    "target_risk": target,
                    "status": "NO_EMPIRICALLY_FEASIBLE_POLICY",
                    "min_specificity": args.min_specificity,
                }
            )
            continue
        best = min(
            candidates,
            key=lambda c: (
                c["calibration"]["mean_images"],
                c["calibration"]["defer_rate"],
                -c["calibration"]["specificity_among_decided"],
            ),
        )
        val_metrics = evaluate_policy(validation, best["stop_low"], best["stop_high"], best["final_threshold"])
        test_metrics = evaluate_policy(sealed_test, best["stop_low"], best["stop_high"], best["final_threshold"])
        selected.append(
            {
                "target_risk": target,
                "status": "EMPIRICALLY_SELECTED_UNCERTIFIED",
                "min_specificity": args.min_specificity,
                **{k: best[k] for k in ["stop_low", "stop_high", "final_threshold"]},
                "calibration": best["calibration"],
                "validation": {k: v for k, v in val_metrics.items() if k != "rows"},
                "validation_ci": {
                    metric: bootstrap_metric(validation, best["stop_low"], best["stop_high"], best["final_threshold"], metric, n_boot, seed + offset)
                    for offset, metric in enumerate(("missed_referable_dr_risk", "mean_images", "image_savings_vs_always_two_view", "defer_rate"))
                },
                "sealed_test": {k: v for k, v in test_metrics.items() if k != "rows"},
                "sealed_test_ci": {
                    "missed_referable_dr_risk": bootstrap_metric(
                        sealed_test,
                        best["stop_low"],
                        best["stop_high"],
                        best["final_threshold"],
                        "missed_referable_dr_risk",
                        n_boot,
                        seed,
                    ),
                    "mean_images": bootstrap_metric(
                        sealed_test,
                        best["stop_low"],
                        best["stop_high"],
                        best["final_threshold"],
                        "mean_images",
                        n_boot,
                        seed + 1,
                    ),
                    "image_savings_vs_always_two_view": bootstrap_metric(
                        sealed_test,
                        best["stop_low"],
                        best["stop_high"],
                        best["final_threshold"],
                        "image_savings_vs_always_two_view",
                        n_boot,
                        seed + 2,
                    ),
                },
            }
        )

    viable = [
        s
        for s in selected
        if s["status"] == "EMPIRICALLY_SELECTED_UNCERTIFIED"
        and s["validation"]["image_savings_vs_always_two_view"] > 0.05
        and s["validation"]["missed_referable_dr_risk"] <= float(s["target_risk"])
    ]
    decision = "DEVELOPMENT_SIGNAL_UNCERTIFIED" if viable else "METHOD_ONLY_WEAK_EVIDENCE"
    if not any(s["status"] == "EMPIRICALLY_SELECTED_UNCERTIFIED" for s in selected):
        decision = "NO_GO"

    write_json(
        output,
        {
            "status": "ACTIVE_RETINA_RC_POLICY_EVALUATED",
            "decision": decision,
            "risk_certification": "NONE_THRESHOLD_SEARCH_USES_EMPIRICAL_ERROR_ONLY",
            "bootstrap_unit": "patient_id",
            "test_evaluation_enabled": args.include_test,
            "dataset": cfg["dataset"]["name"],
            "view_mapping": cfg["dataset"].get("view_mapping", {}),
            "view_mapping_status": cfg["dataset"].get("view_mapping_status", "UNKNOWN"),
            "fixed_first_view": first_view,
            "second_view": second_view,
            "n_examples": len(examples),
            "n_calibration": len(calibration),
            "n_validation": len(validation),
            "n_sealed_test": len(sealed_test),
            "min_specificity_constraint": args.min_specificity,
            "selected_policies": selected,
            "interpretation_limits": [
                "These are exploratory development results; no risk certificate is issued.",
                "Empirical calibration error does not certify risk; independent certification with selection correction is required.",
                "The historical test holdout was already inspected; it cannot serve as an untouched confirmatory cohort.",
                "DeepDRiD supports paired two-view acquisition, not four-field non-myopic planning.",
            ],
        },
    )


if __name__ == "__main__":
    main()
