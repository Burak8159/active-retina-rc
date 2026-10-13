"""Fit calibration-only fusion of paired-view DR scores and evaluate once.

The calibration CSV must be patient-disjoint from the evaluation CSV. This is
development analysis only; it cannot certify clinical risk on an inspected set.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FEATURES = ("paired", "macula", "optic_disc")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def matrix(rows: list[dict[str, str]], columns: tuple[str, ...]) -> np.ndarray:
    values = np.asarray([[float(row[name]) for name in columns] for row in rows], dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Prediction inputs must be a finite rectangular matrix")
    return values


def percentile_map(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map each score to its calibration empirical CDF, preserving ties."""
    return np.column_stack([
        np.searchsorted(np.sort(reference[:, j]), values[:, j], side="right") / len(reference)
        for j in range(reference.shape[1])
    ])


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, steps: int = 3000):
    mean = x.mean(axis=0)
    scale = x.std(axis=0).clip(1e-6)
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    # Full-batch gradient descent with a conservative fixed step size.
    for _ in range(steps):
        logits = np.clip(design @ beta, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        grad = design.T @ (prob - y) / len(y) + l2 * (penalty @ beta)
        beta -= 0.05 * grad
    return {"mean": mean, "scale": scale, "beta": beta, "l2": l2}


def apply_model(x: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    return np.column_stack([np.ones(len(z)), z]) @ model["beta"]


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    clear = score < threshold
    negative, positive = y == 0, y == 1
    return {
        "n": int(len(y)),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "threshold": float(threshold),
        "misses": int(np.sum(positive & clear)),
        "true_negatives": int(np.sum(negative & clear)),
        "specificity": float(np.mean(clear[negative])) if negative.any() else float("nan"),
        "coverage": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--pooled-other-tn", type=int, default=159)
    parser.add_argument("--pooled-negative-count", type=int, default=385)
    args = parser.parse_args()

    calibration, evaluation = read_rows(args.calibration), read_rows(args.evaluation)
    calibration_patients = {row["patient_id"] for row in calibration}
    evaluation_patients = {row["patient_id"] for row in evaluation}
    overlap = sorted(calibration_patients & evaluation_patients)
    if overlap:
        raise ValueError(f"Calibration/evaluation patient overlap: {len(overlap)}")
    if {row["eye_id"] for row in calibration} & {row["eye_id"] for row in evaluation}:
        raise ValueError("Calibration/evaluation eye overlap detected")
    y_cal = np.asarray([int(row["referable_dr"]) for row in calibration], dtype=np.int64)
    y_eval = np.asarray([int(row["referable_dr"]) for row in evaluation], dtype=np.int64)
    raw_cal, raw_eval = matrix(calibration, FEATURES), matrix(evaluation, FEATURES)
    x_cal = percentile_map(raw_cal, raw_cal)
    x_eval = percentile_map(raw_cal, raw_eval)
    model = fit_logistic(x_cal, y_cal.astype(np.float64), args.l2)
    s_cal, s_eval = apply_model(x_cal, model), apply_model(x_eval, model)
    threshold = float(s_cal[y_cal == 1].min())
    cal_metrics, eval_metrics = metrics(y_cal, s_cal, threshold), metrics(y_eval, s_eval, threshold)
    eval_metrics["pooled_true_negatives"] = args.pooled_other_tn + int(eval_metrics["true_negatives"])
    eval_metrics["pooled_negative_count"] = args.pooled_negative_count
    eval_metrics["pooled_specificity"] = eval_metrics["pooled_true_negatives"] / args.pooled_negative_count

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "EXPOSED_VALIDATION_DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "method": "Empirical-CDF transformed paired/macula/optic-disc scores with L2 logistic fusion; fit and zero-miss threshold use calibration patients only.",
        "features": FEATURES,
        "l2": args.l2,
        "calibration_patients": len(calibration_patients),
        "evaluation_patients": len(evaluation_patients),
        "patient_overlap": len(overlap),
        "calibration_metrics": cal_metrics,
        "evaluation_metrics": eval_metrics,
        "feature_coefficients": {name: float(value) for name, value in zip(("intercept", *FEATURES), model["beta"])},
        "feature_percentile_means": {name: float(value) for name, value in zip(FEATURES, model["mean"])},
        "feature_percentile_scales": {name: float(value) for name, value in zip(FEATURES, model["scale"])},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "evaluation_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["eye_id", "patient_id", "grade", "referable_dr", "fused_score", "decision_clear"])
        for row, score in zip(evaluation, s_eval):
            writer.writerow([row["eye_id"], row["patient_id"], row["grade"], row["referable_dr"],
                             float(score), int(score < threshold)])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
