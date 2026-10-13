"""Patient-fold cross-fit a score-only ensemble for exposed DRTiD development."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


NEW_FEATURES = ("paired", "macula", "optic_disc", "mean_view", "max_view")
OLD_FEATURES = ("risk", "ordinal_risk", "max_grade_margin", "grade_boundary_margin")


def read(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks / max(1, len(values) - 1)


def rank_features(train: np.ndarray, test: np.ndarray):
    train_ranks, test_ranks = [], []
    for column in range(train.shape[1]):
        train_ranks.append(average_rank(train[:, column]))
        ordered = np.sort(train[:, column])
        test_ranks.append(np.searchsorted(ordered, test[:, column], side="right") / len(ordered))
    return np.column_stack(train_ranks), np.column_stack(test_ranks)


def logistic_fit(x: np.ndarray, y: np.ndarray, penalty: float):
    mean, scale = x.mean(axis=0), x.std(axis=0).clip(1e-6)
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(design.shape[1])
    reg = np.eye(len(beta)); reg[0, 0] = 0.0
    for _ in range(4000):
        probability = 1 / (1 + np.exp(-np.clip(design @ beta, -30, 30)))
        gradient = design.T @ (probability - y) / len(y) + penalty * (reg @ beta)
        beta -= 0.01 * gradient
    return mean, scale, beta


def apply(x: np.ndarray, fit):
    mean, scale, beta = fit
    return np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ beta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-predictions", required=True, type=Path)
    parser.add_argument("--baseline-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--other-source-tn", type=int, default=159)
    parser.add_argument("--pooled-negatives", type=int, default=385)
    args = parser.parse_args()

    new_rows = read(args.new_predictions)
    old_rows = [r for r in read(args.baseline_predictions) if r.get("source_dataset") == "drtid_paired"]
    new = {r["eye_id"]: r for r in new_rows}
    old = {r["eye_id"]: r for r in old_rows}
    if len(new) != len(new_rows) or len(old) != len(old_rows) or set(new) != set(old):
        raise ValueError("Prediction files must have identical unique DRTiD eye IDs")
    eyes = sorted(new)
    patients = [new[e]["patient_id"] for e in eyes]
    if len(set(patients)) != len(patients):
        raise ValueError("This implementation expects one eye per unique patient")
    labels = np.asarray([int(new[e]["referable_dr"]) for e in eyes], dtype=np.int64)
    if any(int(old[e]["referable"]) != int(new[e]["referable_dr"]) for e in eyes):
        raise ValueError("Labels differ between model prediction files")
    features = np.asarray([
        [float(new[e][name]) for name in NEW_FEATURES]
        + [float(old[e][name]) for name in OLD_FEATURES]
        for e in eyes
    ], dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite score found")

    rng = np.random.default_rng(args.seed)
    fold_id = np.full(len(eyes), -1, dtype=np.int64)
    for label in (0, 1):
        indexes = np.flatnonzero(labels == label)
        rng.shuffle(indexes)
        fold_id[indexes] = np.arange(len(indexes)) % args.folds

    out_scores = np.full(len(eyes), np.nan)
    out_thresholds = np.full(len(eyes), np.nan)
    fold_reports = []
    for fold in range(args.folds):
        test = fold_id == fold
        train = ~test
        x_train, x_test = rank_features(features[train], features[test])
        fitted = logistic_fit(x_train, labels[train].astype(float), args.l2)
        score_train, score_test = apply(x_train, fitted), apply(x_test, fitted)
        threshold = float(score_train[labels[train] == 1].min())
        out_scores[test], out_thresholds[test] = score_test, threshold
        misses = int(np.sum((labels[test] == 1) & (score_test < threshold)))
        tn = int(np.sum((labels[test] == 0) & (score_test < threshold)))
        fold_reports.append({"fold": fold, "n": int(test.sum()), "positive": int(labels[test].sum()),
                             "negative": int((labels[test] == 0).sum()), "calibration_threshold": threshold,
                             "evaluation_misses": misses, "evaluation_true_negatives": tn})

    misses = int(np.sum((labels == 1) & (out_scores < out_thresholds)))
    tn = int(np.sum((labels == 0) & (out_scores < out_thresholds)))
    pooled_tn = args.other_source_tn + tn
    summary = {
        "status": "EXPOSED_VALIDATION_PATIENT_CROSSFIT_DEVELOPMENT_ONLY",
        "boundary": "Base model predictions were previously inspected for model selection. This cross-fit only evaluates the newly fitted score stack; it is not untouched clinical validation.",
        "patients": len(set(patients)), "folds": args.folds, "features": [*NEW_FEATURES, *OLD_FEATURES],
        "l2": args.l2, "misses": misses, "drtid_true_negatives": tn,
        "drtid_negatives": int((labels == 0).sum()), "coverage": 1.0,
        "pooled_true_negatives": pooled_tn, "pooled_negatives": args.pooled_negatives,
        "pooled_specificity": pooled_tn / args.pooled_negatives,
        "folds_detail": fold_reports,
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["eye_id", "patient_id", "grade", "referable_dr", "fold", "stack_score", "fold_threshold", "clear"])
        for i, eye in enumerate(eyes):
            writer.writerow([eye, patients[i], new[eye]["grade"], labels[i], fold_id[i], out_scores[i],
                             out_thresholds[i], int(out_scores[i] < out_thresholds[i])])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
