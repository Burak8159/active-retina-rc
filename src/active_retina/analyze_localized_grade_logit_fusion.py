"""Patient-grouped cross-fit fusion of paired five-grade model outputs.

The calibration split is used for grouped out-of-fold model selection and the
evaluation split is touched once with a threshold frozen from calibration.
This is still exploratory because the dataset has already informed development.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load(path: Path):
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"No rows in {path}")
    y = np.asarray([int(row["referable_dr"]) for row in rows], dtype=np.int64)
    groups = np.asarray([row["patient_id"] for row in rows], dtype=str)
    probs = np.asarray([
        [[float(row[f"view{view}_grade_{grade}"]) for grade in range(5)] for view in range(2)]
        for row in rows
    ], dtype=np.float64)
    paired = np.asarray([[float(row[f"paired_grade_{grade}"]) for grade in range(5)]
                         for row in rows], dtype=np.float64)
    old_score = np.asarray([float(row["paired"]) for row in rows], dtype=np.float64)
    return rows, y, groups, probs, paired, old_score


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-8, 1.0))).sum(axis=1)


def make_features(probs: np.ndarray, paired: np.ndarray) -> dict[str, np.ndarray]:
    grades = np.arange(5, dtype=np.float64)
    risk = probs[:, :, 2:].sum(axis=2)
    expected = probs @ grades
    pair_risk = paired[:, 2:].sum(axis=1, keepdims=True)
    pair_expected = (paired @ grades)[:, None]
    disagreement = np.column_stack([
        np.abs(risk[:, 0] - risk[:, 1]),
        np.abs(expected[:, 0] - expected[:, 1]),
        np.abs(entropy(probs[:, 0]) - entropy(probs[:, 1])),
    ])
    basic = np.column_stack([pair_risk, pair_expected, risk, expected, disagreement])
    full = np.column_stack([
        np.log(np.clip(paired, 1e-7, 1.0)),
        np.log(np.clip(probs.reshape(len(probs), 10), 1e-7, 1.0)),
        basic,
        entropy(paired)[:, None],
        entropy(probs[:, 0])[:, None],
        entropy(probs[:, 1])[:, None],
    ])
    return {
        "risk_and_disagreement": basic,
        "grade_probabilities": np.column_stack([paired, probs.reshape(len(probs), 10)]),
        "full": full,
    }


def zero_miss(y: np.ndarray, score: np.ndarray, threshold: float | None = None):
    positive, negative = y == 1, y == 0
    if threshold is None:
        threshold = float(score[positive].min())
    cleared = score < threshold
    return {
        "threshold": float(threshold),
        "tn": int(np.sum(negative & cleared)),
        "n_negative": int(negative.sum()),
        "misses": int(np.sum(positive & cleared)),
        "n_positive": int(positive.sum()),
        "specificity": float(np.mean(cleared[negative])),
        "coverage": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    cal_rows, y, groups, probs, paired, base_score = load(Path(args.calibration))
    eval_rows, y_eval, groups_eval, probs_eval, paired_eval, base_eval = load(Path(args.evaluation))
    if set(groups) & set(groups_eval):
        raise ValueError("Calibration and evaluation patient IDs overlap")
    if set(y.tolist()) != {0, 1} or set(y_eval.tolist()) != {0, 1}:
        raise ValueError("Both partitions must contain positive and negative eyes")
    feature_sets = make_features(probs, paired)
    eval_feature_sets = make_features(probs_eval, paired_eval)
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=261012)
    rows = []
    for feature_name, x in feature_sets.items():
        for c_value in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
            oof = np.full(len(y), np.nan, dtype=np.float64)
            for train_idx, valid_idx in cv.split(x, y, groups):
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=c_value, class_weight="balanced", max_iter=1000,
                    solver="lbfgs", random_state=261012,
                ))
                model.fit(x[train_idx], y[train_idx])
                oof[valid_idx] = model.predict_proba(x[valid_idx])[:, 1]
            if not np.isfinite(oof).all():
                raise ValueError("Cross-fit predictions are incomplete")
            m = zero_miss(y, oof)
            rows.append({
                "feature_set": feature_name,
                "C": c_value,
                **m,
                "oof_auroc": float(roc_auc_score(y, oof)),
            })
    rows.sort(key=lambda row: (row["tn"], row["oof_auroc"]), reverse=True)
    winner = rows[0]
    x_cal = feature_sets[winner["feature_set"]]
    x_eval = eval_feature_sets[winner["feature_set"]]
    final_model = make_pipeline(StandardScaler(), LogisticRegression(
        C=winner["C"], class_weight="balanced", max_iter=1000,
        solver="lbfgs", random_state=261012,
    ))
    final_model.fit(x_cal, y)
    score_cal = final_model.predict_proba(x_cal)[:, 1]
    score_eval = final_model.predict_proba(x_eval)[:, 1]
    frozen = zero_miss(y, score_cal)
    evaluation = zero_miss(y_eval, score_eval, frozen["threshold"])
    baseline_cal = zero_miss(y, base_score)
    baseline_eval = zero_miss(y_eval, base_eval, baseline_cal["threshold"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "EXPLORATORY_CALIBRATION_CROSSFIT_FUSION",
        "boundary": "Patient-grouped out-of-fold calibration selects feature family and regularization; threshold is fit on calibration and applied unchanged to evaluation. All source data have already informed development, so this is not independent confirmation or clinical certification.",
        "calibration_patients": int(len(set(groups))),
        "calibration_eyes": int(len(y)),
        "evaluation_patients": int(len(set(groups_eval))),
        "evaluation_eyes": int(len(y_eval)),
        "selected_from_grouped_oof_calibration": winner,
        "calibration_fit_metrics": frozen,
        "evaluation_frozen_metrics": evaluation,
        "paired_score_baseline_calibration": baseline_cal,
        "paired_score_baseline_evaluation_frozen": baseline_eval,
        "all_oof_candidates": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "candidate_grid.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "evaluation_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["eye_id", "patient_id", "referable_dr", "risk_score", "clear_threshold"])
        for row, score in zip(eval_rows, score_eval):
            writer.writerow([row["eye_id"], row["patient_id"], row["referable_dr"],
                             float(score), frozen["threshold"]])
    print(json.dumps({k: v for k, v in summary.items() if k != "all_oof_candidates"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
