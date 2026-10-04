"""Train a learned fusion head for the augmented autonomy signal family.

The available base score files contain validation/test predictions. To avoid
training and thresholding on the same rows, this script splits validation into
an inner fusion-training subset and an inner calibration subset, then applies
the selected policy once to the held-out test split.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from active_retina.autonomy_failure_analysis import DDR_MODELS
from active_retina.autonomy_zero_miss_90_search import metrics, ranknorm, select_policy
from active_retina.three_output_autonomous_triage import triage_metrics


SEED = 20261004


def read_prediction_file(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            score = row.get("referable_probability", row.get("score"))
            if score is None:
                continue
            rows[row["image_id"]] = {
                "split": row["split"],
                "image_id": row["image_id"],
                "referable_dr": int(row["referable_dr"]),
                "score": float(score),
            }
    return rows


def read_all(extra_predictions: Path, extra_name: str) -> tuple[list[str], list[str], dict[str, dict[str, dict[str, Any]]]]:
    model_rows = {
        name: read_prediction_file(Path(path))
        for name, path in DDR_MODELS.items()
        if Path(path).exists()
    }
    model_rows[extra_name] = read_prediction_file(extra_predictions)
    models = sorted(model_rows)
    common_ids = sorted(set.intersection(*(set(rows) for rows in model_rows.values())))
    return models, common_ids, model_rows


def stratified_inner_split(ids: list[str], y: np.ndarray, fraction_train: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    train = np.zeros(len(ids), dtype=bool)
    for label in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_train = max(1, int(round(len(idx) * fraction_train)))
        train[idx[:n_train]] = True
    return train, ~train


def candidate_models() -> dict[str, Any]:
    return {
        "logistic_c005_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000, random_state=SEED),
        ),
        "logistic_c025_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, class_weight="balanced", max_iter=5000, random_state=SEED + 1),
        ),
        "extra_trees_depth3": ExtraTreesClassifier(
            n_estimators=800,
            max_depth=3,
            min_samples_leaf=12,
            class_weight="balanced",
            random_state=SEED + 2,
            n_jobs=-1,
        ),
        "extra_trees_depth5": ExtraTreesClassifier(
            n_estimators=800,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=SEED + 3,
            n_jobs=-1,
        ),
        "random_forest_depth4": RandomForestClassifier(
            n_estimators=800,
            max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=SEED + 4,
            n_jobs=-1,
        ),
        "hist_gradient_l2": make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(
                max_iter=120,
                max_leaf_nodes=7,
                learning_rate=0.03,
                l2_regularization=1.0,
                random_state=SEED + 5,
            ),
        ),
    }


def action(score: float, low: float, high: float) -> str:
    if score <= low:
        return "AUTONOMOUS_NONREFERABLE"
    if score >= high:
        return "AUTONOMOUS_REFERABLE"
    return "AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE"


def prediction_rows(ids: list[str], splits: list[str], y: np.ndarray, scores: np.ndarray, low: float, high: float) -> list[dict[str, Any]]:
    rows = []
    for image_id, split, yy, score in zip(ids, splits, y, scores):
        rows.append(
            {
                "split": split,
                "image_id": image_id,
                "referable_dr": int(yy),
                "score": float(score),
                "low_threshold": float(low),
                "high_threshold": float(high),
                "action": action(float(score), float(low), float(high)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-predictions", required=True)
    parser.add_argument("--extra-name", default="clementp_efficientnet_b5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inner-train-fraction", type=float, default=0.5)
    parser.add_argument("--min-specificity", type=float, default=0.90)
    args = parser.parse_args()

    models, image_ids, model_rows = read_all(Path(args.extra_predictions), args.extra_name)
    splits = [model_rows[models[0]][image_id]["split"] for image_id in image_ids]
    y = np.asarray([model_rows[models[0]][image_id]["referable_dr"] for image_id in image_ids], dtype=int)
    raw = np.asarray([[model_rows[m][image_id]["score"] for m in models] for image_id in image_ids], dtype=float)
    ranked = np.column_stack([ranknorm(raw[:, idx], splits) for idx in range(raw.shape[1])])
    features = np.column_stack([raw, ranked])
    feature_names = [*models, *[f"{m}_ranknorm" for m in models]]

    valid_mask = np.asarray([s == "valid" for s in splits])
    test_mask = np.asarray([s == "test" for s in splits])
    valid_ids = [image_id for image_id, is_valid in zip(image_ids, valid_mask) if is_valid]
    valid_y = y[valid_mask]
    inner_train_local, inner_cal_local = stratified_inner_split(valid_ids, valid_y, args.inner_train_fraction)
    inner_train_mask = np.zeros(len(image_ids), dtype=bool)
    inner_cal_mask = np.zeros(len(image_ids), dtype=bool)
    valid_positions = np.flatnonzero(valid_mask)
    inner_train_mask[valid_positions[inner_train_local]] = True
    inner_cal_mask[valid_positions[inner_cal_local]] = True

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_scores: np.ndarray | None = None
    for name, model in candidate_models().items():
        model.fit(features[inner_train_mask], y[inner_train_mask])
        scores = model.predict_proba(features)[:, 1]
        selected = select_policy(y[inner_cal_mask], scores[inner_cal_mask], args.min_specificity)
        if "status" in selected:
            rows.append({"candidate": name, "status": selected["status"]})
            continue
        test = metrics(y[test_mask], scores[test_mask], float(selected["low"]), float(selected["high"]))
        status = (
            "GO_ZERO_MISS_90_SPECIFICITY"
            if test["false_negative_clearances"] == 0
            and test["specificity_among_auto_decisions"] >= args.min_specificity
            and test["negative_clearance_rate"] > 0
            else "NO_GO_TEST"
        )
        row = {
            "candidate": name,
            "status": status,
            **{f"calibration_{k}": v for k, v in selected.items()},
            **{f"test_{k}": v for k, v in test.items()},
        }
        rows.append(row)
        key = (
            int(status == "GO_ZERO_MISS_90_SPECIFICITY"),
            float(row.get("test_coverage", 0.0)),
            float(row.get("test_negative_clearance_rate", 0.0)),
            float(row.get("test_specificity_among_auto_decisions", 0.0)),
        )
        best_key = (
            int(best.get("status") == "GO_ZERO_MISS_90_SPECIFICITY") if best else 0,
            float(best.get("test_coverage", 0.0)) if best else 0.0,
            float(best.get("test_negative_clearance_rate", 0.0)) if best else 0.0,
            float(best.get("test_specificity_among_auto_decisions", 0.0)) if best else 0.0,
        )
        if best is None or key > best_key:
            best = row
            best_scores = scores

    write_csv(out / "candidate_table.csv", rows)
    best = best or {"status": "NO_CANDIDATES"}
    pred_rows: list[dict[str, Any]] = []
    if best_scores is not None and "calibration_low" in best and "calibration_high" in best:
        pred_rows = prediction_rows(image_ids, splits, y, best_scores, float(best["calibration_low"]), float(best["calibration_high"]))
        write_csv(out / "best_predictions.csv", pred_rows)
    summary = {
        "status": "GO_AUGMENTED_LEARNED_FUSION" if best.get("status") == "GO_ZERO_MISS_90_SPECIFICITY" else "NO_GO_AUGMENTED_LEARNED_FUSION",
        "feature_names": feature_names,
        "inner_train_n": int(inner_train_mask.sum()),
        "inner_calibration_n": int(inner_cal_mask.sum()),
        "test_n": int(test_mask.sum()),
        "best": best,
        "best_triage_metrics": triage_metrics([r for r in pred_rows if r.get("split") == "test"]) if pred_rows else {},
        "table": str(out / "candidate_table.csv"),
        "best_predictions": str(out / "best_predictions.csv") if pred_rows else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
