"""Full-coverage autonomy search using existing DDR model scores.

This is the opposite of the selective 90/0 mode: every test image receives a
binary CLEAR/REFER decision. Models are trained on train scores, thresholds are
selected on validation scores, and the frozen threshold is applied to test.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from active_retina.autonomy_failure_analysis import DDR_MODELS, write_csv


def read_rows() -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    per_model = {}
    for model, rel in DDR_MODELS.items():
        path = Path(rel)
        if not path.exists():
            continue
        rows = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                score = row.get("referable_probability", row.get("score"))
                rows[row["image_id"]] = {
                    "split": row["split"],
                    "label": int(row["referable_dr"]),
                    "score": float(score),
                }
        per_model[model] = rows
    models = sorted(per_model)
    common_ids = sorted(set.intersection(*(set(rows) for rows in per_model.values())))
    merged = {}
    for image_id in common_ids:
        base = per_model[models[0]][image_id]
        merged[image_id] = {
            "split": base["split"],
            "label": base["label"],
            "scores": np.asarray([per_model[m][image_id]["score"] for m in models], dtype=float),
        }
    return models, common_ids, merged


def rank_features(scores: np.ndarray, splits: np.ndarray) -> np.ndarray:
    ranks = np.zeros_like(scores, dtype=float)
    for split in sorted(set(splits.tolist())):
        idx = splits == split
        for j in range(scores.shape[1]):
            vals = scores[idx, j]
            order = np.argsort(vals, kind="mergesort")
            rr = np.empty_like(vals)
            rr[order] = np.linspace(0.0, 1.0, len(vals), endpoint=True)
            ranks[idx, j] = rr
    return ranks


def make_xy(ids: list[str], merged: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    splits = np.asarray([merged[i]["split"] for i in ids])
    y = np.asarray([merged[i]["label"] for i in ids], dtype=int)
    raw = np.vstack([merged[i]["scores"] for i in ids]).astype(float)
    ranks = rank_features(raw, splits)
    engineered = np.column_stack(
        [
            raw,
            ranks,
            raw.mean(axis=1),
            raw.max(axis=1),
            raw.min(axis=1),
            np.median(raw, axis=1),
            np.sort(raw, axis=1)[:, -2:].mean(axis=1),
            ranks.mean(axis=1),
            ranks.max(axis=1),
            ranks.min(axis=1),
            np.median(ranks, axis=1),
            np.sort(ranks, axis=1)[:, -2:].mean(axis=1),
        ]
    )
    return engineered.astype("float32"), y, splits, ids


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = score >= threshold
    pos = y == 1
    neg = y == 0
    tp = int((pos & pred).sum())
    fn = int((pos & ~pred).sum())
    tn = int((neg & ~pred).sum())
    fp = int((neg & pred).sum())
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "coverage": 1.0,
    }


def select_zero_miss_threshold(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    thresholds = sorted(set(float(x) for x in score))
    thresholds = [min(thresholds) - 1e-12] + thresholds
    candidates = []
    for t in thresholds:
        m = binary_metrics(y, score, t)
        if m["fn"] == 0:
            candidates.append(m)
    if not candidates:
        return {"status": "NO_ZERO_MISS_THRESHOLD"}
    return max(candidates, key=lambda m: (m["specificity"], m["threshold"]))


def predict_score(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict(x).astype(float)


def main() -> None:
    models, ids, merged = read_rows()
    x, y, splits, ids = make_xy(ids, merged)
    test = splits == "test"
    valid_pool = splits == "valid"
    if int(test.sum()) == 0:
        raise RuntimeError("No test rows found in the merged DDR score files.")
    if int(valid_pool.sum()) == 0:
        raise RuntimeError("No validation rows found for meta-model training/calibration.")

    valid_indices = np.flatnonzero(valid_pool)
    meta_train_idx, meta_cal_idx = train_test_split(
        valid_indices,
        test_size=0.35,
        random_state=20261004,
        stratify=y[valid_indices],
    )
    meta_train = np.zeros_like(valid_pool, dtype=bool)
    meta_cal = np.zeros_like(valid_pool, dtype=bool)
    meta_train[meta_train_idx] = True
    meta_cal[meta_cal_idx] = True
    candidates: dict[str, Any] = {
        "logistic_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.25, solver="lbfgs"),
        ),
        "logistic_poly2_balanced": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.05, solver="lbfgs"),
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=800,
            max_depth=8,
            min_samples_leaf=3,
            class_weight={0: 1.0, 1: 12.0},
            random_state=20261004,
            n_jobs=-1,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=800,
            max_depth=10,
            min_samples_leaf=4,
            class_weight={0: 1.0, 1: 12.0},
            random_state=20261004,
            n_jobs=-1,
        ),
        "hist_gradient": HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            l2_regularization=0.1,
            max_leaf_nodes=15,
            random_state=20261004,
        ),
        "rbf_svm_balanced": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", class_weight={0: 1.0, 1: 12.0}, probability=True, random_state=20261004),
        ),
    }
    rows = []
    pred_rows = []
    for name, model in candidates.items():
        fit_kwargs = {}
        if name == "hist_gradient":
            sw = np.ones(int(meta_train.sum()), dtype=float)
            sw[y[meta_train] == 1] = 12.0
            fit_kwargs["sample_weight"] = sw
        model.fit(x[meta_train], y[meta_train], **fit_kwargs)
        valid_score = predict_score(model, x[meta_cal])
        test_score = predict_score(model, x[test])
        selected = select_zero_miss_threshold(y[meta_cal], valid_score)
        if "threshold" not in selected:
            rows.append({"model": name, "status": selected["status"]})
            continue
        test_m = binary_metrics(y[test], test_score, float(selected["threshold"]))
        try:
            auc = roc_auc_score(y[test], test_score)
            ap = average_precision_score(y[test], test_score)
        except ValueError:
            auc = float("nan")
            ap = float("nan")
        status = (
            "GO_FULL_COVERAGE_90_0"
            if test_m["fn"] == 0 and test_m["specificity"] >= 0.90
            else "NO_GO_FULL_COVERAGE"
        )
        rows.append(
            {
                "model": name,
                "status": status,
                "test_auroc": auc,
                "test_average_precision": ap,
                "meta_train_n": int(meta_train.sum()),
                "meta_calibration_n": int(meta_cal.sum()),
                "test_n": int(test.sum()),
                **{f"valid_{k}": v for k, v in selected.items()},
                **{f"test_{k}": v for k, v in test_m.items()},
            }
        )
        for image_id, yy, score in zip(np.asarray(ids)[test], y[test], test_score):
            pred_rows.append(
                {
                    "model": name,
                    "split": "test",
                    "image_id": image_id,
                    "referable_dr": int(yy),
                    "score": float(score),
                    "threshold": float(selected["threshold"]),
                    "prediction": int(score >= float(selected["threshold"])),
                }
            )
    write_csv(Path("results/autonomy_full_coverage_meta_search.csv"), rows)
    write_csv(Path("results/autonomy_full_coverage_meta_predictions.csv"), pred_rows)
    feasible = [r for r in rows if r.get("status") == "GO_FULL_COVERAGE_90_0"]
    best = max(
        rows,
        key=lambda r: (
            int(r.get("status") == "GO_FULL_COVERAGE_90_0"),
            int(float(r.get("test_fn", 999)) == 0),
            float(r.get("test_specificity", 0.0)),
            float(r.get("test_sensitivity", 0.0)),
            float(r.get("test_auroc", 0.0)),
        ),
    )
    summary = {
        "status": "GO_FULL_COVERAGE_90_0_FOUND" if feasible else "NO_GO_FULL_COVERAGE_90_0",
        "n_models": len(rows),
        "n_feasible": len(feasible),
        "best": best,
        "input_score_models": models,
        "selection_rule": "train meta-model on 65% of validation scores; select zero-miss threshold on the remaining 35%; apply to every test image",
        "meta_train_n": int(meta_train.sum()),
        "meta_calibration_n": int(meta_cal.sum()),
        "test_n": int(test.sum()),
        "table": "results/autonomy_full_coverage_meta_search.csv",
        "predictions": "results/autonomy_full_coverage_meta_predictions.csv",
    }
    Path("results/autonomy_full_coverage_meta_search_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Full-Coverage 90/0 Meta-Model Search\n\n",
        "This experiment tries to remove abstention entirely. Meta-models are trained on existing DDR model scores, a zero-miss threshold is selected on validation, and every test image receives CLEAR or REFER.\n\n",
        f"Status: **{summary['status']}**\n\n",
        f"Best model: `{best.get('model')}`\n\n",
        f"Test false negatives: **{best.get('test_fn')}**\n\n",
        f"Test specificity: **{100 * float(best.get('test_specificity', 0.0)):.1f}%**\n\n",
        f"Test sensitivity: **{100 * float(best.get('test_sensitivity', 0.0)):.1f}%**\n",
    ]
    Path("docs/autonomy_full_coverage_meta_search_report.md").write_text("".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
