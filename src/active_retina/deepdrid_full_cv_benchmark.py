"""Full DeepDRiD-only patient-level cross-validation benchmark.

This is the no-MFIDDR journal-track experiment:
- outer patient folds,
- inner calibration patients inside each outer training fold,
- referable-DR binary models,
- five-class DR grading models,
- fixed one-view, fixed two-view, confidence triage, and two-stage acquisition,
- patient-bootstrap intervals over out-of-fold decisions.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate_gates import auc_binary, average_precision, brier_score
from .write_fast_paper_pack import (
    BRANCHES,
    candidate_grid,
    confidence_triage_metrics,
    summarize_decisions,
    two_stage_metrics,
    write_csv,
)


SEED = 20261003
OUTER_FOLDS = 5
INNER_CALIBRATION_FRACTION = 0.20
TARGET_RISKS = (0.05, 0.10, 0.15)
MIN_SPECIFICITY = 0.50
BOOTSTRAP_N = 1000
BRANCH_INPUTS = {
    "suffix2_initial": {
        "manifest": Path("results/deepdrid_manifest.csv"),
        "features": Path("results/deepdrid_resnet50_features.npz"),
        "first_view": "macula",
        "second_view": "optic_disc",
    },
    "suffix1_initial": {
        "manifest": Path("results/deepdrid_swapped_mapping_manifest.csv"),
        "features": Path("results/deepdrid_swapped_resnet50_features.npz"),
        "first_view": "macula",
        "second_view": "optic_disc",
    },
}


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


def group_by_eye(rows: list[dict[str, Any]], features: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id = row["image_id"]
        if image_id not in features:
            continue
        item = out.setdefault(
            row["eye_id"],
            {
                "patient_id": row["patient_id"],
                "eye_id": row["eye_id"],
                "grade": row["grade"],
                "referable_dr": row["referable_dr"],
                "views": {},
            },
        )
        item["views"][row["field"]] = features[image_id]
    return {k: v for k, v in out.items() if len(v["views"]) >= 2}


def pooled_feature(item: dict[str, Any], views: tuple[str, ...]) -> np.ndarray | None:
    if any(v not in item["views"] for v in views):
        return None
    feats = [item["views"][v] for v in views]
    return np.concatenate([np.mean(feats, axis=0), np.max(feats, axis=0), np.asarray([len(feats)], dtype=np.float32)]).astype(np.float32)


def patient_folds(patients: list[str], folds: int, seed: int) -> list[set[str]]:
    shuffled = list(patients)
    random.Random(seed).shuffle(shuffled)
    return [set(shuffled[i::folds]) for i in range(folds)]


def split_inner_calibration(train_patients: list[str], fold_id: int) -> tuple[set[str], set[str]]:
    shuffled = list(train_patients)
    random.Random(SEED + 101 * fold_id).shuffle(shuffled)
    n_cal = max(1, int(round(len(shuffled) * INNER_CALIBRATION_FRACTION)))
    calibration = set(shuffled[:n_cal])
    fit = set(shuffled[n_cal:])
    return fit, calibration


def fit_binary(x: np.ndarray, y: np.ndarray, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=seed),
    ).fit(x, y)


def fit_multiclass(x: np.ndarray, y: np.ndarray, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=seed),
    ).fit(x, y)


def matrix_for(items: list[dict[str, Any]], views: tuple[str, ...]) -> tuple[list[dict[str, Any]], np.ndarray]:
    usable, xs = [], []
    for item in items:
        feat = pooled_feature(item, views)
        if feat is not None:
            usable.append(item)
            xs.append(feat)
    return usable, np.vstack(xs).astype(np.float32)


def predict_block(
    grouped: dict[str, dict[str, Any]],
    fit_patients: set[str],
    calibration_patients: set[str],
    validation_patients: set[str],
    first_view: str,
    second_view: str,
    fold_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    items = list(grouped.values())
    fit_items = [x for x in items if x["patient_id"] in fit_patients]
    cal_items = [x for x in items if x["patient_id"] in calibration_patients]
    val_items = [x for x in items if x["patient_id"] in validation_patients]
    view_sets = {
        "first": (first_view,),
        "two": tuple(sorted((first_view, second_view))),
    }

    models: dict[str, Any] = {}
    grade_models: dict[str, Any] = {}
    for name, views in view_sets.items():
        fit_usable, x_fit = matrix_for(fit_items, views)
        y_ref = np.asarray([x["referable_dr"] for x in fit_usable], dtype=int)
        y_grade = np.asarray([x["grade"] for x in fit_usable], dtype=int)
        models[name] = fit_binary(x_fit, y_ref, SEED + fold_id)
        grade_models[name] = fit_multiclass(x_fit, y_grade, SEED + 1000 + fold_id)

    def predict(items_subset: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        first_items, x_first = matrix_for(items_subset, view_sets["first"])
        two_items, x_two = matrix_for(items_subset, view_sets["two"])
        first_by_eye = {
            item["eye_id"]: (float(score), int(pred))
            for item, score, pred in zip(
                first_items,
                models["first"].predict_proba(x_first)[:, 1],
                grade_models["first"].predict(x_first),
            )
        }
        two_by_eye = {
            item["eye_id"]: (float(score), int(pred))
            for item, score, pred in zip(
                two_items,
                models["two"].predict_proba(x_two)[:, 1],
                grade_models["two"].predict(x_two),
            )
        }
        rows = []
        for item in items_subset:
            if item["eye_id"] not in first_by_eye or item["eye_id"] not in two_by_eye:
                continue
            first_score, first_grade = first_by_eye[item["eye_id"]]
            two_score, two_grade = two_by_eye[item["eye_id"]]
            rows.append(
                {
                    "outer_fold": fold_id,
                    "split": split,
                    "patient_id": item["patient_id"],
                    "eye_id": item["eye_id"],
                    "grade": item["grade"],
                    "referable_dr": item["referable_dr"],
                    "first_score": first_score,
                    "two_view_score": two_score,
                    "first_grade_pred": first_grade,
                    "two_view_grade_pred": two_grade,
                }
            )
        return rows

    return predict(cal_items, "inner_calibration"), predict(val_items, "outer_validation"), predict(fit_items, "fit")


def threshold_metrics(examples: list[dict[str, Any]], score_key: str, threshold: float, images: int) -> dict[str, Any]:
    rows = []
    for ex in examples:
        pred = 1 if ex[score_key] >= threshold else 0
        rows.append({**ex, "prediction": pred, "score": ex[score_key], "images": images, "deferred": 0, "action": score_key})
    metrics = summarize_decisions(rows)
    metrics["rows"] = rows
    return metrics


def confidence_triage_with_rows(examples: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    rows = []
    for ex in examples:
        if ex["first_score"] <= low:
            rows.append({**ex, "prediction": 0, "score": ex["first_score"], "images": 1, "deferred": 0, "action": "STOP_NEGATIVE"})
        elif ex["first_score"] >= high:
            rows.append({**ex, "prediction": 1, "score": ex["first_score"], "images": 1, "deferred": 0, "action": "STOP_POSITIVE"})
        else:
            rows.append({**ex, "prediction": None, "score": ex["first_score"], "images": 1, "deferred": 1, "action": "DEFER"})
    metrics = summarize_decisions(rows)
    metrics["rows"] = rows
    return metrics


def two_stage_with_rows(examples: list[dict[str, Any]], fl: float, fh: float, sl: float, sh: float) -> dict[str, Any]:
    rows = []
    for ex in examples:
        if ex["first_score"] <= fl:
            rows.append({**ex, "prediction": 0, "score": ex["first_score"], "images": 1, "deferred": 0, "action": "STOP_NEGATIVE"})
        elif ex["first_score"] >= fh:
            rows.append({**ex, "prediction": 1, "score": ex["first_score"], "images": 1, "deferred": 0, "action": "STOP_POSITIVE"})
        elif ex["two_view_score"] <= sl:
            rows.append({**ex, "prediction": 0, "score": ex["two_view_score"], "images": 2, "deferred": 0, "action": "ACQUIRE_STOP_NEGATIVE"})
        elif ex["two_view_score"] >= sh:
            rows.append({**ex, "prediction": 1, "score": ex["two_view_score"], "images": 2, "deferred": 0, "action": "ACQUIRE_STOP_POSITIVE"})
        else:
            rows.append({**ex, "prediction": None, "score": ex["two_view_score"], "images": 2, "deferred": 1, "action": "ACQUIRE_DEFER"})
    metrics = summarize_decisions(rows)
    metrics["rows"] = rows
    return metrics


def select_threshold(calibration: list[dict[str, Any]], score_key: str, target: float, images: int) -> tuple[Any, dict[str, Any]]:
    candidates = []
    for threshold in candidate_grid([x[score_key] for x in calibration], 101):
        metrics = threshold_metrics(calibration, score_key, threshold, images)
        if math.isfinite(metrics["missed_referable_risk_decided"]) and metrics["missed_referable_risk_decided"] <= target:
            candidates.append((threshold, metrics))
    if not candidates:
        return None, {"status": "NO_POLICY"}
    return min(candidates, key=lambda x: (-x[1]["specificity_decided"], x[1]["mean_images"]))


def select_triage(calibration: list[dict[str, Any]], target: float) -> tuple[Any, dict[str, Any]]:
    candidates = []
    scores = candidate_grid([x["first_score"] for x in calibration], 51)
    for low in scores:
        for high in scores:
            if low >= high:
                continue
            metrics = confidence_triage_metrics(calibration, low, high)
            if (
                math.isfinite(metrics["missed_referable_risk_decided"])
                and metrics["missed_referable_risk_decided"] <= target
                and math.isfinite(metrics["specificity_decided"])
                and metrics["specificity_decided"] >= MIN_SPECIFICITY
            ):
                candidates.append((low, high, metrics))
    if not candidates:
        return None, {"status": "NO_POLICY"}
    return min(candidates, key=lambda x: (x[2]["defer_rate"], -x[2]["specificity_decided"]))


def select_two_stage(calibration: list[dict[str, Any]], target: float) -> tuple[Any, dict[str, Any]]:
    candidates = []
    first_scores = candidate_grid([x["first_score"] for x in calibration], 9)
    two_scores = candidate_grid([x["two_view_score"] for x in calibration], 9)
    for fl in first_scores:
        for fh in first_scores:
            if fl >= fh:
                continue
            for sl in two_scores:
                for sh in two_scores:
                    if sl >= sh:
                        continue
                    metrics = two_stage_metrics(calibration, fl, fh, sl, sh)
                    if (
                        math.isfinite(metrics["missed_referable_risk_decided"])
                        and metrics["missed_referable_risk_decided"] <= target
                        and math.isfinite(metrics["specificity_decided"])
                        and metrics["specificity_decided"] >= MIN_SPECIFICITY
                    ):
                        candidates.append((fl, fh, sl, sh, metrics))
    if not candidates:
        return None, {"status": "NO_POLICY"}
    return min(candidates, key=lambda x: (x[4]["defer_rate"], x[4]["mean_images"], -x[4]["specificity_decided"]))


def apply_policy(validation: list[dict[str, Any]], method: str, params: Any) -> dict[str, Any]:
    if params is None:
        return {"status": "NO_POLICY"}
    if method == "fixed_one":
        return threshold_metrics(validation, "first_score", params, 1)
    if method == "fixed_two":
        return threshold_metrics(validation, "two_view_score", params, 2)
    if method == "confidence_triage":
        low, high = params
        return confidence_triage_with_rows(validation, low, high)
    if method == "two_stage":
        fl, fh, sl, sh = params
        return two_stage_with_rows(validation, fl, fh, sl, sh)
    raise ValueError(method)


def binary_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, float]:
    y = [r["referable_dr"] for r in rows]
    s = [r[score_key] for r in rows]
    return {
        "auroc": auc_binary(y, s),
        "auprc": average_precision(y, s),
        "brier": brier_score(y, s),
        "ece10": ece(y, s, 10),
    }


def ece(y: list[int], s: list[float], bins: int) -> float:
    total = len(y)
    val = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j, score in enumerate(s) if (lo <= score < hi) or (i == bins - 1 and score == hi)]
        if not idx:
            continue
        conf = sum(s[j] for j in idx) / len(idx)
        acc = sum(y[j] for j in idx) / len(idx)
        val += len(idx) / total * abs(conf - acc)
    return val


def grade_metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, float]:
    y = [r["grade"] for r in rows]
    p = [r[pred_key] for r in rows]
    return {
        "accuracy": float(accuracy_score(y, p)),
        "qwk": float(cohen_kappa_score(y, p, weights="quadratic")),
    }


def metric_ci(rows: list[dict[str, Any]], fn, key: str) -> tuple[float, float, float]:
    obs = fn(rows)[key]
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    patients = sorted(by_patient)
    rng = np.random.default_rng(SEED)
    reps = []
    for _ in range(BOOTSTRAP_N):
        sample = [x for patient in rng.choice(patients, size=len(patients), replace=True) for x in by_patient[patient]]
        try:
            val = fn(sample)[key]
        except Exception:
            continue
        if math.isfinite(val):
            reps.append(val)
    lo, hi = np.quantile(reps, [0.025, 0.975]) if reps else (float("nan"), float("nan"))
    return float(obs), float(lo), float(hi)


def compact_policy(branch: str, target: float, method: str, metrics: dict[str, Any]) -> dict[str, Any]:
    if metrics.get("status") == "NO_POLICY":
        return {"branch": branch, "target_risk": target, "method": method, "status": "NO_POLICY"}
    return {
        "branch": branch,
        "target_risk": target,
        "method": method,
        "status": "OK",
        "images": metrics["mean_images"],
        "defer": metrics["defer_rate"],
        "coverage": 1.0 - metrics["defer_rate"],
        "miss_risk": metrics["missed_referable_risk_decided"],
        "specificity": metrics["specificity_decided"],
        "patient_miss_risk": metrics["patient_missed_risk_decided_positive"],
    }


def main() -> None:
    prediction_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for branch, meta in BRANCH_INPUTS.items():
        features = {k: v for k, v in np.load(meta["features"], allow_pickle=False).items()}
        grouped = group_by_eye(read_manifest(meta["manifest"]), features)
        patients = sorted({x["patient_id"] for x in grouped.values()})
        folds = patient_folds(patients, OUTER_FOLDS, SEED)
        branch_predictions: list[dict[str, Any]] = []
        branch_calibration: dict[int, list[dict[str, Any]]] = {}
        branch_validation: dict[int, list[dict[str, Any]]] = {}
        for fold_id, val_patients in enumerate(folds):
            train_patients = sorted(set(patients) - val_patients)
            fit_patients, cal_patients = split_inner_calibration(train_patients, fold_id)
            cal, val, _fit = predict_block(
                grouped,
                fit_patients,
                cal_patients,
                val_patients,
                meta["first_view"],
                meta["second_view"],
                fold_id,
            )
            branch_calibration[fold_id] = cal
            branch_validation[fold_id] = val
            for row in cal + val:
                prediction_rows.append({"branch": branch, **row})
            branch_predictions.extend(val)
            fold_rows.append(
                {
                    "branch": branch,
                    "fold": fold_id,
                    "fit_patients": len(fit_patients),
                    "calibration_patients": len(cal_patients),
                    "validation_patients": len(val_patients),
                    "validation_eyes": len(val),
                }
            )

        for score_key, label in (("first_score", "fixed_one"), ("two_view_score", "fixed_two")):
            for key in ("auroc", "auprc", "brier", "ece10"):
                obs, lo, hi = metric_ci(branch_predictions, lambda r, sk=score_key: binary_metrics(r, sk), key)
                metric_rows.append({"branch": branch, "task": "referable_binary", "model": label, "metric": key, "value": obs, "ci_low": lo, "ci_high": hi})
        for pred_key, label in (("first_grade_pred", "fixed_one"), ("two_view_grade_pred", "fixed_two")):
            for key in ("accuracy", "qwk"):
                obs, lo, hi = metric_ci(branch_predictions, lambda r, pk=pred_key: grade_metrics(r, pk), key)
                metric_rows.append({"branch": branch, "task": "five_class_grade", "model": label, "metric": key, "value": obs, "ci_low": lo, "ci_high": hi})

        for target in TARGET_RISKS:
            aggregate_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for fold_id in range(OUTER_FOLDS):
                cal = branch_calibration[fold_id]
                val = branch_validation[fold_id]
                one_param, _one_metrics = select_threshold(cal, "first_score", target, 1)
                two_param, _two_metrics = select_threshold(cal, "two_view_score", target, 2)
                triage_selected = select_triage(cal, target)
                acq_selected = select_two_stage(cal, target)
                triage_param = triage_selected[:2] if triage_selected[0] is not None else None
                acq_param = acq_selected[:4] if acq_selected[0] is not None else None
                methods = {
                    "fixed_one": (one_param, "fixed_one"),
                    "fixed_two": (two_param, "fixed_two"),
                    "confidence_triage": (triage_param, "confidence_triage"),
                    "two_stage_acquire": (acq_param, "two_stage"),
                }
                for method, (params, apply_name) in methods.items():
                    metrics = apply_policy(val, apply_name, params)
                    if metrics.get("status") != "NO_POLICY":
                        aggregate_by_method[method].extend(metrics["rows"] if "rows" in metrics else [])
            for method, rows in aggregate_by_method.items():
                policy_rows.append(compact_policy(branch, target, method, summarize_decisions(rows)))

    write_csv(Path("results/deepdrid_full_cv_predictions.csv"), prediction_rows)
    write_csv(Path("results/deepdrid_full_cv_metrics.csv"), metric_rows)
    write_csv(Path("results/deepdrid_full_cv_policy.csv"), policy_rows)
    write_csv(Path("results/deepdrid_full_cv_folds.csv"), fold_rows)
    Path("results/deepdrid_full_cv_summary.json").write_text(
        json.dumps(
            {
                "status": "FULL_DEEPDRID_CV_BENCHMARK_COMPLETE",
                "outer_folds": OUTER_FOLDS,
                "inner_calibration_fraction": INNER_CALIBRATION_FRACTION,
                "feature_source": "frozen ResNet-50 ImageNet embeddings",
                "tasks": ["referable_binary", "five_class_grade", "stop_acquire_defer_policy"],
                "artifacts": [
                    "results/deepdrid_full_cv_predictions.csv",
                    "results/deepdrid_full_cv_metrics.csv",
                    "results/deepdrid_full_cv_policy.csv",
                    "results/deepdrid_full_cv_folds.csv",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
