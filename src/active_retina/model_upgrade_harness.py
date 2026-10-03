"""Development-only model upgrade harness for certification recovery.

This script is intentionally not a certification run. It uses already inspected
data to compare stronger classifier heads and identify candidates worth freezing
before a future untouched certification cohort.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .deepdrid_online_certification import build_manifest
from .formal_risk_control import binomial_upper
from .train_clinical_artifact import FEATURES, MANIFEST, matrix, read_manifest, group_by_eye
from .train_deepdrid_pretrained_encoder import extract_features


OUTPUT = Path("results/model_upgrade_harness_summary.json")
TABLE = Path("results/model_upgrade_harness_table.csv")
PREDICTIONS = Path("results/model_upgrade_harness_predictions.csv")
ONLINE_FEATURES = Path("results/deepdrid_online_resnet50_features.npz")
SEED = 20261003


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_training() -> tuple[np.ndarray, np.ndarray]:
    features = {k: v for k, v in np.load(FEATURES, allow_pickle=False).items()}
    rows = read_manifest(MANIFEST)
    items = group_by_eye(rows, features)
    x_pair, y_ref, _ = matrix(items, ("macula", "optic_disc"))
    return x_pair, y_ref


def load_online_features() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows = build_manifest()
    features = extract_features(
        rows,
        ONLINE_FEATURES,
        "resnet50.a1_in1k",
        True,
        batch_size=16,
        device_name="auto",
    )
    items = group_by_eye(rows, features)
    x_pair, y_ref, _ = matrix(items, ("macula", "optic_disc"))
    return x_pair, y_ref, items


def candidate_models() -> dict[str, Any]:
    return {
        "logistic_c025_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, class_weight="balanced", max_iter=3000, random_state=SEED),
        ),
        "logistic_c1_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED + 1),
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED + 2),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=600,
            max_depth=6,
            class_weight="balanced_subsample",
            random_state=SEED + 3,
            n_jobs=-1,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=800,
            max_depth=8,
            class_weight="balanced",
            random_state=SEED + 4,
            n_jobs=-1,
        ),
    }


def patient_stats(items: list[dict[str, Any]], scores: np.ndarray, low: float, high: float) -> dict[str, Any]:
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for item, score in zip(items, scores):
        if score <= low:
            prediction = 0
            deferred = False
        elif score >= high:
            prediction = 1
            deferred = False
        else:
            prediction = None
            deferred = True
        by_patient.setdefault(str(item["patient_id"]), []).append(
            {
                "referable_dr": int(item["referable_dr"]),
                "prediction": prediction,
                "deferred": deferred,
            }
        )
    positive_patients = 0
    negative_patients = 0
    decided_positive_patients = 0
    decided_negative_patients = 0
    missed = 0
    false_positive_patients = 0
    decided_patients = 0
    for rows in by_patient.values():
        is_positive = any(r["referable_dr"] == 1 for r in rows)
        positive_patients += int(is_positive)
        negative_patients += int(not is_positive)
        decided_rows = [r for r in rows if not r["deferred"]]
        if decided_rows:
            decided_patients += 1
        if is_positive:
            decided_ref = [r for r in rows if r["referable_dr"] == 1 and not r["deferred"]]
            if decided_ref:
                decided_positive_patients += 1
                if any(r["prediction"] == 0 for r in decided_ref):
                    missed += 1
        else:
            decided_negative_patients += int(bool(decided_rows))
            if any(r["prediction"] == 1 for r in decided_rows):
                false_positive_patients += 1
    ucb = binomial_upper(missed, decided_positive_patients, 0.05) if decided_positive_patients else float("nan")
    specificity = (
        1.0 - false_positive_patients / decided_negative_patients
        if decided_negative_patients
        else float("nan")
    )
    return {
        "low": low,
        "high": high,
        "positive_patients": positive_patients,
        "negative_patients": negative_patients,
        "decided_positive_patients": decided_positive_patients,
        "decided_negative_patients": decided_negative_patients,
        "misses": missed,
        "false_positive_patients": false_positive_patients,
        "patient_missed_risk": missed / decided_positive_patients if decided_positive_patients else float("nan"),
        "patient_missed_risk_ucb_95": ucb,
        "certifies_5_percent_95_confidence": math.isfinite(ucb) and ucb <= 0.05,
        "specificity_decided_negative_patients": specificity,
        "coverage_patients": decided_patients / len(by_patient) if by_patient else float("nan"),
        "defer_patients": 1.0 - decided_patients / len(by_patient) if by_patient else float("nan"),
        "n_patients": len(by_patient),
    }


def search_modes(items: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    grid = sorted(set(float(x) for x in np.quantile(scores, np.linspace(0.0, 1.0, 101))))
    grid = sorted(set([0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0] + grid))
    candidates: list[dict[str, Any]] = []
    for low in grid:
        for high in grid:
            if low >= high:
                continue
            candidates.append(patient_stats(items, scores, low, high))
    zero_miss = [c for c in candidates if c["misses"] == 0]
    zero_miss_specificity_ge_50 = [
        c for c in zero_miss if c["decided_negative_patients"] > 0 and c["specificity_decided_negative_patients"] >= 0.5
    ]
    zero_miss_specificity_ge_80 = [
        c for c in zero_miss if c["decided_negative_patients"] > 0 and c["specificity_decided_negative_patients"] >= 0.8
    ]
    certified = [c for c in candidates if c["certifies_5_percent_95_confidence"]]
    return {
        "n_candidates": len(candidates),
        "best_zero_miss": max(zero_miss, key=lambda c: (c["decided_positive_patients"], c["coverage_patients"]), default=None),
        "best_zero_miss_specificity_ge_50": max(
            zero_miss_specificity_ge_50,
            key=lambda c: (c["decided_positive_patients"], c["coverage_patients"], c["specificity_decided_negative_patients"]),
            default=None,
        ),
        "best_zero_miss_specificity_ge_80": max(
            zero_miss_specificity_ge_80,
            key=lambda c: (c["decided_positive_patients"], c["coverage_patients"], c["specificity_decided_negative_patients"]),
            default=None,
        ),
        "best_certified": max(certified, key=lambda c: c["coverage_patients"], default=None),
        "top_zero_miss": sorted(
            zero_miss,
            key=lambda c: (-c["decided_positive_patients"], -c["coverage_patients"]),
        )[:5],
    }


def default_threshold_stats(items: list[dict[str, Any]], scores: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    return patient_stats(items, scores, threshold, threshold + 1e-12)


def main() -> None:
    x_train, y_train = load_training()
    x_online, y_online, online_items = load_online_features()
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for name, model in candidate_models().items():
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_online)[:, 1]
        auc = roc_auc_score(y_online, scores) if len(set(y_online)) == 2 else float("nan")
        ap = average_precision_score(y_online, scores) if len(set(y_online)) == 2 else float("nan")
        default_stats = default_threshold_stats(online_items, scores)
        mode_search = search_modes(online_items, scores)
        best_zero = mode_search["best_zero_miss"] or {}
        best_spec50 = mode_search["best_zero_miss_specificity_ge_50"] or {}
        best_spec80 = mode_search["best_zero_miss_specificity_ge_80"] or {}
        rows.append(
            {
                "candidate": name,
                "feature_source": "deepdrid_resnet50_pair_features",
                "online_auroc_dev_only": auc,
                "online_average_precision_dev_only": ap,
                "default_threshold_misses": default_stats["misses"],
                "default_threshold_decided_positive_patients": default_stats["decided_positive_patients"],
                "best_zero_miss_decided_positive_patients": best_zero.get("decided_positive_patients", ""),
                "best_zero_miss_ucb95": best_zero.get("patient_missed_risk_ucb_95", ""),
                "best_zero_miss_coverage_patients": best_zero.get("coverage_patients", ""),
                "best_zero_miss_specificity": best_zero.get("specificity_decided_negative_patients", ""),
                "best_zero_miss_specificity_ge_50_decided_positive_patients": best_spec50.get("decided_positive_patients", ""),
                "best_zero_miss_specificity_ge_50_ucb95": best_spec50.get("patient_missed_risk_ucb_95", ""),
                "best_zero_miss_specificity_ge_50_coverage": best_spec50.get("coverage_patients", ""),
                "best_zero_miss_specificity_ge_50_specificity": best_spec50.get("specificity_decided_negative_patients", ""),
                "best_zero_miss_specificity_ge_80_decided_positive_patients": best_spec80.get("decided_positive_patients", ""),
                "best_zero_miss_specificity_ge_80_ucb95": best_spec80.get("patient_missed_risk_ucb_95", ""),
                "best_zero_miss_specificity_ge_80_coverage": best_spec80.get("coverage_patients", ""),
                "best_zero_miss_specificity_ge_80_specificity": best_spec80.get("specificity_decided_negative_patients", ""),
                "certified_posthoc": bool(mode_search["best_certified"]),
                "note": "development-only; online cohort has been inspected and cannot certify a frozen product",
            }
        )
        for item, score in zip(online_items, scores):
            prediction_rows.append(
                {
                    "candidate": name,
                    "patient_id": item["patient_id"],
                    "eye_id": item["eye_id"],
                    "grade": item["grade"],
                    "referable_dr": item["referable_dr"],
                    "referable_probability": f"{score:.8f}",
                }
            )
    write_csv(TABLE, rows)
    write_csv(PREDICTIONS, prediction_rows)
    best_row = max(
        rows,
        key=lambda r: (
            float(r["best_zero_miss_specificity_ge_50_decided_positive_patients"] or -1),
            float(r["online_auroc_dev_only"]),
        ),
    )
    summary = {
        "status": "DEVELOPMENT_ONLY_MODEL_UPGRADE_HARNESS_EVALUATED",
        "warning": "The online cohort is inspected. These numbers are model-development evidence, not a prospective risk certificate.",
        "training_feature_cache": str(FEATURES),
        "online_feature_cache": str(ONLINE_FEATURES),
        "predictions": str(PREDICTIONS),
        "table": str(TABLE),
        "n_training_eyes": int(len(y_train)),
        "n_online_eyes": int(len(y_online)),
        "n_online_positive_eyes": int(y_online.sum()),
        "n_online_positive_patients": int(len({item["patient_id"] for item in online_items if int(item["referable_dr"]) == 1})),
        "best_development_candidate": best_row,
        "certificate_blocker_after_upgrade": (
            "Even a zero-miss post-hoc mode on the inspected online cohort must reach at least 59 decided positive patients "
            "for a 95% one-sided 5% missed-DR certificate. The online cohort has 48 positive patients total."
        ),
        "next_required_step": (
            "Freeze one candidate and run it once on a new untouched certification cohort with at least 59, preferably 80-120, "
            "referable-DR positive patients."
        ),
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
