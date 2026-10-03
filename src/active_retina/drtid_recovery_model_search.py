"""Development recovery search after the DRTiD external failure.

This is not an untouched certification command. DRTiD has now been inspected,
so its official train/test split can be used to diagnose and improve model
robustness, while a future untouched cohort remains required for certification.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from .formal_risk_control import binomial_upper
from .train_clinical_artifact import matrix, read_manifest


MANIFEST = Path("results/drtid_untouched_certification/certification_manifest.csv")
FEATURES = Path("results/drtid_untouched_certification/certification_features.npz")
OUT_DIR = Path("results/drtid_recovery_model_search")
TABLE = OUT_DIR / "candidate_table.csv"
PREDICTIONS = OUT_DIR / "candidate_predictions.csv"
SUMMARY = OUT_DIR / "summary.json"
BEST_ARTIFACT = Path("models/active_retina_rc_drtid_recovery_candidate.joblib")
BEST_METADATA = Path("models/active_retina_rc_drtid_recovery_candidate.metadata.json")
SEED = 20261003


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
            {"referable_dr": int(item["referable_dr"]), "prediction": prediction, "deferred": deferred}
        )

    positive_patients = negative_patients = 0
    decided_positive_patients = decided_negative_patients = decided_patients = 0
    missed = false_positive_patients = 0
    for rows in by_patient.values():
        is_positive = any(r["referable_dr"] == 1 for r in rows)
        positive_patients += int(is_positive)
        negative_patients += int(not is_positive)
        decided_rows = [r for r in rows if not r["deferred"]]
        decided_patients += int(bool(decided_rows))
        if is_positive:
            decided_ref = [r for r in rows if r["referable_dr"] == 1 and not r["deferred"]]
            if decided_ref:
                decided_positive_patients += 1
                missed += int(any(r["prediction"] == 0 for r in decided_ref))
        else:
            decided_negative_patients += int(bool(decided_rows))
            false_positive_patients += int(any(r["prediction"] == 1 for r in decided_rows))

    ucb = binomial_upper(missed, decided_positive_patients, 0.05) if decided_positive_patients else float("nan")
    specificity = (
        1.0 - false_positive_patients / decided_negative_patients
        if decided_negative_patients
        else float("nan")
    )
    return {
        "low": float(low),
        "high": float(high),
        "n_patients": len(by_patient),
        "positive_patients": positive_patients,
        "negative_patients": negative_patients,
        "decided_patients": decided_patients,
        "decided_positive_patients": decided_positive_patients,
        "decided_negative_patients": decided_negative_patients,
        "misses": missed,
        "false_positive_patients": false_positive_patients,
        "patient_missed_risk": missed / decided_positive_patients if decided_positive_patients else float("nan"),
        "patient_missed_risk_ucb_95": ucb,
        "specificity_decided_negative_patients": specificity,
        "coverage_patients": decided_patients / len(by_patient) if by_patient else float("nan"),
        "certifies_5_percent_95_confidence": bool(math.isfinite(ucb) and ucb <= 0.05),
    }


def search_modes(items: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 251))
    grid = sorted(set(float(x) for x in quantiles))
    grid = sorted(set([0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0] + grid))
    candidates = [patient_stats(items, scores, low, high) for low in grid for high in grid if low < high]
    useful = [
        c
        for c in candidates
        if c["decided_positive_patients"] >= 59
        and c["misses"] == 0
        and c["decided_negative_patients"] > 0
        and c["specificity_decided_negative_patients"] >= 0.5
    ]
    certified = [c for c in candidates if c["certifies_5_percent_95_confidence"]]
    zero_miss = [c for c in candidates if c["misses"] == 0]
    return {
        "best_useful_zero_miss_59": max(
            useful,
            key=lambda c: (c["specificity_decided_negative_patients"], c["coverage_patients"], c["decided_positive_patients"]),
            default=None,
        ),
        "best_certified_by_coverage": max(
            certified,
            key=lambda c: (c["coverage_patients"], c["specificity_decided_negative_patients"]),
            default=None,
        ),
        "best_zero_miss_by_positive_count": max(
            zero_miss,
            key=lambda c: (c["decided_positive_patients"], c["coverage_patients"]),
            default=None,
        ),
        "n_threshold_candidates": len(candidates),
    }


def candidate_models() -> dict[str, Any]:
    return {
        "logistic_l2_c001_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.01, class_weight="balanced", max_iter=4000, random_state=SEED),
        ),
        "logistic_l2_c01_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=4000, random_state=SEED + 1),
        ),
        "logistic_l2_c1_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=SEED + 2),
        ),
        "normalized_logistic_l2_c01": make_pipeline(
            Normalizer(),
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=4000, random_state=SEED + 3),
        ),
        "calibrated_linear_svc": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                LinearSVC(C=0.05, class_weight="balanced", random_state=SEED + 4, max_iter=5000),
                method="sigmoid",
                cv=5,
            ),
        ),
        "extra_trees_depth6": ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=SEED + 5,
            n_jobs=-1,
        ),
        "random_forest_depth8": RandomForestClassifier(
            n_estimators=800,
            max_depth=8,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=SEED + 6,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(
                max_iter=250,
                max_leaf_nodes=15,
                learning_rate=0.03,
                l2_regularization=0.2,
                random_state=SEED + 7,
            ),
        ),
    }


def split_items() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features = {k: v for k, v in np.load(FEATURES, allow_pickle=False).items()}
    rows = read_manifest(MANIFEST)
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["image_id"] not in features:
            continue
        item = grouped.setdefault(
            row["eye_id"],
            {
                "patient_id": row["patient_id"],
                "eye_id": row["eye_id"],
                "grade": row["grade"],
                "referable_dr": row["referable_dr"],
                "source_split": row["source_split"],
                "views": {},
            },
        )
        item["views"][row["field"]] = features[row["image_id"]]
    items = [x for x in grouped.values() if "macula" in x["views"] and "optic_disc" in x["views"]]
    train = [item for item in items if str(item.get("source_split", "")).endswith("training")]
    test = [item for item in items if str(item.get("source_split", "")).endswith("testing")]
    if not train or not test:
        raise SystemExit("DRTiD manifest must contain training and testing source_split values.")
    return train, test


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_items, test_items = split_items()
    x_train, y_train, _ = matrix(train_items, ("macula", "optic_disc"))
    x_test, y_test, _ = matrix(test_items, ("macula", "optic_disc"))

    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    for name, model in candidate_models().items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        scores = model.predict_proba(x_test)[:, 1]
        pred = (scores >= 0.5).astype(int)
        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
        bal_acc = balanced_accuracy_score(y_test, pred)
        modes = search_modes(test_items, scores)
        useful = modes["best_useful_zero_miss_59"] or {}
        cert = modes["best_certified_by_coverage"] or {}
        zero = modes["best_zero_miss_by_positive_count"] or {}
        rows.append(
            {
                "candidate": name,
                "train_split": "drtid_training",
                "test_split": "drtid_testing_inspected_after_external_failure",
                "auroc": auc,
                "average_precision": ap,
                "balanced_accuracy_at_0_5": bal_acc,
                "best_useful_zero_miss_59_decided_positive_patients": useful.get("decided_positive_patients", ""),
                "best_useful_zero_miss_59_specificity": useful.get("specificity_decided_negative_patients", ""),
                "best_useful_zero_miss_59_coverage": useful.get("coverage_patients", ""),
                "best_useful_zero_miss_59_low": useful.get("low", ""),
                "best_useful_zero_miss_59_high": useful.get("high", ""),
                "best_certified_coverage": cert.get("coverage_patients", ""),
                "best_certified_ucb95": cert.get("patient_missed_risk_ucb_95", ""),
                "best_certified_misses": cert.get("misses", ""),
                "best_zero_miss_decided_positive_patients": zero.get("decided_positive_patients", ""),
                "best_zero_miss_ucb95": zero.get("patient_missed_risk_ucb_95", ""),
                "best_zero_miss_coverage": zero.get("coverage_patients", ""),
                "development_boundary": "DRTiD is inspected and cannot be reused as final untouched certification after this search.",
            }
        )
        for item, score in zip(test_items, scores):
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

    def score_row(row: dict[str, Any]) -> tuple[float, float, float]:
        useful_pos = float(row["best_useful_zero_miss_59_decided_positive_patients"] or -1)
        useful_spec = float(row["best_useful_zero_miss_59_specificity"] or -1)
        return (useful_pos, useful_spec, float(row["auroc"]))

    best = max(rows, key=score_row)
    write_csv(TABLE, rows)
    write_csv(PREDICTIONS, prediction_rows)

    best_name = str(best["candidate"])
    best_model = fitted_models[best_name]
    policy = None
    if best["best_useful_zero_miss_59_low"] != "":
        policy = {
            "low_threshold": float(best["best_useful_zero_miss_59_low"]),
            "high_threshold": float(best["best_useful_zero_miss_59_high"]),
            "selected_from": "DRTiD official train-to-test recovery search after external failure",
            "decision_rule": "low=autonomous_nonreferable, high=autonomous_referable, middle=defer_to_clinician",
        }
        BEST_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": "active-retina-rc-drtid-recovery-candidate-2026-10-03",
                "status": "RECOVERY_CANDIDATE_NOT_UNTOUCHED_CERTIFIED",
                "encoder": "resnet50.a1_in1k",
                "view_order": {"first": "macula", "second": "optic_disc"},
                "candidate_family": best_name,
                "policy": policy,
                "models": {"paired_binary": best_model},
            },
            BEST_ARTIFACT,
        )
        BEST_METADATA.write_text(
            json.dumps(
                {
                    "status": "RECOVERY_CANDIDATE_NOT_UNTOUCHED_CERTIFIED",
                    "artifact": str(BEST_ARTIFACT),
                    "candidate_family": best_name,
                    "policy": policy,
                    "boundary": "Selected after inspecting DRTiD test results; must be certified on a different untouched cohort.",
                    "best_row": best,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    summary = {
        "status": "DRTID_RECOVERY_SEARCH_COMPLETE",
        "boundary": "Development after external failure; not a clinical certificate.",
        "train_patients": len(train_items),
        "test_patients": len(test_items),
        "test_positive_patients": int(sum(item["referable_dr"] for item in test_items)),
        "table": str(TABLE),
        "predictions": str(PREDICTIONS),
        "best_candidate": best,
        "recovery_artifact": str(BEST_ARTIFACT) if policy else None,
        "next_gate": (
            "If recovery artifact exists, run it once on a different untouched cohort such as mBRSET. "
            "If no useful zero-miss mode exists, fine-tune the encoder rather than changing thresholds."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
