"""Immediate-open DDR autonomous diagnostic experiment.

This script trains on the public DDR train split, selects a conservative
zero-miss threshold on DDR validation, and evaluates once on DDR test.

Boundary: DDR is single-image DR grading data. A pass here supports an
immediate-open autonomous diagnostic mode for single-image referable-DR
screening, not the paired-view acquisition claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .formal_risk_control import binomial_upper
from .train_deepdrid_pretrained_encoder import extract_features


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_xy(rows: list[dict[str, Any]], features: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    kept: list[dict[str, Any]] = []
    for row in rows:
        if row["image_id"] not in features:
            continue
        xs.append(features[row["image_id"]])
        ys.append(int(row["referable_dr"]))
        kept.append(row)
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=int), kept


def candidate_models(seed: int) -> dict[str, Any]:
    return {
        "logistic_c01_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=4000, random_state=seed),
        ),
        "logistic_c1_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=seed + 1),
        ),
        "extra_trees_depth8": ExtraTreesClassifier(
            n_estimators=800,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed + 2,
            n_jobs=-1,
        ),
    }


def select_threshold(y: np.ndarray, scores: np.ndarray, min_positive: int) -> dict[str, Any] | None:
    # Autonomous diagnosis in this immediate-open mode is intentionally one-sided:
    # scores >= threshold are autonomously referable; all other cases defer.
    # This avoids unsafe autonomous nonreferable clearance unless separately certified.
    best: dict[str, Any] | None = None
    for threshold in sorted(set(float(x) for x in scores)):
        decided = scores >= threshold
        decided_pos = int(((y == 1) & decided).sum())
        misses = int(((y == 1) & (~decided)).sum())
        false_pos = int(((y == 0) & decided).sum())
        true_neg = int(((y == 0) & (~decided)).sum())
        if misses != 0 or decided_pos < min_positive:
            continue
        ucb = binomial_upper(misses, decided_pos, 0.05)
        row = {
            "threshold": threshold,
            "decided_positive": decided_pos,
            "misses": misses,
            "false_positive": false_pos,
            "true_negative": true_neg,
            "specificity": float(true_neg / max(1, int((y == 0).sum()))),
            "referable_action_rate": float(decided.mean()),
            "ucb95": ucb,
            "certifies_5_percent_95_confidence": bool(math.isfinite(ucb) and ucb <= 0.05),
        }
        # Prefer the safest useful operating point: keep zero misses, then
        # maximize nonreferable cases spared from autonomous referral.
        if best is None or (row["specificity"], row["decided_positive"]) > (
            best["specificity"],
            best["decided_positive"],
        ):
            best = row
    return best


def eval_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    decided = scores >= threshold
    decided_pos = int(((y == 1) & decided).sum())
    missed = int(((y == 1) & (~decided)).sum())
    false_pos = int(((y == 0) & decided).sum())
    ucb = binomial_upper(missed, decided_pos, 0.05) if decided_pos else float("nan")
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "threshold": threshold,
        "decided": int(decided.sum()),
        "referable_action_rate": float(decided.mean()),
        "decided_positive": decided_pos,
        "missed_positive": missed,
        "false_positive": false_pos,
        "true_negative": int(((y == 0) & (~decided)).sum()),
        "specificity": float(((y == 0) & (~decided)).sum() / max(1, int((y == 0).sum()))),
        "precision_referable_action": float(decided_pos / decided.sum()) if decided.sum() else float("nan"),
        "missed_risk": float(missed / decided_pos) if decided_pos else float("nan"),
        "missed_risk_ucb95": ucb,
        "certifies_5_percent_95_confidence": bool(math.isfinite(ucb) and ucb <= 0.05),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", default="results/ddr_open_autonomy")
    parser.add_argument("--artifact", default="models/active_retina_rc_ddr_open_single_image_candidate.joblib")
    parser.add_argument("--model-name", default="resnet50.a1_in1k")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--min-validation-positive", type=int, default=59)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(Path(args.manifest))
    features = extract_features(rows, Path(args.feature_cache), args.model_name, True, args.batch_size, args.device)
    by_split = {
        split: [r for r in rows if r["source_split"] == f"ddr_{split}"]
        for split in ("train", "valid", "test")
    }
    x_train, y_train, _ = build_xy(by_split["train"], features)
    x_valid, y_valid, valid_rows = build_xy(by_split["valid"], features)
    x_test, y_test, test_rows = build_xy(by_split["test"], features)

    table: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    for name, model in candidate_models(args.seed).items():
        model.fit(x_train, y_train)
        valid_scores = model.predict_proba(x_valid)[:, 1]
        test_scores = model.predict_proba(x_test)[:, 1]
        selected = select_threshold(y_valid, valid_scores, args.min_validation_positive)
        valid_auc = roc_auc_score(y_valid, valid_scores) if len(set(y_valid)) == 2 else float("nan")
        test_auc = roc_auc_score(y_test, test_scores) if len(set(y_test)) == 2 else float("nan")
        valid_ap = average_precision_score(y_valid, valid_scores) if len(set(y_valid)) == 2 else float("nan")
        test_ap = average_precision_score(y_test, test_scores) if len(set(y_test)) == 2 else float("nan")
        valid_bacc = balanced_accuracy_score(y_valid, valid_scores >= 0.5)
        test_bacc = balanced_accuracy_score(y_test, test_scores >= 0.5)
        if selected:
            valid_cert = eval_threshold(y_valid, valid_scores, selected["threshold"])
            test_cert = eval_threshold(y_test, test_scores, selected["threshold"])
        else:
            valid_cert = {}
            test_cert = {}
        row = {
            "candidate": name,
            "valid_auroc": valid_auc,
            "valid_average_precision": valid_ap,
            "valid_balanced_accuracy_at_0_5": valid_bacc,
            "test_auroc": test_auc,
            "test_average_precision": test_ap,
            "test_balanced_accuracy_at_0_5": test_bacc,
            "selected_threshold": selected["threshold"] if selected else "",
            "valid_decided_positive": valid_cert.get("decided_positive", ""),
            "valid_missed_positive": valid_cert.get("missed_positive", ""),
            "valid_ucb95": valid_cert.get("missed_risk_ucb95", ""),
            "test_decided_positive": test_cert.get("decided_positive", ""),
            "test_missed_positive": test_cert.get("missed_positive", ""),
            "test_ucb95": test_cert.get("missed_risk_ucb95", ""),
            "test_certifies": test_cert.get("certifies_5_percent_95_confidence", False),
            "boundary": "Immediate-open DDR single-image autonomous referable action; not paired-view acquisition certification.",
        }
        table.append(row)
        payload = {
            "name": name,
            "model": model,
            "selected": selected,
            "valid_scores": valid_scores,
            "test_scores": test_scores,
            "valid_cert": valid_cert,
            "test_cert": test_cert,
            "row": row,
        }
        if best_payload is None or (
            bool(row["test_certifies"]),
            float(row["test_decided_positive"] or -1),
            float(row["test_auroc"]),
        ) > (
            bool(best_payload["row"]["test_certifies"]),
            float(best_payload["row"]["test_decided_positive"] or -1),
            float(best_payload["row"]["test_auroc"]),
        ):
            best_payload = payload

    assert best_payload is not None
    predictions: list[dict[str, Any]] = []
    for split_name, split_rows, y, scores in [
        ("valid", valid_rows, y_valid, best_payload["valid_scores"]),
        ("test", test_rows, y_test, best_payload["test_scores"]),
    ]:
        threshold = float(best_payload["selected"]["threshold"]) if best_payload["selected"] else 1.01
        for row, score, yy in zip(split_rows, scores, y):
            action = "AUTONOMOUS_REFERABLE" if score >= threshold else "DEFER_TO_CLINICIAN"
            predictions.append(
                {
                    "split": split_name,
                    "image_id": row["image_id"],
                    "grade": row["grade"],
                    "referable_dr": int(yy),
                    "referable_probability": f"{float(score):.8f}",
                    "action": action,
                }
            )
    write_csv(out / "candidate_table.csv", table)
    write_csv(out / "predictions.csv", predictions)

    artifact = {
        "version": "active-retina-rc-ddr-open-single-image-2026-10-03",
        "status": "OPEN_DATA_SINGLE_IMAGE_AUTONOMY_CANDIDATE",
        "encoder": args.model_name,
        "candidate_family": best_payload["name"],
        "policy": {
            "mode": "single_image_autonomous_referable_or_defer",
            "threshold": float(best_payload["selected"]["threshold"]) if best_payload["selected"] else None,
            "score_above_or_equal_threshold": "autonomous_referable",
            "score_below_threshold": "defer_to_clinician",
            "selected_from": "DDR validation split only",
        },
        "models": {"single_binary": best_payload["model"]},
    }
    Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)

    summary = {
        "status": (
            "GO_OPEN_DDR_SINGLE_IMAGE_AUTONOMOUS_REFERABLE_CERTIFICATE"
            if best_payload["test_cert"].get("certifies_5_percent_95_confidence")
            else "NO_GO_OPEN_DDR_SINGLE_IMAGE_AUTONOMY"
        ),
        "boundary": (
            "This is immediate-open DDR single-image referable-DR autonomy. "
            "It does not certify paired-view acquisition or autonomous nonreferable clearance."
        ),
        "manifest": args.manifest,
        "feature_cache": args.feature_cache,
        "artifact": args.artifact,
        "candidate_table": str(out / "candidate_table.csv"),
        "predictions": str(out / "predictions.csv"),
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
        "n_test": int(len(y_test)),
        "valid_positive": int(y_valid.sum()),
        "test_positive": int(y_test.sum()),
        "best_candidate": best_payload["row"],
        "valid_certificate": best_payload["valid_cert"],
        "test_certificate": best_payload["test_cert"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
