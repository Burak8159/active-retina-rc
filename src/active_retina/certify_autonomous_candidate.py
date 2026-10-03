"""Run a frozen autonomous candidate on a certification manifest.

The command is designed for a future untouched cohort. It does not tune models
or thresholds. It applies the frozen artifact exactly once and reports whether
the 95%/5% zero-miss positive-patient certificate is satisfied.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .formal_risk_control import binomial_upper
from .train_clinical_artifact import group_by_eye, matrix
from .train_deepdrid_pretrained_encoder import extract_features


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row.get("grade", 0) or 0)
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


def apply_policy(score: float, low: float, high: float) -> tuple[str, int | None]:
    if score <= low:
        return "AUTONOMOUS_NONREFERABLE", 0
    if score >= high:
        return "AUTONOMOUS_REFERABLE", 1
    return "DEFER_TO_CLINICIAN", None


def patient_certificate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    positive_patients = 0
    negative_patients = 0
    decided_positive_patients = 0
    decided_negative_patients = 0
    missed_positive_patients = 0
    false_positive_patients = 0
    decided_patients = 0
    for patient_rows in by_patient.values():
        is_positive = any(int(r["referable_dr"]) == 1 for r in patient_rows)
        positive_patients += int(is_positive)
        negative_patients += int(not is_positive)
        decided_rows = [r for r in patient_rows if r["prediction"] != ""]
        decided_patients += int(bool(decided_rows))
        if is_positive:
            decided_ref = [r for r in patient_rows if int(r["referable_dr"]) == 1 and r["prediction"] != ""]
            if decided_ref:
                decided_positive_patients += 1
                missed_positive_patients += int(any(int(r["prediction"]) == 0 for r in decided_ref))
        elif decided_rows:
            decided_negative_patients += 1
            false_positive_patients += int(any(int(r["prediction"]) == 1 for r in decided_rows))
    ucb = (
        binomial_upper(missed_positive_patients, decided_positive_patients, 0.05)
        if decided_positive_patients
        else float("nan")
    )
    specificity = (
        1.0 - false_positive_patients / decided_negative_patients
        if decided_negative_patients
        else float("nan")
    )
    return {
        "n_patients": len(by_patient),
        "positive_patients": positive_patients,
        "negative_patients": negative_patients,
        "decided_patients": decided_patients,
        "coverage_patients": decided_patients / len(by_patient) if by_patient else float("nan"),
        "decided_positive_patients": decided_positive_patients,
        "decided_negative_patients": decided_negative_patients,
        "missed_positive_patients": missed_positive_patients,
        "false_positive_patients": false_positive_patients,
        "specificity_decided_negative_patients": specificity,
        "patient_missed_risk": (
            missed_positive_patients / decided_positive_patients if decided_positive_patients else float("nan")
        ),
        "patient_missed_risk_ucb_95": ucb,
        "zero_miss_59_case_condition_met": missed_positive_patients == 0 and decided_positive_patients >= 59,
        "certifies_5_percent_95_confidence": math.isfinite(ucb) and ucb <= 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact", default="models/active_retina_rc_autonomous_candidate.joblib")
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--model-name", default="resnet50.a1_in1k")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--certification-context",
        choices=["untouched", "inspected-development"],
        default="untouched",
        help="Use inspected-development for replay/recovery checks that must not be treated as final certificates.",
    )
    args = parser.parse_args()

    artifact = joblib.load(args.artifact)
    policy = artifact["policy"]
    rows = read_manifest(Path(args.manifest))
    features = extract_features(
        rows,
        Path(args.feature_cache),
        args.model_name,
        True,
        args.batch_size,
        args.device,
    )
    items = group_by_eye(rows, features)
    x_pair, _, _ = matrix(items, ("macula", "optic_disc"))
    scores = artifact["models"]["paired_binary"].predict_proba(x_pair)[:, 1]
    pred_rows: list[dict[str, Any]] = []
    for item, score in zip(items, scores):
        action, pred = apply_policy(float(score), float(policy["low_threshold"]), float(policy["high_threshold"]))
        pred_rows.append(
            {
                "patient_id": item["patient_id"],
                "eye_id": item["eye_id"],
                "grade": item["grade"],
                "referable_dr": item["referable_dr"],
                "referable_probability": f"{float(score):.8f}",
                "action": action,
                "prediction": "" if pred is None else pred,
            }
        )
    write_csv(Path(args.output_predictions), pred_rows)
    cert = patient_certificate(pred_rows)
    raw_go = cert["certifies_5_percent_95_confidence"]
    decision = (
        "GO_AUTONOMOUS_5_PERCENT_95_CONFIDENCE_CERTIFIED"
        if raw_go and args.certification_context == "untouched"
        else "DEVELOPMENT_GO_NOT_A_CERTIFICATE"
        if raw_go
        else "NO_GO_AUTONOMOUS_CERTIFICATION"
    )
    summary = {
        "status": "FROZEN_AUTONOMOUS_CANDIDATE_CERTIFICATION_RUN_COMPLETE",
        "artifact": args.artifact,
        "artifact_version": artifact["version"],
        "artifact_status": artifact.get("status", ""),
        "certification_context": args.certification_context,
        "manifest": args.manifest,
        "feature_cache": args.feature_cache,
        "predictions": args.output_predictions,
        "policy": policy,
        "patient_certificate": cert,
        "decision": decision,
        "warning": (
            "This is a final autonomous certificate only when certification_context is untouched "
            "and the cohort was untouched before this run."
        ),
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
