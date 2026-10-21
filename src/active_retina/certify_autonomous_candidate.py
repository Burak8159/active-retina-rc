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

MINIMUM_SPECIFICITY = 0.80


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
    positive_eye_rows = 0
    missed_positive_eye_rows = 0
    predicted_eye_rows = 0
    for patient_rows in by_patient.values():
        is_positive = any(int(r["referable_dr"]) == 1 for r in patient_rows)
        positive_patients += int(is_positive)
        negative_patients += int(not is_positive)
        decided_rows = [r for r in patient_rows if str(r["prediction"]) in {"0", "1"}]
        predicted_eye_rows += len(decided_rows)
        patient_fully_decided = len(decided_rows) == len(patient_rows)
        decided_patients += int(patient_fully_decided)
        positive_eye_rows += sum(int(r["referable_dr"]) == 1 for r in patient_rows)
        missed_positive_eye_rows += sum(
            int(r["referable_dr"]) == 1 and str(r["prediction"]) == "0" for r in patient_rows
        )
        if is_positive:
            if patient_fully_decided:
                decided_positive_patients += 1
                missed_positive_patients += int(not any(str(r["prediction"]) == "1" for r in patient_rows))
        elif patient_fully_decided:
            decided_negative_patients += 1
            false_positive_patients += int(any(str(r["prediction"]) == "1" for r in patient_rows))
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
    row_coverage = predicted_eye_rows / len(rows) if rows else float("nan")
    full_binary_coverage = bool(rows) and predicted_eye_rows == len(rows)
    zero_misses = missed_positive_patients == 0 and missed_positive_eye_rows == 0
    risk_certified = math.isfinite(ucb) and ucb <= 0.05
    specificity_target_met = math.isfinite(specificity) and specificity >= MINIMUM_SPECIFICITY
    full_binary_target_met = full_binary_coverage and zero_misses and risk_certified and specificity_target_met
    return {
        "n_patients": len(by_patient),
        "positive_patients": positive_patients,
        "negative_patients": negative_patients,
        "decided_patients": decided_patients,
        "coverage_patients": decided_patients / len(by_patient) if by_patient else float("nan"),
        "binary_eye_row_coverage": row_coverage,
        "full_binary_eye_row_coverage": full_binary_coverage,
        "positive_eye_rows": positive_eye_rows,
        "missed_positive_eye_rows": missed_positive_eye_rows,
        "decided_positive_patients": decided_positive_patients,
        "decided_negative_patients": decided_negative_patients,
        "missed_positive_patients": missed_positive_patients,
        "false_positive_patients": false_positive_patients,
        "specificity_decided_negative_patients": specificity,
        "patient_missed_risk": (
            missed_positive_patients / decided_positive_patients if decided_positive_patients else float("nan")
        ),
        "patient_missed_risk_ucb_95": ucb,
        "minimum_specificity_target": MINIMUM_SPECIFICITY,
        "zero_miss_59_case_condition_met": (
            full_binary_coverage and zero_misses and decided_positive_patients >= 59
        ),
        "certifies_5_percent_95_confidence": risk_certified,
        "meets_full_binary_80_0_100_research_target": full_binary_target_met,
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
    cert = patient_certificate(pred_rows, args.minimum_specificity)
    decision = (
        "GO_BINARY_80_0_100_RESEARCH_TARGET_MET"
        if cert["meets_full_binary_80_0_100_research_target"] and args.certification_context == "untouched"
        else "DEVELOPMENT_TARGET_MET_NOT_A_CERTIFICATE"
        if cert["meets_full_binary_80_0_100_research_target"]
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
        "clinical_validation_status": "NOT_ESTABLISHED_BY_THIS_COMPUTATIONAL_CERTIFICATE",
        "warning": (
            "A research risk certificate is not clinical validation. Full binary target additionally requires "
            "every eye row to receive a binary output, zero positive-eye and patient misses, the prespecified "
            "patient risk bound, and minimum specificity. Clinical validation requires a separate clinical study."
        ),
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
