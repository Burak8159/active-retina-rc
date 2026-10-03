"""Build and evaluate a sealed DeepDRiD online-certification cohort.

This uses the official DeepDRiD Online-Challenge1&2-Evaluation labels that are
present locally. It does not tune thresholds on this cohort. It evaluates the
locked clinical artifact as-is and reports whether the 59-positive zero-miss
condition is met.
"""
from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import joblib
import numpy as np

from .clinical_case_runner import build_encoder, classify, embed, enrich_decision
from .formal_risk_control import binomial_upper


ROOT = Path("external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/Online-Challenge1&2-Evaluation")
CH1_LABELS = ROOT / "Challenge1_labels.xlsx"
MANIFEST = Path("results/deepdrid_online_certification_manifest.csv")
PREDICTIONS = Path("results/deepdrid_online_certification_predictions.csv")
SUMMARY = Path("results/deepdrid_online_certification_summary.json")
ARTIFACT = Path("models/active_retina_rc_clinical_artifact.joblib")


def read_xlsx_first_sheet(path: Path) -> list[dict[str, str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                text = "".join(t.text or "" for t in si.findall(".//a:t", ns))
                shared.append(text)
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows: list[list[str]] = []
    for row in sheet.findall(".//a:sheetData/a:row", ns):
        vals: list[str] = []
        for c in row.findall("a:c", ns):
            cell_type = c.attrib.get("t")
            v = c.find("a:v", ns)
            value = "" if v is None or v.text is None else v.text
            if cell_type == "s" and value:
                value = shared[int(value)]
            vals.append(value)
        rows.append(vals)
    if not rows:
        return []
    header = rows[0]
    out = []
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        out.append({h: padded[i] for i, h in enumerate(header)})
    return out


def eye_from_image_id(image_id: str) -> str:
    return "left" if "_l" in image_id else "right"


def field_from_image_id(image_id: str) -> str:
    return "optic_disc" if image_id.endswith("1") else "macula"


def build_manifest() -> list[dict[str, Any]]:
    labels = read_xlsx_first_sheet(CH1_LABELS)
    rows: list[dict[str, Any]] = []
    for row in labels:
        image_id = str(row.get("image_id", "")).strip()
        raw = str(row.get("DR_Level", row.get("DR_Levels", ""))).strip()
        if not image_id or raw == "":
            continue
        grade = int(float(raw))
        patient_id = image_id.split("_")[0]
        path = ROOT / "Images" / patient_id / f"{image_id}.jpg"
        rows.append(
            {
                "patient_id": patient_id,
                "eye_id": f"{patient_id}_{eye_from_image_id(image_id)}",
                "image_id": image_id,
                "field": field_from_image_id(image_id),
                "grade": grade,
                "referable_dr": 1 if grade >= 2 else 0,
                "image_path": str(path),
                "source_split": "online_challenge_eval",
            }
        )
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


def grouped_eyes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = grouped.setdefault(
            row["eye_id"],
            {
                "patient_id": row["patient_id"],
                "eye_id": row["eye_id"],
                "grade": int(row["grade"]),
                "referable_dr": int(row["referable_dr"]),
                "views": {},
            },
        )
        item["views"][row["field"]] = row["image_path"]
    return [x for x in grouped.values() if "macula" in x["views"] and "optic_disc" in x["views"]]


def patient_miss_summary(preds: list[dict[str, Any]]) -> dict[str, Any]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preds:
        by_patient[row["patient_id"]].append(row)
    positive = 0
    decided_positive = 0
    missed = 0
    for rows in by_patient.values():
        if any(int(r["referable_dr"]) == 1 for r in rows):
            positive += 1
        decided = [r for r in rows if int(r["referable_dr"]) == 1 and not bool(r["deferred"])]
        if decided:
            decided_positive += 1
            if any(int(r["prediction"]) == 0 for r in decided):
                missed += 1
    ucb = binomial_upper(missed, decided_positive, 0.05) if decided_positive else float("nan")
    return {
        "positive_patients": positive,
        "decided_positive_patients": decided_positive,
        "missed_decided_positive_patients": missed,
        "patient_missed_risk": missed / decided_positive if decided_positive else float("nan"),
        "patient_missed_risk_ucb_95": ucb,
        "certifies_5_percent_95_confidence": math.isfinite(ucb) and ucb <= 0.05,
        "zero_miss_59_case_condition_met": missed == 0 and decided_positive >= 59,
    }


def main() -> None:
    rows = build_manifest()
    write_csv(MANIFEST, rows)
    eyes = grouped_eyes(rows)
    artifact = joblib.load(ARTIFACT)
    model, transform = build_encoder()
    preds: list[dict[str, Any]] = []
    for eye in eyes:
        first = embed(Path(eye["views"]["macula"]), model, transform)
        second = embed(Path(eye["views"]["optic_disc"]), model, transform)
        decision = enrich_decision(classify(artifact, first, second))
        action = decision["action"]
        deferred = action == "DEFER_TO_CLINICIAN" or decision.get("needs_second_view", False)
        prediction = 1 if decision["referable_probability"] >= 0.5 else 0
        if "low_referable" in decision["recommendation"]:
            prediction = 0
        if "likely_referable" in decision["recommendation"]:
            prediction = 1
        preds.append(
            {
                "patient_id": eye["patient_id"],
                "eye_id": eye["eye_id"],
                "grade": eye["grade"],
                "referable_dr": eye["referable_dr"],
                "action": action,
                "deferred": int(deferred),
                "prediction": prediction,
                "referable_probability": decision["referable_probability"],
                "predicted_dr_grade": decision["predicted_dr_grade"],
                "recommendation": decision["recommendation"],
            }
        )
    write_csv(PREDICTIONS, preds)
    referable = [p for p in preds if int(p["referable_dr"]) == 1]
    decided_ref = [p for p in referable if not int(p["deferred"])]
    missed = [p for p in decided_ref if int(p["prediction"]) == 0]
    summary = {
        "status": "ONLINE_CERTIFICATION_EVALUATED_LOCKED_ARTIFACT_NO_THRESHOLD_TUNING",
        "manifest": str(MANIFEST),
        "predictions": str(PREDICTIONS),
        "n_images": len(rows),
        "n_complete_eyes": len(eyes),
        "n_patients": len({r["patient_id"] for r in rows}),
        "grade_counts_images": dict(sorted(Counter(int(r["grade"]) for r in rows).items())),
        "referable_eye_count": len(referable),
        "decided_referable_eye_count": len(decided_ref),
        "missed_decided_referable_eye_count": len(missed),
        "eye_missed_risk": len(missed) / len(decided_ref) if decided_ref else float("nan"),
        "eye_missed_risk_ucb_95": binomial_upper(len(missed), len(decided_ref), 0.05) if decided_ref else float("nan"),
        "patient_certificate": patient_miss_summary(preds),
        "decision": None,
    }
    pc = summary["patient_certificate"]
    summary["decision"] = (
        "GO_AUTONOMOUS_5_PERCENT_CERTIFIED"
        if pc["certifies_5_percent_95_confidence"]
        else "NO_GO_AUTONOMOUS_CERTIFICATION_FAILED_ON_ONLINE_COHORT"
    )
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
