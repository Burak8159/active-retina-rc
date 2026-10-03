"""Prepare a new untouched paired-view cohort for certification.

Input CSV columns:
patient_id,eye_id,macula_image_path,optic_disc_image_path,grade,referable_dr

The script writes the canonical two-row-per-eye manifest consumed by
validate_untouched_cohort.py and certify_autonomous_candidate.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED = {
    "patient_id",
    "eye_id",
    "macula_image_path",
    "optic_disc_image_path",
    "grade",
    "referable_dr",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Input CSV has no rows")
    missing = REQUIRED - set(rows[0])
    if missing:
        raise SystemExit(f"Input CSV missing required columns: {sorted(missing)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["patient_id", "eye_id", "image_id", "field", "grade", "referable_dr", "image_path", "source_split"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(rows: list[dict[str, str]], source_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        patient_id = str(row["patient_id"]).strip()
        eye_id = str(row["eye_id"]).strip()
        grade = int(float(row["grade"]))
        referable = int(float(row["referable_dr"]))
        for field, col in [("macula", "macula_image_path"), ("optic_disc", "optic_disc_image_path")]:
            image_path = str(row[col]).strip()
            image_id = f"{eye_id}_{field}"
            out.append(
                {
                    "patient_id": patient_id,
                    "eye_id": eye_id,
                    "image_id": image_id,
                    "field": field,
                    "grade": grade,
                    "referable_dr": referable,
                    "image_path": image_path,
                    "source_split": source_name,
                }
            )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    patients = {}
    eyes = {}
    missing = []
    for row in rows:
        patients.setdefault(row["patient_id"], []).append(row)
        eyes.setdefault(row["eye_id"], []).append(row)
        if not Path(row["image_path"]).exists():
            missing.append(row["image_path"])
    return {
        "n_manifest_rows": len(rows),
        "n_patients": len(patients),
        "n_eyes": len(eyes),
        "positive_patients": sum(
            1 for patient_rows in patients.values() if any(int(r["referable_dr"]) == 1 for r in patient_rows)
        ),
        "referable_eye_rows": sum(int(r["referable_dr"]) for r in rows) // 2,
        "grade_counts_eyes": dict(sorted(Counter(int(rs[0]["grade"]) for rs in eyes.values()).items())),
        "missing_image_count": len(missing),
        "missing_image_examples": missing[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--source-name", default="untouched_external_certification")
    args = parser.parse_args()

    raw = read_csv(Path(args.input))
    manifest = prepare(raw, args.source_name)
    write_csv(Path(args.output_manifest), manifest)
    summary = {
        "status": "UNTOUCHED_CERTIFICATION_MANIFEST_PREPARED",
        "input": args.input,
        "output_manifest": args.output_manifest,
        "source_name": args.source_name,
        **summarize(manifest),
        "next_step": (
            "Run validate_untouched_cohort.py with an attestation JSON. "
            "Only then run certify_autonomous_candidate.py."
        ),
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
