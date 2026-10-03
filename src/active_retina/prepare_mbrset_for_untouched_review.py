"""Prepare mBRSET metadata for ActiveRetina untouched-cohort review.

This script is intentionally conservative. It writes a canonical paired-view
manifest only if the local metadata exposes enough information to identify
same-eye macula and optic-disc images. Otherwise it writes an audit explaining
why mBRSET can support robustness but not paired-view certification yet.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PATIENT_COLS = ["patient", "patient_id", "id_patient"]
FILE_COLS = ["file", "image", "image_id", "filename"]
GRADE_COLS = ["final_icdr", "icdr", "grade", "dr_grade"]
LATERALITY_COLS = ["laterality", "eye", "laterality_eye"]
VIEW_COLS = ["view", "view_position", "field", "center", "anatomical_center"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def first_present(columns: set[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def normalize_view(value: str) -> str | None:
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    if v in {"macula", "macula_centered", "fovea", "fovea_centered"}:
        return "macula"
    if v in {"disc", "optic_disc", "optic_disc_centered", "od"}:
        return "optic_disc"
    return None


def image_path(root: Path, file_value: str) -> str:
    p = Path(file_value)
    if p.suffix:
        candidate = root / p
    else:
        candidate = root / f"{file_value}.jpg"
    return str(candidate)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["patient_id", "eye_id", "image_id", "field", "grade", "referable_dr", "image_path", "source_split"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()

    rows = read_csv(Path(args.labels))
    if not rows:
        raise SystemExit("mBRSET labels file is empty")
    columns = set(rows[0])
    patient_col = first_present(columns, PATIENT_COLS)
    file_col = first_present(columns, FILE_COLS)
    grade_col = first_present(columns, GRADE_COLS)
    laterality_col = first_present(columns, LATERALITY_COLS)
    view_col = first_present(columns, VIEW_COLS)
    missing = [
        name
        for name, value in {
            "patient": patient_col,
            "file": file_col,
            "grade": grade_col,
            "laterality": laterality_col,
            "view_position": view_col,
        }.items()
        if value is None
    ]

    summary: dict[str, Any] = {
        "status": "MBRSET_PAIRED_MANIFEST_NOT_CREATED",
        "labels": args.labels,
        "image_root": args.image_root,
        "n_label_rows": len(rows),
        "columns_detected": {
            "patient": patient_col,
            "file": file_col,
            "grade": grade_col,
            "laterality": laterality_col,
            "view_position": view_col,
        },
        "missing_required_for_paired_manifest": missing,
        "note": (
            "mBRSET can only be used for paired-view untouched certification if metadata identifies "
            "same-eye macula and optic-disc views."
        ),
    }
    if missing:
        Path(args.output_summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    image_root = Path(args.image_root)
    candidate_rows: list[dict[str, Any]] = []
    skipped_unknown_view = 0
    for row in rows:
        field = normalize_view(row[view_col])  # type: ignore[index]
        if field is None:
            skipped_unknown_view += 1
            continue
        patient_id = row[patient_col].strip()  # type: ignore[index]
        laterality = row[laterality_col].strip().lower()  # type: ignore[index]
        eye_id = f"mbrset_{patient_id}_{laterality}"
        file_value = row[file_col].strip()  # type: ignore[index]
        grade = int(float(row[grade_col]))  # type: ignore[index]
        candidate_rows.append(
            {
                "patient_id": f"mbrset_{patient_id}",
                "eye_id": eye_id,
                "image_id": f"mbrset_{file_value}",
                "field": field,
                "grade": grade,
                "referable_dr": 1 if grade >= 2 else 0,
                "image_path": image_path(image_root, file_value),
                "source_split": "mbrset_untouched_candidate",
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[row["eye_id"]].append(row)
    paired_rows: list[dict[str, Any]] = []
    paired_eye_rows: list[list[dict[str, Any]]] = []
    duplicate_field_eyes = 0
    for eye_rows in grouped.values():
        fields = {r["field"] for r in eye_rows}
        if {"macula", "optic_disc"}.issubset(fields):
            # Keep one row per field if duplicates exist.
            chosen = {}
            for row in eye_rows:
                if row["field"] in chosen:
                    duplicate_field_eyes += 1
                chosen.setdefault(row["field"], row)
            pair = [chosen["macula"], chosen["optic_disc"]]
            paired_eye_rows.append(pair)
            paired_rows.extend(pair)
    write_csv(Path(args.output_manifest), paired_rows)
    patients = defaultdict(list)
    for row in paired_rows:
        patients[row["patient_id"]].append(row)
    summary.update(
        {
            "status": "MBRSET_PAIRED_MANIFEST_CREATED",
            "output_manifest": args.output_manifest,
            "candidate_rows_with_known_view": len(candidate_rows),
            "skipped_unknown_view": skipped_unknown_view,
            "paired_manifest_rows": len(paired_rows),
            "paired_eyes": len(paired_rows) // 2,
            "paired_patients": len(patients),
            "paired_eyes_with_duplicate_field_rows": duplicate_field_eyes,
            "positive_paired_patients": sum(
                1 for rs in patients.values() if any(int(r["referable_dr"]) == 1 for r in rs)
            ),
            "grade_counts_paired_eyes": dict(
                sorted(Counter(max(int(r["grade"]) for r in eye_rows) for eye_rows in paired_eye_rows).items())
            ),
        }
    )
    Path(args.output_summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
