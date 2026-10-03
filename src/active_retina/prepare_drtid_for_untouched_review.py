"""Prepare DRTiD for ActiveRetina untouched-cohort review.

Expected DRTiD structure from the official repository:

DRTiD/
  Original Images/
  Ground Truths/
    DR_grade/
      a. DR_grade_Training.csv
      b. DR_grade_Testing.csv

The grade CSVs contain Macula, Optic disc, and Grade columns. If a true patient
identifier column is present, use it. Otherwise this script uses the paired-row
identifier as patient_id and marks the output as pair-level only.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MACULA_COLS = ["Macula", "macula", "macula_image", "macula_image_id"]
DISC_COLS = ["Optic disc", "optic_disc", "disc", "optic_disc_image", "optic_disc_image_id"]
GRADE_COLS = ["Grade", "grade", "dr_grade", "final_icdr"]
PATIENT_COLS = ["patient", "patient_id", "Patient", "ID", "id"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def first_present(columns: set[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def image_path(root: Path, image_id: str) -> str:
    p = Path(image_id)
    if p.suffix:
        return str(root / p)
    return str(root / f"{image_id}.jpg")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["patient_id", "eye_id", "image_id", "field", "grade", "referable_dr", "image_path", "source_split"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drtid-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--include-training", action="store_true", help="Include DRTiD training split as untouched only if not inspected before freeze.")
    parser.add_argument("--include-testing", action="store_true", help="Include DRTiD testing split.")
    args = parser.parse_args()

    root = Path(args.drtid_root)
    image_root = root / "Original Images"
    split_files: list[tuple[str, Path]] = []
    if args.include_training:
        split_files.append(("drtid_training", root / "Ground Truths" / "DR_grade" / "a. DR_grade_Training.csv"))
    if args.include_testing:
        split_files.append(("drtid_testing", root / "Ground Truths" / "DR_grade" / "b. DR_grade_Testing.csv"))
    if not split_files:
        raise SystemExit("Choose at least one of --include-training or --include-testing")

    manifest_rows: list[dict[str, Any]] = []
    split_summaries: list[dict[str, Any]] = []
    used_true_patient_id = True
    for split_name, csv_path in split_files:
        rows = read_csv(csv_path)
        if not rows:
            raise SystemExit(f"{csv_path} has no rows")
        columns = set(rows[0])
        macula_col = first_present(columns, MACULA_COLS)
        disc_col = first_present(columns, DISC_COLS)
        grade_col = first_present(columns, GRADE_COLS)
        patient_col = first_present(columns, PATIENT_COLS)
        missing = [
            name
            for name, value in {"macula": macula_col, "optic_disc": disc_col, "grade": grade_col}.items()
            if value is None
        ]
        if missing:
            raise SystemExit(f"{csv_path} missing required columns: {missing}")
        if patient_col is None:
            used_true_patient_id = False
        for idx, row in enumerate(rows):
            macula_id = row[macula_col].strip()  # type: ignore[index]
            disc_id = row[disc_col].strip()  # type: ignore[index]
            grade = int(float(row[grade_col]))  # type: ignore[index]
            pair_id = f"{split_name}_{idx:05d}_{macula_id}_{disc_id}"
            patient_id = row[patient_col].strip() if patient_col else pair_id  # type: ignore[index]
            eye_id = pair_id
            for field, image_id in [("macula", macula_id), ("optic_disc", disc_id)]:
                manifest_rows.append(
                    {
                        "patient_id": f"drtid_{patient_id}",
                        "eye_id": f"drtid_{eye_id}",
                        "image_id": f"drtid_{image_id}",
                        "field": field,
                        "grade": grade,
                        "referable_dr": 1 if grade >= 2 else 0,
                        "image_path": image_path(image_root, image_id),
                        "source_split": split_name,
                    }
                )
        split_summaries.append(
            {
                "split": split_name,
                "csv": str(csv_path),
                "rows": len(rows),
                "columns_detected": {
                    "macula": macula_col,
                    "optic_disc": disc_col,
                    "grade": grade_col,
                    "patient": patient_col,
                },
            }
        )

    write_csv(Path(args.output_manifest), manifest_rows)
    patients: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eyes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_images: list[str] = []
    for row in manifest_rows:
        patients[row["patient_id"]].append(row)
        eyes[row["eye_id"]].append(row)
        if not Path(row["image_path"]).exists():
            missing_images.append(row["image_path"])
    summary = {
        "status": "DRTID_CERTIFICATION_MANIFEST_PREPARED",
        "output_manifest": args.output_manifest,
        "drtid_root": args.drtid_root,
        "splits": split_summaries,
        "n_manifest_rows": len(manifest_rows),
        "n_eyes_or_pairs": len(eyes),
        "n_patients": len(patients),
        "positive_patients": sum(1 for rs in patients.values() if any(int(r["referable_dr"]) == 1 for r in rs)),
        "grade_counts_pairs": dict(sorted(Counter(int(rs[0]["grade"]) for rs in eyes.values()).items())),
        "missing_image_count": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "true_patient_id_available": used_true_patient_id,
        "certification_boundary": (
            "Patient-level certification is valid only if true patient IDs are present. "
            "If absent, this is pair-level untouched validation and should be reported as such."
        ),
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
