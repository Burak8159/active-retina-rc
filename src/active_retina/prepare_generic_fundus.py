from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from .common import ImageRecord, write_json, write_records_csv


@dataclass(frozen=True)
class ColumnMap:
    image: str
    grade: str
    patient_id: str | None
    eye: str | None
    field: str | None
    split: str | None


def _value(row: dict[str, str], key: str | None, default: str) -> str:
    if key is None:
        return default
    value = row.get(key, "")
    return value if value != "" else default


def _grade(row: dict[str, str], key: str) -> int:
    raw = row[key].strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return int(raw)


def _make_record(row: dict[str, str], image_root: Path, columns: ColumnMap, default_field: str, default_split: str) -> ImageRecord:
    rel = Path(row[columns.image])
    image_path = image_root / rel
    patient_id = _value(row, columns.patient_id, rel.stem)
    eye = _value(row, columns.eye, "unknown").lower()
    field = _value(row, columns.field, default_field)
    source_split = _value(row, columns.split, default_split)
    eye_id = f"{patient_id}_{eye}" if eye != "unknown" else patient_id
    return ImageRecord(
        patient_id=patient_id,
        eye_id=eye_id,
        eye=eye,
        field=field,
        grade=_grade(row, columns.grade),
        image_path=str(image_path),
        source_split=source_split,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a canonical manifest for locally downloaded public fundus datasets.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--image-column", required=True)
    parser.add_argument("--grade-column", required=True)
    parser.add_argument("--patient-column")
    parser.add_argument("--eye-column")
    parser.add_argument("--field-column")
    parser.add_argument("--split-column")
    parser.add_argument("--default-field", default="fundus")
    parser.add_argument("--default-split", default="unspecified")
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--status-output", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    image_root = Path(args.image_root)
    columns = ColumnMap(
        image=args.image_column,
        grade=args.grade_column,
        patient_id=args.patient_column,
        eye=args.eye_column,
        field=args.field_column,
        split=args.split_column,
    )
    records: list[ImageRecord] = []
    missing: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            record = _make_record(row, image_root, columns, args.default_field, args.default_split)
            if not Path(record.image_path).exists():
                missing.append(record.image_path)
            records.append(record)

    status = {
        "gate": "A0_PUBLIC",
        "dataset": args.dataset_name,
        "status": "PASS" if not missing else "BLOCKED",
        "n_rows": len(records),
        "n_patients": len({r.patient_id for r in records}),
        "n_eyes": len({r.eye_id for r in records}),
        "grades": sorted({r.grade for r in records}),
        "fields": sorted({r.field for r in records}),
        "missing_images_count": len(missing),
        "missing_images_sample": missing[:20],
        "manifest_output": args.manifest_output,
    }
    if missing:
        write_json(Path(args.status_output), status)
        raise SystemExit(f"{args.dataset_name} A0_PUBLIC BLOCKED: {len(missing)} referenced images are missing.")

    write_records_csv(Path(args.manifest_output), records)
    write_json(Path(args.status_output), status)


if __name__ == "__main__":
    main()
