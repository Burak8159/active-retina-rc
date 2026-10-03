from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from .common import ImageRecord, read_yaml_like, sha256_file, write_json, write_records_csv


FILENAME_RE = re.compile(r"^(?P<patient>[^_]+)_(?P<anon>[^_]+)_(?P<eye>left|right)_(?P<code>\d{2})\.jpe?g$", re.I)


def _grade_from_row(row: dict[str, str]) -> int:
    for key in ("grade", "label", "DR_grade", "dr_grade", "level"):
        if key in row and row[key] != "":
            return int(row[key])
    raise ValueError(f"Could not find grade column in row keys: {sorted(row)}")


def _filename_from_row(row: dict[str, str]) -> str:
    for key in ("image", "filename", "file", "path", "image_path"):
        if key in row and row[key]:
            return Path(row[key]).name
    for value in row.values():
        if isinstance(value, str) and value.lower().endswith((".jpg", ".jpeg")):
            return Path(value).name
    raise ValueError(f"Could not find image filename in row keys: {sorted(row)}")


def _records_from_wide_row(row: dict[str, str], image_dir: Path, source_split: str, code_map: dict[str, dict[str, str]]) -> list[ImageRecord]:
    grade = _grade_from_row(row)
    records: list[ImageRecord] = []
    for key in ("id1", "id2", "id3", "id4"):
        stem = row.get(key, "")
        if not stem:
            continue
        filename = stem if stem.lower().endswith((".jpg", ".jpeg")) else f"{stem}.jpg"
        match = FILENAME_RE.match(filename)
        if not match:
            raise ValueError(f"Unexpected MFIDDR filename format: {filename}")
        eye = match.group("eye").lower()
        code = match.group("code")
        field = code_map[eye][code]
        image_path = image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"CSV references missing image: {image_path}")
        records.append(
            ImageRecord(
                patient_id=match.group("patient"),
                eye_id=f"{match.group('patient')}_{eye}",
                eye=eye,
                field=field,
                grade=grade,
                image_path=str(image_path),
                source_split=source_split,
            )
        )
    return records


def load_split(csv_path: Path, image_dir: Path, source_split: str, code_map: dict[str, dict[str, str]]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if {"id1", "id2", "id3", "id4"}.issubset(row):
                records.extend(_records_from_wide_row(row, image_dir, source_split, code_map))
                continue
            filename = _filename_from_row(row)
            match = FILENAME_RE.match(filename)
            if not match:
                raise ValueError(f"Unexpected MFIDDR filename format: {filename}")
            eye = match.group("eye").lower()
            code = match.group("code")
            field = code_map[eye][code]
            grade = _grade_from_row(row)
            image_path = image_dir / filename
            if not image_path.exists():
                raise FileNotFoundError(f"CSV references missing image: {image_path}")
            records.append(
                ImageRecord(
                    patient_id=match.group("patient"),
                    eye_id=f"{match.group('patient')}_{eye}",
                    eye=eye,
                    field=field,
                    grade=grade,
                    image_path=str(image_path),
                    source_split=source_split,
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = read_yaml_like(config_path)
    root = Path(cfg["dataset"]["root"])
    required = [
        root / cfg["dataset"]["train_images"],
        root / cfg["dataset"]["test_images"],
        root / cfg["dataset"]["train_csv"],
        root / cfg["dataset"]["test_csv"],
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        status_output = Path(cfg["dataset"].get("status_output", "results/A0_access_schema_status.json"))
        write_json(
            status_output,
            {
                "gate": "A0",
                "status": "BLOCKED",
                "reason": "MFIDDR files are not present locally.",
                "missing": missing,
                "source": cfg["dataset"]["official_source"],
                "field_mapping": cfg["dataset"]["fields"],
            },
        )
        raise SystemExit(f"A0 BLOCKED: MFIDDR files are not present locally. See {status_output}")

    code_map = cfg["dataset"]["filename_field_codes"]
    records = []
    records.extend(load_split(root / cfg["dataset"]["train_csv"], root / cfg["dataset"]["train_images"], "official_train", code_map))
    records.extend(load_split(root / cfg["dataset"]["test_csv"], root / cfg["dataset"]["test_images"], "official_test", code_map))
    grades = sorted({r.grade for r in records})
    expected = cfg["dataset"]["expected_grades"]
    if cfg["dataset"].get("allow_incomplete_grade_set"):
        unexpected = sorted(set(grades) - set(expected))
        if unexpected:
            raise ValueError(f"Unexpected grades {unexpected}; expected subset of {expected}")
    elif grades != expected:
        raise ValueError(f"Unexpected grades {grades}; expected {expected}")

    write_records_csv(Path(cfg["dataset"].get("manifest_output", "results/mfiddr_canonical_manifest.csv")), records)
    write_json(
        Path(cfg["dataset"].get("status_output", "results/A0_access_schema_status.json")),
        {
            "gate": "A0",
            "status": "PASS",
            "n_images": len(records),
            "n_patients": len({r.patient_id for r in records}),
            "grades": grades,
            "field_mapping": cfg["dataset"]["fields"],
            "csv_hashes": {
                cfg["dataset"]["train_csv"]: sha256_file(root / cfg["dataset"]["train_csv"]),
                cfg["dataset"]["test_csv"]: sha256_file(root / cfg["dataset"]["test_csv"]),
            },
        },
    )


if __name__ == "__main__":
    main()
