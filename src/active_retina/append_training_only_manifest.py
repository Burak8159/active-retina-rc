"""Append a separately sourced training-only manifest without altering frozen splits."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def combine(base: list[dict[str, str]], added: list[dict[str, str]]) -> list[dict[str, str]]:
    if not base or not added:
        raise ValueError("Both base and training-only manifests must contain rows")
    required = {"patient_id", "eye_id", "image_id", "field", "grade", "referable_dr", "image_path", "source_split", "source_dataset"}
    for name, rows in (("base", base), ("added", added)):
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"{name} manifest missing fields: {sorted(missing)}")
    if any(str(row["source_split"]).lower().rsplit("_", 1)[-1] != "train" for row in added):
        raise ValueError("Added manifest must contain training rows only")
    if any(str(row["source_dataset"]) != "aptos_hf" for row in added):
        raise ValueError("Added rows must be tagged aptos_hf")
    existing_ids = {str(row["image_id"]) for row in base}
    added_ids = [str(row["image_id"]) for row in added]
    if existing_ids.intersection(added_ids) or len(set(added_ids)) != len(added_ids):
        raise ValueError("Image identifiers overlap or duplicate existing rows")
    if any(int(row["grade"]) not in range(5) or int(row["referable_dr"]) != int(int(row["grade"]) >= 2) for row in added):
        raise ValueError("Unexpected added-source grade or referable mapping")
    return [*base, *added]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()
    base_path, added_path = Path(args.base_manifest), Path(args.training_manifest)
    output, summary_path = Path(args.output_manifest), Path(args.summary_output)
    if output.exists() or summary_path.exists():
        raise SystemExit("Refusing to overwrite existing output")
    base_fields, base = load(base_path)
    added_fields, added = load(added_path)
    rows = combine(base, added)
    fields = list(base_fields)
    for field in added_fields:
        if field not in fields:
            fields.append(field)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grade_counts = Counter(int(row["grade"]) for row in added)
    summary: dict[str, Any] = {
        "status": "TRAIN_ONLY_SOURCE_APPENDED_FROZEN_BASE_UNCHANGED",
        "base_manifest": str(base_path),
        "training_manifest": str(added_path),
        "output_manifest": str(output),
        "base_rows": len(base),
        "added_training_rows": len(added),
        "output_rows": len(rows),
        "added_source": "aptos_hf",
        "added_split_counts": dict(Counter(row["source_split"] for row in added)),
        "added_grade_counts": dict(sorted(grade_counts.items())),
        "base_split_counts_unchanged": dict(Counter(row["source_split"] for row in base)),
        "validation_and_test_rows_added": 0,
        "boundary": "APTOS-derived rows are training-only; base validation/test rows remain byte-for-byte field/value unchanged. This does not create independent patient-level evidence.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
