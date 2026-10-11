"""Extract only the training images and DR grades from the public DR-Grading Parquet.

The source is APTOS 2019-derived. Synthetic context columns are intentionally
ignored; the official validation and test rows are never copied into this manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


def image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw:
            return raw
        path = value.get("path")
        if path:
            return Path(path).read_bytes()
    if isinstance(value, bytes):
        return value
    raise ValueError("Parquet image value has neither embedded bytes nor a readable path")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    parquet = Path(args.parquet)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest_output)
    summary_path = Path(args.summary_output)
    if not parquet.is_file():
        raise SystemExit(f"Parquet not found: {parquet}")
    if output_root.exists() or manifest_path.exists() or summary_path.exists():
        raise SystemExit("Refusing to overwrite existing output")

    table = pq.read_table(parquet)
    required = {"image", "image_id", "dr_grade", "split"}
    missing = required - set(table.column_names)
    if missing:
        raise SystemExit(f"Missing expected columns: {sorted(missing)}")
    rows = table.to_pylist()
    train = [row for row in rows if str(row["split"]).lower() == "train"]
    if not train:
        raise SystemExit("No rows explicitly marked split=train")
    if len({str(row["image_id"]) for row in train}) != len(train):
        raise SystemExit("Duplicate image_id values in the training split")

    output_root.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    grade_counts: Counter[int] = Counter()
    entries: list[dict[str, Any]] = []
    for row in train:
        image_id = str(row["image_id"]).strip()
        if not image_id or any(char in image_id for char in "/\\"):
            raise ValueError(f"Unsafe image_id: {image_id!r}")
        grade = int(row["dr_grade"])
        if grade not in range(5):
            raise ValueError(f"Expected APTOS ICDR grade 0-4, got {grade} for {image_id}")
        raw = image_bytes(row["image"])
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB")
            path = output_root / f"{image_id}.jpg"
            image.save(path, format="JPEG", quality=95, subsampling=0)
        grade_counts[grade] += 1
        entries.append({
            "patient_id": f"aptos_train:{image_id}",
            "eye_id": f"aptos_train:{image_id}",
            "image_id": f"aptos_train:{image_id}",
            "field": "single_fundus",
            "grade": grade,
            "referable_dr": int(grade >= 2),
            "image_path": str(path).replace("\\", "/"),
            "source_split": "aptos_hf_train",
            "original_source_split": "train",
            "source_dataset": "aptos_hf",
        })

    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    summary = {
        "status": "APTOS_TRAIN_ONLY_EXTRACTED",
        "source_repo": "macular/diabetic-retinopathy-grading-africa",
        "source_revision": args.source_revision,
        "source_image_provenance": "APTOS 2019-derived fundus images; synthetic context ignored",
        "source_license_declared": "Apache-2.0 (verify upstream APTOS terms before redistribution)",
        "source_parquet": str(parquet),
        "source_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "n_train_images": len(entries),
        "grade_counts": dict(sorted(grade_counts.items())),
        "output_root": str(output_root),
        "manifest_output": str(manifest_path),
        "used_splits": ["train"],
        "excluded_splits": ["validation", "test"],
        "patient_identity_boundary": "Source contains image identifiers, not patient identifiers; this source is training-only and not used as patient-level validation or certification.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
