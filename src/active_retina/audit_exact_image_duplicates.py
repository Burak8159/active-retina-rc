"""Detect and collapse exact-image duplicates before patient-level evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def resolve(root: Path, raw_path: str) -> Path:
    path = Path(raw_path.replace("\\", "/"))
    if path.is_file():
        return path
    normalized = str(path)
    if "/external/" in normalized:
        candidate = root / "external" / normalized.split("/external/", 1)[1]
        if candidate.is_file():
            return candidate
    candidate = root / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(raw_path)


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def duplicate_groups(rows: list[dict[str, str]], root: Path) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_hash[file_digest(resolve(root, row["image_path"]))].append(row)
    result = []
    split_priority = {"train": 0, "valid": 1, "test": 2}
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        sources = {str(row["source_dataset"]) for row in group}
        grades = {int(row["grade"]) for row in group}
        if len(sources) != 1:
            raise ValueError(f"Exact duplicate spans multiple dataset sources: {sorted(sources)}")
        if len(grades) != 1:
            raise ValueError(f"Exact duplicate has conflicting grades: {sorted(grades)}")
        ordered = sorted(
            group,
            key=lambda row: (
                split_priority.get(str(row["source_split"]).lower().rsplit("_", 1)[-1], 9),
                str(row["image_id"]),
            ),
        )
        result.append({
            "sha256": digest,
            "source_dataset": next(iter(sources)),
            "grade": next(iter(grades)),
            "kept_image_id": ordered[0]["image_id"],
            "removed_image_ids": [row["image_id"] for row in ordered[1:]],
            "members": [
                {"image_id": row["image_id"], "eye_id": row["eye_id"], "patient_id": row["patient_id"], "source_split": row["source_split"]}
                for row in ordered
            ],
        })
    return sorted(result, key=lambda item: item["sha256"])


def remove_duplicate_rows(rows: list[dict[str, str]], groups: list[dict[str, Any]]) -> list[dict[str, str]]:
    removed = {image_id for group in groups for image_id in group["removed_image_ids"]}
    return [row for row in rows if row["image_id"] not in removed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clean-manifest-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()
    manifest, clean_path, summary_path = map(Path, (args.manifest, args.clean_manifest_output, args.summary_output))
    if clean_path.exists() or summary_path.exists():
        raise SystemExit("Refusing to overwrite existing output")
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not rows:
        raise SystemExit("Input manifest is empty or lacks a header")
    groups = duplicate_groups(rows, Path.cwd())
    cleaned = remove_duplicate_rows(rows, groups)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    with clean_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cleaned)
    removed = {image_id for group in groups for image_id in group["removed_image_ids"]}
    summary = {
        "status": "EXACT_DUPLICATES_COLLAPSED" if groups else "NO_EXACT_DUPLICATES",
        "method": "SHA-256 over original image file bytes",
        "input_manifest": str(manifest),
        "clean_manifest": str(clean_path),
        "input_rows": len(rows),
        "removed_duplicate_rows": len(removed),
        "output_rows": len(cleaned),
        "duplicate_groups": groups,
        "split_counts_before": dict(Counter(row["source_split"] for row in rows)),
        "split_counts_after": dict(Counter(row["source_split"] for row in cleaned)),
        "boundary": "Exact byte-identical files only. Near-duplicates are not detected. Existing model and validation results must be recomputed or filtered against the cleaned manifest; this does not make any prior test partition confirmatory.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
