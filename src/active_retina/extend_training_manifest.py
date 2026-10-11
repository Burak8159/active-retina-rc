"""Append train-only rows to an existing autonomy manifest without rewriting it."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = [
    "patient_id", "eye_id", "image_id", "field", "grade", "referable_dr",
    "image_path", "source_split", "original_source_split", "source_dataset",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalize_path(value: str) -> str:
    return value.replace("\\", "/")


def heldout_digest(rows: list[dict[str, str]]) -> str:
    records = sorted(
        "\t".join(row[key] for key in FIELDS)
        for row in rows if row["source_split"].endswith(("_valid", "_test"))
    )
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def extend_manifest(
    base_rows: list[dict[str, str]],
    additions: list[dict[str, Any]],
    dataset: str,
    root: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not base_rows:
        raise ValueError("Base manifest must not be empty")
    if dataset in {"", ".", ".."} or any(ch in dataset for ch in "/\\:"):
        raise ValueError("Dataset namespace must be a simple identifier")

    base_digest = heldout_digest(base_rows)
    by_path: dict[str, dict[str, str]] = {}
    for row in base_rows:
        key = normalize_path(row["image_path"])
        if key in by_path and (row["grade"], row["referable_dr"]) != (
            by_path[key]["grade"], by_path[key]["referable_dr"]
        ):
            raise ValueError(f"Conflicting labels for existing image path: {key}")
        by_path[key] = row

    result = [dict(row) for row in base_rows]
    added = 0
    duplicate = 0
    grade_counts: Counter[int] = Counter()
    for raw in additions:
        split = str(raw.get("source_split", "")).lower()
        if split not in {"train", "training", "ddr_train", "ddr_training"}:
            raise ValueError(f"Addition is not train-only: source_split={split!r}")
        grade = int(raw["grade"])
        label = int(raw.get("referable_dr", grade >= 2))
        if not 0 <= grade <= 5 or label != int(grade >= 2):
            raise ValueError(f"Inconsistent DDR grade/label: grade={grade}, referable_dr={label}")
        path = normalize_path(str(raw["image_path"]))
        if path in by_path:
            existing = by_path[path]
            if (str(grade), str(label)) != (existing["grade"], existing["referable_dr"]):
                raise ValueError(f"Duplicate path has conflicting labels: {path}")
            duplicate += 1
            continue
        if not (root / path).is_file():
            raise ValueError(f"Training image is missing: {path}")
        image_id = f"{dataset}:{raw.get('image_id') or Path(path).stem}"
        patient_id = f"{dataset}:{raw.get('patient_id') or raw.get('image_id') or Path(path).stem}"
        eye_id = f"{dataset}:{raw.get('eye_id') or raw.get('patient_id') or raw.get('image_id') or Path(path).stem}"
        row = {
            "patient_id": patient_id,
            "eye_id": eye_id,
            "image_id": image_id,
            "field": str(raw.get("field") or "single_fundus"),
            "grade": str(grade),
            "referable_dr": str(label),
            "image_path": path,
            "source_split": f"{dataset}_train",
            "original_source_split": str(raw["source_split"]),
            "source_dataset": dataset,
        }
        result.append(row)
        by_path[path] = row
        grade_counts[grade] += 1
        added += 1

    if heldout_digest(result) != base_digest:
        raise AssertionError("Validation/test rows changed during train-only augmentation")
    summary = {
        "status": "TRAIN_ONLY_MANIFEST_EXTENSION_READY",
        "dataset_added": dataset,
        "base_rows": len(base_rows),
        "added_train_rows": added,
        "duplicate_paths_skipped": duplicate,
        "total_rows": len(result),
        "train_rows": sum(row["source_split"].endswith("_train") for row in result),
        "validation_rows": sum(row["source_split"].endswith("_valid") for row in result),
        "test_rows": sum(row["source_split"].endswith("_test") for row in result),
        "added_grade_counts": dict(sorted(grade_counts.items())),
        "heldout_rows_preserved_verbatim": True,
        "heldout_sha256": base_digest,
        "claim_boundary": "Training-only data expansion; held-out development partitions remain unchanged and are not an untouched certification cohort.",
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--add-train", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    base = read_rows(Path(args.base))
    additions = read_rows(Path(args.add_train))
    result, summary = extend_manifest(base, additions, args.dataset, root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(result)
    summary["manifest_output"] = str(output)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
