"""Build the ActiveRetina autonomy-route training manifest.

The route is deliberately provenance-first. It combines already prepared local
public manifests, normalizes split names, drops missing images, and writes the
exact cohort composition used by the next model run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = [
    "patient_id",
    "eye_id",
    "image_id",
    "field",
    "grade",
    "referable_dr",
    "image_path",
    "source_split",
    "original_source_split",
    "source_dataset",
]


DEFAULT_INPUTS = [
    ("ddr_hf_specificity300", "results/ddr_hf_specificity300_manifest.csv"),
    ("drtid_paired", "results/drtid_untouched_certification/certification_manifest.csv"),
]


def canonical_split(value: str) -> str | None:
    split = value.lower()
    if split in {"train", "training"} or split.endswith("_train") or split.endswith("_training"):
        return "train"
    if split in {"valid", "validation", "val"} or split.endswith("_valid") or split.endswith("_validation") or split.endswith("_val"):
        return "valid"
    if split in {"test", "testing"} or split.endswith("_test") or split.endswith("_testing"):
        return "test"
    return None


def read_manifest(dataset: str, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            split = canonical_split(str(row.get("source_split", "")))
            if split is None:
                continue
            image_path = str(row["image_path"])
            image_id = str(row.get("image_id") or Path(image_path).stem)
            rows.append(
                {
                    "patient_id": f"{dataset}:{row.get('patient_id') or image_id}",
                    "eye_id": f"{dataset}:{row.get('eye_id') or row.get('patient_id') or image_id}",
                    "image_id": f"{dataset}:{image_id}",
                    "field": row.get("field") or "single_fundus",
                    "grade": int(row["grade"]),
                    "referable_dr": int(row.get("referable_dr", 1 if int(row["grade"]) >= 2 else 0)),
                    "image_path": image_path,
                    "source_split": f"{dataset}_{split}",
                    "original_source_split": str(row.get("source_split", "")),
                    "source_dataset": dataset,
                }
            )
    return rows


def drop_incomplete_paired_eyes(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row["source_dataset"]).endswith("_paired"):
            grouped.setdefault((str(row["source_dataset"]), str(row["eye_id"])), []).append(row)
    incomplete = {
        key for key, eye_rows in grouped.items()
        if {str(row["field"]) for row in eye_rows} != {"macula", "optic_disc"}
    }
    counts: dict[str, int] = {}
    for dataset, _eye_id in incomplete:
        counts[dataset] = counts.get(dataset, 0) + 1
    return [
        row for row in rows
        if not (str(row["source_dataset"]).endswith("_paired")
                and (str(row["source_dataset"]), str(row["eye_id"])) in incomplete)
    ], counts


def image_exists(root: Path, image_path: str) -> bool:
    image_path = normalize_image_path(image_path)
    path = Path(image_path)
    return path.exists() or (root / image_path).exists()


def normalize_image_path(image_path: str) -> str:
    normalized = image_path.replace("\\", "/")
    marker = "/external/"
    if marker in normalized:
        return "external/" + normalized.split(marker, 1)[1]
    return image_path


def assign_split(key: str, train_frac: float, valid_frac: float) -> str:
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_frac:
        return "train"
    if value < train_frac + valid_frac:
        return "valid"
    return "test"


def force_patient_resplit(rows: list[dict[str, Any]], train_frac: float, valid_frac: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        split = assign_split(str(row["patient_id"]), train_frac, valid_frac)
        copied["source_split"] = f"{row['source_dataset']}_{split}"
        out.append(copied)
    return out


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", nargs=2, metavar=("DATASET", "MANIFEST"))
    parser.add_argument("--output", default="results/autonomy_route_combined_manifest.csv")
    parser.add_argument("--summary-output", default="results/autonomy_route_combined_manifest_summary.json")
    parser.add_argument("--force-patient-resplit", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--valid-frac", type=float, default=0.15)
    args = parser.parse_args()

    root = Path.cwd()
    inputs = args.input or DEFAULT_INPUTS
    rows: list[dict[str, Any]] = []
    unavailable_inputs: list[dict[str, str]] = []
    for dataset, raw_path in inputs:
        path = Path(raw_path)
        if not path.exists():
            unavailable_inputs.append({"dataset": dataset, "manifest": raw_path, "reason": "manifest_missing"})
            continue
        rows.extend(read_manifest(dataset, path))

    rows, incomplete_pairs_dropped = drop_incomplete_paired_eyes(rows)

    deduped: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    missing_images: list[str] = []
    for row in rows:
        row["image_path"] = normalize_image_path(str(row["image_path"]))
        key = row["image_path"]
        if key in seen_paths:
            continue
        seen_paths.add(key)
        if not image_exists(root, key):
            missing_images.append(key)
            continue
        deduped.append(row)

    if args.force_patient_resplit:
        deduped = force_patient_resplit(deduped, args.train_frac, args.valid_frac)

    write_manifest(Path(args.output), deduped)
    split_counts = Counter(canonical_split(str(row["source_split"])) or "unknown" for row in deduped)
    patients_by_split: dict[str, set[str]] = {}
    eyes_by_split: dict[str, set[str]] = {}
    for row in deduped:
        split = canonical_split(str(row["source_split"])) or "unknown"
        patients_by_split.setdefault(split, set()).add(str(row["patient_id"]))
        eyes_by_split.setdefault(split, set()).add(str(row["eye_id"]))
    dataset_counts = Counter(str(row["source_dataset"]) for row in deduped)
    grade_counts = Counter(int(row["grade"]) for row in deduped)
    referable_counts = Counter(int(row["referable_dr"]) for row in deduped)
    summary = {
        "status": "AUTONOMY_ROUTE_MANIFEST_READY" if deduped else "NO_GO_NO_ROWS",
        "manifest_output": args.output,
        "inputs": [{"dataset": dataset, "manifest": manifest} for dataset, manifest in inputs],
        "unavailable_inputs": unavailable_inputs,
        "incomplete_paired_eyes_dropped": incomplete_pairs_dropped,
        "n_rows": len(deduped),
        "n_patients": len({row["patient_id"] for row in deduped}),
        "n_eyes": len({row["eye_id"] for row in deduped}),
        "n_missing_images_dropped": len(missing_images),
        "missing_images_sample": missing_images[:20],
        "force_patient_resplit": bool(args.force_patient_resplit),
        "train_frac": args.train_frac,
        "valid_frac": args.valid_frac,
        "split_counts": dict(sorted(split_counts.items())),
        "patient_counts_by_split": {key: len(value) for key, value in sorted(patients_by_split.items())},
        "eye_counts_by_split": {key: len(value) for key, value in sorted(eyes_by_split.items())},
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "grade_counts": dict(sorted(grade_counts.items())),
        "referable_counts": dict(sorted(referable_counts.items())),
        "boundary": (
            "This is a development/training manifest. Certification still requires a sealed "
            "patient-level cohort not used for model selection."
        ),
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
