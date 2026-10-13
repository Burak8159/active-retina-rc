"""Build a leakage-aware manifest for one MMRDR modality."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split


def _find_csv(root: Path, modality: str) -> Path:
    name = {"CFP": "FP.csv", "UWF": "UWF.csv", "OCT": "OCT.csv"}[modality]
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} under {root}; found {len(matches)}")
    return matches[0]


def _image_index(root: Path, modality: str) -> dict[str, Path]:
    """Index one modality once; avoid a full recursive scan for every CSV row."""
    modality_root = next((p for p in root.rglob(f"MMRDR-{modality}") if p.is_dir()), None)
    search_root = modality_root or root
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in search_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        key = path.name.lower()
        if key in index:
            duplicates.add(key)
        else:
            index[key] = path
    for key in duplicates:
        index.pop(key, None)
    return index


def _image_path(image_index: dict[str, Path], raw_name: str, modality: str) -> Path:
    name = Path(raw_name.strip()).name
    if not Path(name).suffix:
        name += ".jpg"
    path = image_index.get(name.lower())
    if path is None:
        raise ValueError(f"Expected exactly one image for {raw_name!r} in {modality}; missing or duplicated")
    return path


def _split_from_name(name: str) -> str:
    prefix = Path(name).stem.lower()[:2]
    if prefix == "tr":
        return "train"
    if prefix == "ts":
        return "test"
    raise ValueError(f"Cannot identify official split from image filename: {name}")


def build_manifest(root: Path, modality: str, validation_fraction: float, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    modality = modality.upper()
    if modality not in {"CFP", "UWF", "OCT"}:
        raise ValueError(f"Unsupported modality: {modality}")
    csv_path = _find_csv(root, modality)
    image_index = _image_index(root, modality)
    project_root = Path.cwd().resolve()
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image", "grade"}.issubset(reader.fieldnames):
            raise ValueError(f"Expected image and grade columns in {csv_path}; got {reader.fieldnames}")
        for raw in reader:
            image_name = Path(raw["image"].strip()).name
            path = _image_path(image_index, image_name, modality)
            grade = int(float(raw["grade"]))
            if grade < 0 or grade > 4:
                raise ValueError(f"Grade outside MMRDR 0-4 range: {grade} ({image_name})")
            split = _split_from_name(image_name)
            image_id = f"mmrdr_{modality.lower()}_{Path(image_name).stem}"
            absolute_path = path.resolve()
            try:
                stored_path = absolute_path.relative_to(project_root).as_posix()
            except ValueError:
                stored_path = str(absolute_path)
            rows.append({
                "patient_id": "",
                "eye_id": "",
                "image_id": image_id,
                "field": modality.lower(),
                "grade": grade,
                "referable_dr": int(grade >= 2),
                "image_path": stored_path,
                "source_split": split,
                "source_dataset": f"mmrdr_{modality.lower()}",
            })

    train = [row for row in rows if row["source_split"] == "train"]
    test = [row for row in rows if row["source_split"] == "test"]
    if not train or not test:
        raise ValueError(f"Both official tr* and ts* rows are required; train={len(train)}, test={len(test)}")
    grades = [int(row["grade"]) for row in train]
    indices = list(range(len(train)))
    try:
        train_indices, valid_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=seed,
            stratify=grades,
        )
    except ValueError as exc:
        raise ValueError(
            "Cannot make the requested stratified development split; inspect rare grade counts "
            "and choose a larger validation fraction or a predeclared unstratified split."
        ) from exc
    for index in valid_indices:
        train[index]["source_split"] = "valid"

    output = train + test
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        subset = [row for row in output if row["source_split"] == split]
        counts[split] = {
            "images": len(subset),
            "positive_grade_ge_2": sum(int(row["referable_dr"]) for row in subset),
            "negative_grade_0_1": sum(int(row["referable_dr"]) == 0 for row in subset),
            "grade_counts": {str(k): v for k, v in sorted(Counter(int(row["grade"]) for row in subset).items())},
        }
    summary = {
        "dataset": "MMRDR",
        "modality": modality,
        "annotation_csv": str(csv_path),
        "official_split": "filename prefix tr/ts preserved",
        "official_test_patient_level": modality in {"UWF", "OCT"},
        "official_test_patient_level_source": "MMRDR dataset paper; CFP is image-level because source patient IDs are unavailable",
        "patient_ids_available_in_exported_table": False,
        "internal_train_valid_split": "stratified image-level split; patient grouping unavailable in exported table",
        "validation_fraction_of_official_train": validation_fraction,
        "seed": seed,
        "split_counts": counts,
        "warning": "The CFP official test split is image-level and cannot support patient-independent confirmation. Prefer UWF for the primary test; do not treat UWF as a paired-view acquisition dataset.",
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Extracted MMRDR dataset root")
    parser.add_argument("--modality", choices=("CFP", "UWF", "OCT"), default="UWF")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--development-manifest-output", type=Path)
    parser.add_argument("--test-manifest-output", type=Path)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    records, summary = build_manifest(args.root, args.modality, args.validation_fraction, args.seed)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if args.development_manifest_output:
        development = [row for row in records if row["source_split"] in {"train", "valid"}]
        args.development_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with args.development_manifest_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(development)
    if args.test_manifest_output:
        test_rows = [row for row in records if row["source_split"] == "test"]
        args.test_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with args.test_manifest_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(test_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
