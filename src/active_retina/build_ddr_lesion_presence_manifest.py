"""Join DDR lesion masks to existing DDR training rows only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


LESIONS = ("EX", "HE", "MA", "SE")


def read_grades(path: Path) -> dict[str, int]:
    grades: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"Expected filename and grade at {path}:{line_number}")
            filename, raw_grade = parts
            grade = int(raw_grade)
            if filename in grades and grades[filename] != grade:
                raise ValueError(f"Conflicting DDR grades for {filename}")
            grades[filename] = grade
    return grades


def build_rows(
    manifest: Path, dataset_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    grades = read_grades(dataset_root / "DR_grading" / "train.txt")
    segmentation = dataset_root / "lesion_segmentation"
    image_dir = segmentation / "images" / "train"
    annotation_root = segmentation / "annotations" / "train"
    with manifest.open(newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    train_rows = [
        row for row in manifest_rows
        if row.get("source_dataset") == "ddr_hf_specificity300"
        and row.get("source_split") == "ddr_hf_specificity300_train"
    ]
    heldout_names = {
        Path(row["image_path"]).name for row in manifest_rows
        if row.get("source_dataset") == "ddr_hf_specificity300"
        and row.get("source_split") != "ddr_hf_specificity300_train"
    }
    train_by_name: dict[str, dict[str, str]] = {}
    for row in train_rows:
        filename = Path(row["image_path"]).name
        if filename in train_by_name:
            raise ValueError(f"Duplicate DDR training filename in manifest: {filename}")
        train_by_name[filename] = row
    annotation_images = sorted(path.name for path in image_dir.glob("*.jpg"))
    if not annotation_images:
        raise ValueError(f"No train images found under {image_dir}")
    heldout_annotation_overlap = sorted(heldout_names.intersection(annotation_images))
    missing_official_train_grade = sorted(name for name in annotation_images if name not in grades)
    eligible_annotation_images = [
        name for name in annotation_images if name not in heldout_names and name in grades
    ]

    expanded_manifest = list(manifest_rows)
    added_images = 0
    for filename in eligible_annotation_images:
        if filename not in grades:
            raise ValueError(f"No official DDR train grade for lesion image {filename}")
        if filename not in train_by_name:
            grade = grades[filename]
            image_id = f"ddr_hf_specificity300:ddr_train_{Path(filename).stem}"
            added = {
                "patient_id": image_id,
                "eye_id": image_id,
                "image_id": image_id,
                "field": "single_fundus",
                "grade": str(grade),
                "referable_dr": str(int(2 <= grade <= 4)),
                "image_path": f"external/DDR_lesion_supervision/lesion_segmentation/images/train/{filename}",
                "source_split": "ddr_hf_specificity300_train",
                "original_source_split": "ddr_train",
                "source_dataset": "ddr_hf_specificity300",
            }
            expanded_manifest.append(added)
            train_by_name[filename] = added
            added_images += 1

    with manifest.open(newline="", encoding="utf-8") as stream:
        fieldnames = list(csv.DictReader(stream).fieldnames or [])
    if not fieldnames:
        raise ValueError(f"Empty manifest: {manifest}")
    output: list[dict[str, Any]] = []
    missing_mask: list[str] = []
    grade_mismatch: list[dict[str, Any]] = []
    dimension_mismatch: list[str] = []
    for filename in eligible_annotation_images:
        row = train_by_name[filename]
        image_path = image_dir / filename
        grade = int(row["grade"])
        if grade != grades[filename]:
            grade_mismatch.append({"filename": filename, "manifest_grade": grade, "official_train_grade": grades[filename]})
            continue
        masks: dict[str, int] = {}
        image_size: tuple[int, int] | None = None
        with Image.open(image_path) as image:
            image_size = image.size
        for lesion in LESIONS:
            mask_path = annotation_root / lesion / Path(filename).with_suffix(".tif")
            if not mask_path.is_file():
                missing_mask.append(f"{lesion}/{mask_path.name}")
                break
            with Image.open(mask_path) as mask:
                if mask.size != image_size:
                    dimension_mismatch.append(f"{lesion}/{mask_path.name}")
                    break
                masks[lesion] = int(np.asarray(mask).max() > 0)
        if len(masks) != len(LESIONS):
            continue
        output.append({
            "image_basename": filename,
            "grade": grade,
            "ex": masks["EX"],
            "he": masks["HE"],
            "ma": masks["MA"],
            "se": masks["SE"],
        })

    if grade_mismatch or dimension_mismatch:
        raise ValueError(json.dumps({"grade_mismatch": grade_mismatch[:10], "dimension_mismatch": dimension_mismatch[:10]}))
    summary = {
        "status": "AUDIT_COMPLETE_TRAIN_SPLIT_ONLY",
        "manifest_train_ddr_rows_before_expansion": len(train_rows),
        "official_train_segmentation_images": len(annotation_images),
        "segmentation_images_without_official_train_grade_excluded": len(missing_official_train_grade),
        "heldout_annotation_overlap_excluded": len(heldout_annotation_overlap),
        "new_train_images_added_from_official_split": added_images,
        "matched_annotated_images": len(output),
        "annotated_grade_counts": {
            str(grade): sum(int(row["grade"]) == grade for row in output) for grade in range(6)
        },
        "lesion_presence_counts": {
            key: sum(int(row[key.lower()]) for row in output) for key in LESIONS
        },
        "unannotated_rows_not_labeled_absent": True,
        "missing_mask_files": missing_mask,
        "grade_mismatch_count": len(grade_mismatch),
        "dimension_mismatch_count": len(dimension_mismatch),
        "annotation_split": "train only; validation and test annotations excluded",
        "patient_id_note": "Newly appended image IDs are image-level surrogates because DDR patient identifiers are not supplied here.",
    }
    return output, expanded_manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--expanded-manifest-output", required=True, type=Path)
    args = parser.parse_args()

    rows, expanded_manifest, summary = build_rows(args.manifest, args.dataset_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image_basename", "grade", "ex", "he", "ma", "se"))
        writer.writeheader()
        writer.writerows(rows)
    args.expanded_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with args.expanded_manifest_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(expanded_manifest[0]))
        writer.writeheader()
        writer.writerows(expanded_manifest)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
