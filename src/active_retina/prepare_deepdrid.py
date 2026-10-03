from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_yaml_like, write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _eye_from_image_id(image_id: str) -> str:
    if "_l" in image_id:
        return "left"
    if "_r" in image_id:
        return "right"
    return "unknown"


def _suffix_from_image_id(image_id: str) -> str:
    return image_id[-1]


def _field_from_suffix(suffix: str, view_mapping: dict[str, str]) -> str:
    return view_mapping.get(f"suffix_{suffix}", f"suffix_{suffix}")


def _grade_for_eye(row: dict[str, str], eye: str) -> int | None:
    key = "left_eye_DR_Level" if eye == "left" else "right_eye_DR_Level"
    raw = row.get(key, "").strip()
    if raw == "":
        return None
    return int(raw)


def _normalize_path(raw: str) -> str:
    return raw.strip().lstrip("\\/").replace("\\", "/")


def _resolved_image_path(local_root: Path | None, row: dict[str, str], source_split: str) -> str:
    normalized = _normalize_path(row["image_path"])
    if local_root is None:
        return normalized
    split_dir = "regular-fundus-training" if source_split == "official_train" else "regular-fundus-validation"
    candidate = local_root / split_dir / "Images" / row["patient_id"] / f"{row['image_id']}.jpg"
    return str(candidate) if candidate.exists() else str(local_root / normalized)


def _load_rows(metadata_root: Path, view_mapping: dict[str, str], local_root: Path | None) -> list[dict[str, Any]]:
    split_specs = [
        ("official_train", metadata_root / "regular_fundus_training" / "regular-fundus-training.csv"),
        ("official_validation", metadata_root / "regular_fundus_validation" / "regular-fundus-validation.csv"),
    ]
    rows: list[dict[str, Any]] = []
    for source_split, path in split_specs:
        for row in _read_csv(path):
            eye = _eye_from_image_id(row["image_id"])
            grade = _grade_for_eye(row, eye)
            if grade is None:
                continue
            suffix = _suffix_from_image_id(row["image_id"])
            rows.append(
                {
                    "patient_id": row["patient_id"],
                    "eye_id": f"{row['patient_id']}_{eye}",
                    "eye": eye,
                    "image_id": row["image_id"],
                    "view_suffix": suffix,
                    "field": _field_from_suffix(suffix, view_mapping),
                    "grade": grade,
                    "referable_dr": 1 if 2 <= grade <= 4 else 0,
                    "patient_grade": int(row["patient_DR_Level"]),
                    "overall_quality": int(row["Overall quality"]),
                    "clarity": int(row["Clarity"]),
                    "field_definition": int(row["Field definition"]),
                    "artifact": int(row["Artifact"]),
                    "image_path": _resolved_image_path(local_root, row, source_split),
                    "source_split": source_split,
                }
            )
    return rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "patient_id",
        "eye_id",
        "eye",
        "image_id",
        "view_suffix",
        "field",
        "grade",
        "referable_dr",
        "patient_grade",
        "overall_quality",
        "clarity",
        "field_definition",
        "artifact",
        "image_path",
        "source_split",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pair_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_eye: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_eye[row["eye_id"]].add(row["view_suffix"])
    counts = Counter(tuple(sorted(v)) for v in by_eye.values())
    complete = sum(1 for v in by_eye.values() if v == {"1", "2"})
    return {
        "n_eyes": len(by_eye),
        "n_complete_suffix_1_2_eye_pairs": complete,
        "suffix_set_counts": {"+".join(k): v for k, v in counts.items()},
        "incomplete_eye_examples": [eye for eye, views in by_eye.items() if views != {"1", "2"}][:20],
    }


def _image_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    existing = [row for row in rows if Path(row["image_path"]).exists()]
    missing = [row for row in rows if not Path(row["image_path"]).exists()]
    return {
        "n_manifest_rows": len(rows),
        "n_existing_images": len(existing),
        "n_missing_images": len(missing),
        "missing_examples": [row["image_path"] for row in missing[:20]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_yaml_like(Path(args.config))
    metadata_root = Path(cfg["dataset"]["metadata_root"])
    local_root = Path(cfg["dataset"]["local_root"]) if cfg["dataset"].get("local_root") else None
    manifest_output = Path(cfg["dataset"]["manifest_output"])
    status_output = Path(cfg["dataset"]["status_output"])

    required = [
        metadata_root / "README.md",
        metadata_root / "LICENSE",
        metadata_root / "regular_fundus_training" / "regular-fundus-training.csv",
        metadata_root / "regular_fundus_training" / "regular-fundus-source-training.csv",
        metadata_root / "regular_fundus_validation" / "regular-fundus-validation.csv",
        metadata_root / "regular_fundus_validation" / "regular-fundus-source-validation.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        write_json(status_output, {"status": "BLOCKED", "reason": "DeepDRiD metadata missing", "missing": missing})
        raise SystemExit(f"DeepDRiD D1 blocked: missing metadata files. See {status_output}")

    view_mapping = cfg["dataset"].get("view_mapping", {})
    rows = _load_rows(metadata_root, view_mapping, local_root)
    _write_manifest(manifest_output, rows)
    grade_counts = Counter(row["grade"] for row in rows)
    patient_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        patient_split[row["patient_id"]].add(row["source_split"])
    mixed = sorted(patient for patient, splits in patient_split.items() if len(splits) > 1)

    status = {
        "status": "SCHEMA_AND_LOCAL_IMAGES_VERIFIED" if _image_inventory(rows)["n_missing_images"] == 0 else "PARTIAL_SCHEMA_VERIFIED_IMAGES_INCOMPLETE",
        "dataset": "DeepDRiD",
        "official_source": cfg["dataset"]["official_source"],
        "license": cfg["dataset"]["license"],
        "metadata_root": str(metadata_root),
        "local_root": str(local_root) if local_root else "",
        "manifest_output": str(manifest_output),
        "n_images_labeled_regular_fundus": len(rows),
        "n_patients": len({row["patient_id"] for row in rows}),
        "n_eyes": len({row["eye_id"] for row in rows}),
        "grade_counts": dict(sorted(grade_counts.items())),
        "referable_counts": dict(sorted(Counter(row["referable_dr"] for row in rows).items())),
        "source_split_counts": dict(sorted(Counter(row["source_split"] for row in rows).items())),
        "patient_leakage_between_official_splits": mixed,
        "pair_status": _pair_status(rows),
        "image_inventory": _image_inventory(rows),
        "quality_counts": {
            "overall_quality": dict(sorted(Counter(row["overall_quality"] for row in rows).items())),
            "field_definition": dict(sorted(Counter(row["field_definition"] for row in rows).items())),
        },
        "field_mapping": view_mapping,
        "field_mapping_status": cfg["dataset"].get("view_mapping_status", "UNKNOWN"),
        "field_mapping_note": cfg["dataset"].get("view_mapping_note", ""),
        "can_train": _image_inventory(rows)["n_missing_images"] == 0,
        "blocking_items": [] if _image_inventory(rows)["n_missing_images"] == 0 else ["Full image inventory is incomplete locally."],
        "risk_items": [
            "Suffix-to-field mapping is visually inferred from sample images; no official suffix mapping statement has been found in public DeepDRiD metadata reviewed so far.",
        ],
    }
    write_json(status_output, status)


if __name__ == "__main__":
    main()
