"""Extract only IDRiD Disease-Grading training images and ICDR grades."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


def normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def build_manifest(
    archive: Path,
    output_root: Path,
    manifest_path: Path,
    summary_path: Path,
    heldout_manifest: Path | None = None,
    expected_images: int = 413,
) -> dict:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output_root.exists() or manifest_path.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite an IDRiD output")

    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if not name.endswith("/")]
        label_candidates = [name for name in members if "traininglabels" in normalized(Path(name).name)]
        if len(label_candidates) != 1:
            raise ValueError(f"Expected one explicit training-label CSV, found {label_candidates}")
        with bundle.open(label_candidates[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            labels = list(csv.DictReader(text))
        if not labels:
            raise ValueError("IDRiD training label CSV is empty")
        columns = {normalized(name): name for name in labels[0]}
        image_column = next((columns[key] for key in ("imagename", "image", "imageid") if key in columns), None)
        grade_column = next((columns[key] for key in ("retinopathygrade", "drgrade", "grade") if key in columns), None)
        if image_column is None or grade_column is None:
            raise ValueError(f"Unrecognized IDRiD label columns: {list(labels[0])}")

        images_by_name: dict[str, str] = {}
        for name in members:
            path = Path(name)
            if path.suffix.lower() in {".jpg", ".jpeg"} and "training" in normalized(name):
                key = normalized(path.stem)
                if key in images_by_name:
                    raise ValueError(f"Ambiguous image filename in archive: {path.stem}")
                images_by_name[key] = name

        heldout_hashes: set[str] = set()
        if heldout_manifest is not None:
            with heldout_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
                for heldout in csv.DictReader(stream):
                    if str(heldout["source_split"]).lower().rsplit("_", 1)[-1] not in {"valid", "test"}:
                        continue
                    raw_path = heldout["image_path"].replace("\\", "/")
                    image_path = Path(raw_path)
                    if not image_path.is_file() and "/external/" in raw_path:
                        image_path = Path("external") / raw_path.split("/external/", 1)[1]
                    if not image_path.is_file():
                        raise FileNotFoundError(f"Held-out image missing during overlap audit: {heldout['image_path']}")
                    heldout_hashes.add(hashlib.sha256(image_path.read_bytes()).hexdigest())

        if len(labels) != expected_images:
            raise ValueError(f"Expected {expected_images} labeled training rows, found {len(labels)}")
        seen_ids = set()
        candidates: dict[str, list[tuple[str, int, bytes]]] = {}
        for row in labels:
            image_name = Path(row[image_column].strip()).stem
            key = normalized(image_name)
            if not key or key in seen_ids or image_name in {".", ".."} or "/" in image_name or "\\" in image_name:
                raise ValueError(f"Blank, unsafe, or duplicate IDRiD training image label: {image_name!r}")
            seen_ids.add(key)
            if key not in images_by_name:
                raise ValueError(f"Training image missing from archive: {image_name}")
            grade = int(row[grade_column])
            if grade not in range(5):
                raise ValueError(f"Expected ICDR grade 0-4 for {image_name}, got {grade}")
            image_bytes = bundle.read(images_by_name[key])
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            candidates.setdefault(image_hash, []).append((image_name, grade, image_bytes))

        entries = []
        grades: Counter[int] = Counter()
        overlap_excluded = 0
        exact_duplicate_rows_excluded = 0
        conflicting_duplicate_rows_excluded = 0
        duplicate_groups = []
        retained = []
        for image_hash, group in candidates.items():
            if len(group) > 1:
                if len({item[1] for item in group}) > 1:
                    conflicting_duplicate_rows_excluded += len(group)
                    duplicate_groups.append({
                        "disposition": "excluded_all_conflicting_labels",
                        "images_and_grades": [{"image": item[0], "grade": item[1]} for item in group],
                    })
                    continue
                exact_duplicate_rows_excluded += len(group) - 1
                duplicate_groups.append({
                    "disposition": "kept_first_same_grade_copy",
                    "images_and_grades": [{"image": item[0], "grade": item[1]} for item in group],
                })
                group = group[:1]
            image_name, grade, image_bytes = group[0]
            if image_hash in heldout_hashes:
                overlap_excluded += 1
                continue
            with Image.open(io.BytesIO(image_bytes)) as source:
                source.verify()
            relative_path = Path(f"{image_name}.jpg")
            uid = f"idrid_train:{image_name}"
            entries.append({
                "patient_id": uid,
                "eye_id": uid,
                "image_id": uid,
                "field": "single_fundus",
                "grade": grade,
                "referable_dr": int(grade >= 2),
                "image_path": str((output_root / relative_path).as_posix()),
                "source_split": "idrid_train",
                "source_dataset": "idrid_train_only",
            })
            grades[grade] += 1
            retained.append((relative_path, image_bytes))
        if not entries:
            raise ValueError("No unique, non-overlapping IDRiD training images remain")

        output_root.mkdir(parents=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        for relative_path, image_bytes in retained:
            (output_root / relative_path).write_bytes(image_bytes)

    fields = list(entries[0])
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)
    summary = {
        "status": "IDRID_TRAIN_ONLY_EXTRACTED",
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "label_member": label_candidates[0],
        "n_training_images": len(entries),
        "n_exact_duplicate_rows_excluded": exact_duplicate_rows_excluded,
        "n_conflicting_duplicate_rows_excluded": conflicting_duplicate_rows_excluded,
        "exact_duplicate_groups": duplicate_groups,
        "n_exact_heldout_overlaps_excluded": overlap_excluded,
        "heldout_manifest": str(heldout_manifest) if heldout_manifest else None,
        "grade_counts": dict(sorted(grades.items())),
        "referable_grade_definition": "ICDR grade >= 2",
        "used_partition": "explicit training-label CSV and training image directory only",
        "test_partition_used": False,
        "patient_identity_boundary": "IDRiD grading data provides image-level labels but no patient/eye IDs; it is appended to training only and provides no patient-level validation or certification evidence.",
        "source_license_boundary": "Official IDRiD page links to IEEE DataPort; Zenodo mirror declares CC BY 4.0 while retaining the original license. Preserve attribution and verify original terms before redistribution.",
        "manifest": str(manifest_path),
        "images": str(output_root),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--heldout-manifest")
    args = parser.parse_args()
    print(json.dumps(build_manifest(
        Path(args.archive), Path(args.output_root), Path(args.manifest_output), Path(args.summary_output),
        Path(args.heldout_manifest) if args.heldout_manifest else None,
    ), indent=2))


if __name__ == "__main__":
    main()
