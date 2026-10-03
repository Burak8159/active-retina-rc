"""Prepare the public Hugging Face mirror of DDR for ActiveRetina.

DDR is a large single-image DR grading dataset. It cannot certify the paired
macula/disc acquisition policy, but it can harden the autonomous diagnostic
head against domain shift using immediate public data.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


REPO_ID = "ctmedtech/DDR-dataset"
LABEL_FILES = {
    "train": "DR_grading/train.txt",
    "valid": "DR_grading/valid.txt",
    "test": "DR_grading/test.txt",
}


def read_label_file(
    path: Path,
    split: str,
    limit_per_split: int | None,
    limit_per_grade_per_split: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_grade: dict[int, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, grade = line.split()
            grade_int = int(grade)
            if limit_per_grade_per_split is not None and per_grade[grade_int] >= limit_per_grade_per_split:
                continue
            rows.append({"split": split, "filename": name, "grade": grade_int})
            per_grade[grade_int] += 1
            if limit_per_split is not None and len(rows) >= limit_per_split:
                break
    return rows


def download_labels(local_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for split, repo_file in LABEL_FILES.items():
        paths[split] = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=repo_file,
                local_dir=local_dir,
            )
        )
    return paths


def download_image(local_dir: Path, split: str, filename: str) -> Path:
    repo_file = f"DR_grading/{split}/{filename}"
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=repo_file,
            local_dir=local_dir,
        )
    )


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["patient_id", "eye_id", "image_id", "field", "grade", "referable_dr", "image_path", "source_split"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default="external/DDR_hf")
    parser.add_argument("--manifest-output", default="results/ddr_hf_manifest.csv")
    parser.add_argument("--summary-output", default="results/ddr_hf_summary.json")
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--limit-per-grade-per-split", type=int)
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    label_paths = download_labels(local_dir)
    label_rows: list[dict[str, Any]] = []
    for split, path in label_paths.items():
        label_rows.extend(read_label_file(path, split, args.limit_per_split, args.limit_per_grade_per_split))

    manifest: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in label_rows:
        split = row["split"]
        filename = row["filename"]
        if args.download_images:
            image_path = download_image(local_dir, split, filename)
        else:
            image_path = local_dir / "DR_grading" / split / filename
        if not image_path.exists():
            missing.append(str(image_path))
        image_id = f"ddr_{split}_{Path(filename).stem}"
        grade = int(row["grade"])
        manifest.append(
            {
                "patient_id": image_id,
                "eye_id": image_id,
                "image_id": image_id,
                "field": "single_fundus",
                "grade": grade,
                "referable_dr": 1 if grade >= 2 else 0,
                "image_path": str(image_path),
                "source_split": f"ddr_{split}",
            }
        )

    write_manifest(Path(args.manifest_output), manifest)
    grade_counts = Counter(int(r["grade"]) for r in manifest)
    split_counts = Counter(r["source_split"] for r in manifest)
    summary = {
        "status": "DDR_HF_MANIFEST_READY" if not missing else "DDR_HF_MANIFEST_LABELS_READY_IMAGES_MISSING",
        "repo_id": REPO_ID,
        "local_dir": str(local_dir),
        "download_images": args.download_images,
        "limit_per_split": args.limit_per_split,
        "limit_per_grade_per_split": args.limit_per_grade_per_split,
        "n_rows": len(manifest),
        "n_positive": sum(int(r["referable_dr"]) for r in manifest),
        "grade_counts": dict(sorted(grade_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "missing_images_count": len(missing),
        "missing_images_sample": missing[:20],
        "manifest_output": args.manifest_output,
        "boundary": (
            "DDR is single-image DR grading data. Use it for diagnostic robustness/model hardening, "
            "not paired-view acquisition certification."
        ),
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
