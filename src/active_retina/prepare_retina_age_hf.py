"""Prepare the public retina-age-analysis HF dataset for quality/reacquisition work.

This dataset is not a DR-grading replacement. Its value for ActiveRetina is the
patient-level split plus quality/anatomy/camera metadata, which can train and
validate the input-quality gate used before autonomous diagnosis.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


REPO_ID = "ramankamran/retina-age-analysis"
SPLIT_FILES = {
    "train": "splits/train.csv",
    "valid": "splits/val.csv",
    "test": "splits/test.csv",
}


def download_metadata(local_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for split, filename in SPLIT_FILES.items():
        out[split] = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename, local_dir=local_dir))
    out["stats"] = Path(
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="splits/split_statistics.json", local_dir=local_dir)
    )
    return out


def image_path(local_dir: Path, image_id: str) -> Path:
    return local_dir / "images" / f"{image_id}.jpg"


def download_image(local_dir: Path, image_id: str) -> Path:
    expected = image_path(local_dir, image_id)
    if expected.exists() and expected.stat().st_size > 0:
        return expected
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"images/{image_id}.jpg",
            local_dir=local_dir,
        )
    )


def read_rows(paths: dict[str, Path], local_dir: Path, download_images: bool, limit_per_split: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, path in paths.items():
        if split == "stats":
            continue
        count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                image_id = str(row["image_id"])
                if download_images:
                    img = download_image(local_dir, image_id)
                else:
                    img = image_path(local_dir, image_id)
                quality = str(row.get("quality", "")).strip() or "unknown"
                dr = int(float(row.get("diabetic_retinopathy", 0) or 0))
                rows.append(
                    {
                        "patient_id": f"retina_age:{row['patient_id']}",
                        "eye_id": f"retina_age:{row['patient_id']}:{row.get('exam_eye', 'unknown')}",
                        "image_id": f"retina_age:{image_id}",
                        "image_path": str(img),
                        "source_split": f"retina_age_{split}",
                        "diabetic_retinopathy_binary": dr,
                        "quality": quality,
                        "is_adequate_quality": 1 if quality.lower() == "adequate" else 0,
                        "camera": row.get("camera", ""),
                        "exam_eye": row.get("exam_eye", ""),
                        "optic_disc": row.get("optic_disc", ""),
                        "vessels": row.get("vessels", ""),
                        "macula": row.get("macula", ""),
                        "hemorrhage": row.get("hemorrhage", ""),
                        "macular_edema": row.get("macular_edema", ""),
                    }
                )
                count += 1
                if limit_per_split is not None and count >= limit_per_split:
                    break
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "patient_id",
        "eye_id",
        "image_id",
        "image_path",
        "source_split",
        "diabetic_retinopathy_binary",
        "quality",
        "is_adequate_quality",
        "camera",
        "exam_eye",
        "optic_disc",
        "vessels",
        "macula",
        "hemorrhage",
        "macular_edema",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default="external/retina_age_analysis_hf")
    parser.add_argument("--manifest-output", default="results/retina_age_quality_manifest.csv")
    parser.add_argument("--summary-output", default="results/retina_age_quality_summary.json")
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    paths = download_metadata(local_dir)
    rows = read_rows(paths, local_dir, args.download_images, args.limit_per_split)
    write_csv(Path(args.manifest_output), rows)

    missing_images = [row["image_path"] for row in rows if not Path(row["image_path"]).exists()]
    summary = {
        "status": "RETINA_AGE_QUALITY_MANIFEST_READY" if rows else "NO_GO_NO_ROWS",
        "repo_id": REPO_ID,
        "manifest_output": args.manifest_output,
        "download_images": bool(args.download_images),
        "limit_per_split": args.limit_per_split,
        "n_rows": len(rows),
        "n_patients": len({row["patient_id"] for row in rows}),
        "split_counts": dict(sorted(Counter(row["source_split"] for row in rows).items())),
        "quality_counts": dict(sorted(Counter(row["quality"] for row in rows).items())),
        "dr_binary_counts": dict(sorted(Counter(int(row["diabetic_retinopathy_binary"]) for row in rows).items())),
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images[:20],
        "boundary": (
            "Use this dataset for input-quality/reacquisition and external metadata robustness. "
            "It is not a five-grade DR certification dataset."
        ),
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
