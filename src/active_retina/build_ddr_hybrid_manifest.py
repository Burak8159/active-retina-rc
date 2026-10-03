"""Build a DDR manifest from an expanded train folder plus fixed validation/test manifests."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_existing_manifest(path: Path, splits: set[str]) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["source_split"] in splits:
                rows.append(row)
    return rows


def read_train_labels(label_path: Path, image_dir: Path) -> list[dict[str, Any]]:
    rows = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filename, grade = line.split()
            path = image_dir / filename
            if not path.exists():
                continue
            grade_int = int(grade)
            image_id = f"ddr_train_{Path(filename).stem}"
            rows.append(
                {
                    "patient_id": image_id,
                    "eye_id": image_id,
                    "image_id": image_id,
                    "field": "single_fundus",
                    "grade": grade_int,
                    "referable_dr": 1 if grade_int >= 2 else 0,
                    "image_path": str(path),
                    "source_split": "ddr_train",
                }
            )
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["patient_id", "eye_id", "image_id", "field", "grade", "referable_dr", "image_path", "source_split"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-train-labels", required=True)
    parser.add_argument("--expanded-train-dir", required=True)
    parser.add_argument("--fixed-manifest", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    train_rows = read_train_labels(Path(args.expanded_train_labels), Path(args.expanded_train_dir))
    heldout_rows = read_existing_manifest(Path(args.fixed_manifest), {"ddr_valid", "ddr_test"})
    rows = train_rows + heldout_rows
    write_manifest(Path(args.manifest_output), rows)
    summary = {
        "status": "DDR_HYBRID_MANIFEST_READY",
        "n_rows": len(rows),
        "n_train": len(train_rows),
        "n_valid_test": len(heldout_rows),
        "n_positive": sum(int(r["referable_dr"]) for r in rows),
        "split_counts": dict(sorted(Counter(r["source_split"] for r in rows).items())),
        "grade_counts": dict(sorted(Counter(int(r["grade"]) for r in rows).items())),
        "boundary": "Expanded train uses whatever DDR images have already downloaded; validation and test are fixed from the 30-per-grade subset.",
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
