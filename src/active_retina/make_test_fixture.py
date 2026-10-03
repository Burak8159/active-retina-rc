from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def filter_split(source_csv: Path, source_images: Path, target_csv: Path, target_images: Path) -> int:
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    target_images.mkdir(parents=True, exist_ok=True)
    kept = []
    with source_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            stems = [row[k] for k in ("id1", "id2", "id3", "id4")]
            paths = [source_images / f"{stem}.jpg" for stem in stems]
            if all(p.exists() for p in paths):
                kept.append(row)
                for p in paths:
                    shutil.copy2(p, target_images / p.name)
    with target_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-sample", default="external/MFIDDR_repo/sample")
    parser.add_argument("--out", default="data/MFIDDR_TEST_FIXTURE")
    args = parser.parse_args()
    source = Path(args.repo_sample)
    out = Path(args.out)
    n_train = filter_split(source / "train_fourpic_label.csv", source / "train-examples", out / "train_fourpic_label.csv", out / "train")
    n_test = filter_split(source / "test_fourpic_label.csv", source / "test-examples", out / "test_fourpic_label.csv", out / "test")
    print(f"TEST_FIXTURE created: train_eyes={n_train}, test_eyes={n_test}, output={out}")


if __name__ == "__main__":
    main()
