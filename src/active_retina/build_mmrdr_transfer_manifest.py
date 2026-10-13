"""Combine MMRDR CFP training images with UWF development rows without false pairing."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_manifest(uwf_manifest: Path, cfp_annotations: Path, cfp_image_dir: Path) -> tuple[list[dict[str, str | int]], dict]:
    uwf_rows = read_rows(uwf_manifest)
    if not uwf_rows or any(row["source_split"] not in {"train", "valid"} for row in uwf_rows):
        raise ValueError("UWF input must contain development train/valid rows only.")

    cfp_rows = read_rows(cfp_annotations)
    if not cfp_rows or not {"image", "grade"}.issubset(cfp_rows[0]):
        raise ValueError("CFP annotation CSV must contain image and grade columns.")

    output: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for row in uwf_rows:
        item = dict(row)
        item["source_dataset"] = "mmrdr_uwf"
        output.append(item)

    for row in cfp_rows:
        image_name = Path(row["image"].strip()).name
        if not image_name.lower().startswith("tr"):
            raise ValueError(f"Non-training CFP row found in training annotations: {image_name}")
        if image_name in seen:
            raise ValueError(f"Duplicate CFP training image: {image_name}")
        seen.add(image_name)
        image_path = cfp_image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        grade = int(float(row["grade"]))
        if grade < 0 or grade > 4:
            raise ValueError(f"CFP grade outside 0-4: {image_name} grade={grade}")
        output.append({
            "patient_id": "",
            "eye_id": "",
            "image_id": f"mmrdr_cfp_{Path(image_name).stem}",
            "field": "cfp",
            "grade": grade,
            "referable_dr": int(grade >= 2),
            "image_path": image_path.as_posix(),
            "source_split": "train",
            "source_dataset": "mmrdr_cfp",
        })

    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "valid"):
        subset = [row for row in output if row["source_split"] == split]
        split_counts[split] = {
            "images": len(subset),
            "referable": sum(int(row["referable_dr"]) for row in subset),
            "nonreferable": sum(1 - int(row["referable_dr"]) for row in subset),
        }
    source_counts = Counter(str(row["source_dataset"]) for row in output)
    summary = {
        "status": "DEVELOPMENT_TRANSFER_MANIFEST; OFFICIAL_TEST_NOT_INCLUDED",
        "pairing": "CFP and UWF are independent cohorts; filename IDs are independently assigned and are not paired-eye identifiers.",
        "cfp_role": "official CFP tr* training images only; CFP ts* test images and labels excluded",
        "uwf_role": "existing development train/valid rows; UWF official test rows excluded",
        "patient_independence": "not established for the internal UWF train/valid split; CFP patient IDs unavailable",
        "source_counts": dict(source_counts),
        "split_counts": split_counts,
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uwf-manifest", type=Path, required=True)
    parser.add_argument("--cfp-annotations", type=Path, required=True)
    parser.add_argument("--cfp-image-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = build_manifest(args.uwf_manifest, args.cfp_annotations, args.cfp_image_dir)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
