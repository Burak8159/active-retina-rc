"""Build a patient-disjoint development split from original training partitions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedGroupKFold


ORIGINAL_TRAIN_SPLITS = {"ddr_train", "drtid_training", "official_train", "idrid_train"}


def _group(row: dict[str, str]) -> str:
    patient = row.get("patient_id", "").strip() or row["eye_id"]
    return f"{row['source_dataset']}::{patient}"


def build_split(
    rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    *,
    n_splits: int = 5,
    fold: int = 0,
    random_state: int = 20261013,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    reference_eyes = {row["eye_id"] for row in reference_rows}
    if not reference_eyes:
        raise ValueError("Reference replay is empty")
    reference_groups = {_group(row) for row in rows if row["eye_id"] in reference_eyes}
    if not reference_groups:
        raise ValueError("Reference eyes do not map to patient groups in the source manifest")

    eligible = [
        row for row in rows
        if row.get("original_source_split") in ORIGINAL_TRAIN_SPLITS
        and row["eye_id"] not in reference_eyes
        and _group(row) not in reference_groups
    ]
    if not eligible:
        raise ValueError("No eligible original-training rows remain after reference exclusion")
    groups = [_group(row) for row in eligible]
    strata = [
        f"{row['source_dataset']}::" + ("u" if int(row["grade"]) == 5 else "p" if int(row["grade"]) >= 2 else "n")
        for row in eligible
    ]
    if fold < 0 or fold >= n_splits:
        raise ValueError("fold must be in [0, n_splits)")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(splitter.split(eligible, strata, groups))
    train_indices, valid_indices = splits[fold]
    valid_index_set = set(valid_indices)

    output: list[dict[str, str]] = []
    train_groups = {groups[index] for index in train_indices}
    valid_groups = {groups[index] for index in valid_indices}
    if train_groups & valid_groups:
        raise RuntimeError("Patient leakage between generated train and validation splits")
    if (train_groups | valid_groups) & reference_groups:
        raise RuntimeError("Reference-patient leakage into generated split")
    for index, row in enumerate(eligible):
        output.append({**row, "source_split": "valid" if index in valid_index_set else "train"})

    summary = {
        "status": "PATIENT_DISJOINT_DEVELOPMENT_SPLIT_NOT_CERTIFICATION",
        "original_train_partitions_only": sorted(ORIGINAL_TRAIN_SPLITS),
        "reference_eye_count": len(reference_eyes),
        "reference_patient_group_count": len(reference_groups),
        "excluded_reference_row_count": sum(row["eye_id"] in reference_eyes for row in rows),
        "eligible_rows": len(eligible),
        "eligible_eye_count": len({row["eye_id"] for row in eligible}),
        "eligible_patient_count": len({ _group(row) for row in eligible }),
        "train_rows": len(train_indices),
        "valid_rows": len(valid_indices),
        "train_patients": len(train_groups),
        "valid_patients": len(valid_groups),
        "train_valid_patient_overlap": len(train_groups & valid_groups),
        "reference_patient_overlap": len((train_groups | valid_groups) & reference_groups),
        "source_rows": dict(Counter(row["source_dataset"] for row in eligible)),
        "valid_source_rows": dict(Counter(eligible[index]["source_dataset"] for index in valid_indices)),
        "valid_grade_rows": dict(Counter(eligible[index]["grade"] for index in valid_indices)),
        "seed": random_state,
        "fold": fold,
        "n_splits": n_splits,
    }
    if summary["train_valid_patient_overlap"] or summary["reference_patient_overlap"]:
        raise RuntimeError("Generated split failed its overlap audit")
    return output, summary


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    manifest = _read(args.manifest)
    reference = _read(args.reference)
    output, summary = build_split(
        manifest, reference, n_splits=args.n_splits, fold=args.fold, random_state=args.seed
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(output)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
