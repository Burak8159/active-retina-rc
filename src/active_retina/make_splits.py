from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from .common import read_yaml_like, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = read_yaml_like(Path(args.config))
    manifest_path = Path(cfg["dataset"].get("manifest_output", "results/mfiddr_canonical_manifest.csv"))
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}; run prepare_mfiddr first.")

    by_patient_split: dict[str, set[str]] = defaultdict(set)
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_patient_split[row["patient_id"]].add(row["source_split"])

    train_patients = sorted(p for p, splits in by_patient_split.items() if splits == {"official_train"})
    test_patients = sorted(p for p, splits in by_patient_split.items() if splits == {"official_test"})
    mixed = sorted(p for p, splits in by_patient_split.items() if len(splits) > 1)
    if mixed:
        raise SystemExit(f"Patient leakage across official train/test splits: {mixed[:10]}")

    rng = random.Random(cfg["splits"]["seed"])
    shuffled = train_patients[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * float(cfg["splits"]["validation_fraction_within_train"])))
    val_patients = sorted(shuffled[:n_val])
    fit_patients = sorted(shuffled[n_val:])

    payload = {
        "status": "IMMUTABLE_SPLIT_PROTOCOL",
        "unit": "patient_id",
        "seed": cfg["splits"]["seed"],
        "fit_patients": fit_patients,
        "validation_patients": val_patients,
        "sealed_test_patients": test_patients,
        "counts": {
            "fit": len(fit_patients),
            "validation": len(val_patients),
            "sealed_test": len(test_patients),
        },
    }
    write_json(Path(cfg["splits"]["immutable_output"]), payload)


if __name__ == "__main__":
    main()
