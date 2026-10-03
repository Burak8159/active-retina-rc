from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from .common import read_yaml_like, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = read_yaml_like(Path(args.config))
    manifest = Path(args.manifest or cfg["dataset"]["manifest_output"])
    output = Path(args.output or cfg["dataset"]["split_output"])
    if not manifest.exists():
        raise SystemExit(f"Missing manifest: {manifest}")

    with manifest.open("r", encoding="utf-8", newline="") as f:
        patients = sorted({row["patient_id"] for row in csv.DictReader(f)})

    fractions = cfg["splits"]["fractions"]
    rng = random.Random(int(cfg["splits"]["seed"]))
    shuffled = patients[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * float(fractions["train"]))
    n_cal = round(n * float(fractions["calibration"]))
    n_val = round(n * float(fractions["validation"]))
    train = sorted(shuffled[:n_train])
    calibration = sorted(shuffled[n_train : n_train + n_cal])
    validation = sorted(shuffled[n_train + n_cal : n_train + n_cal + n_val])
    sealed_test = sorted(shuffled[n_train + n_cal + n_val :])

    write_json(
        output,
        {
            "status": "IMMUTABLE_PATIENT_SPLITS",
            "unit": "patient_id",
            "seed": int(cfg["splits"]["seed"]),
            "source_manifest": str(manifest),
            "train_patients": train,
            "calibration_patients": calibration,
            "validation_patients": validation,
            "sealed_test_patients": sealed_test,
            "counts": {
                "train": len(train),
                "calibration": len(calibration),
                "validation": len(validation),
                "sealed_test": len(sealed_test),
                "total": len(patients),
            },
        },
    )


if __name__ == "__main__":
    main()
