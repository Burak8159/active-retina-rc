from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from .common import read_yaml_like


def stable_noise(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results/TEST_FIXTURE_sealed_field_predictions.csv")
    args = parser.parse_args()
    cfg = read_yaml_like(Path(args.config))
    manifest_path = Path(cfg["dataset"].get("manifest_output", "results/mfiddr_canonical_manifest.csv"))
    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            grade = int(row["grade"])
            base = 0.15 + 0.18 * grade
            field_shift = {"F1": 0.02, "F2": 0.01, "F3": 0.04, "F4": 0.03}[row["field"]]
            noise = (stable_noise(row["eye_id"], row["field"]) - 0.5) * 0.08
            score = min(0.98, max(0.02, base + field_shift + noise))
            rows.append(
                {
                    "patient_id": row["patient_id"],
                    "eye_id": row["eye_id"],
                    "field": row["field"],
                    "grade": row["grade"],
                    "referable_score": f"{score:.6f}",
                    "source": "TEST_FIXTURE_DETERMINISTIC_NOT_MODEL_OUTPUT",
                }
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "eye_id", "field", "grade", "referable_score", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} TEST_FIXTURE prediction rows to {out}")


if __name__ == "__main__":
    main()

