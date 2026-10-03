"""Audit public mBRSET metadata for suffix-pair certification feasibility."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_csv(Path(args.metadata))
    patients: dict[str, list[dict[str, str]]] = defaultdict(list)
    valid_patients: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        patients[row["patient"]].append(row)
        if row.get("final_icdr", "") != "" and row.get("final_quality", "").strip().lower() == "yes":
            valid_patients[row["patient"]].append(row)

    candidate_pairs: list[tuple[str, str, int]] = []
    for patient_id, patient_rows in valid_patients.items():
        by_file = {row["file"]: row for row in patient_rows}
        for a, b, laterality in [
            (f"{patient_id}.1.jpg", f"{patient_id}.2.jpg", "right"),
            (f"{patient_id}.3.jpg", f"{patient_id}.4.jpg", "left"),
        ]:
            if a in by_file and b in by_file:
                grade = max(int(float(by_file[a]["final_icdr"])), int(float(by_file[b]["final_icdr"])))
                candidate_pairs.append((patient_id, laterality, grade))

    positive_patients = {patient_id for patient_id, _, grade in candidate_pairs if grade >= 2}
    summary: dict[str, Any] = {
        "status": "MBRSET_PUBLIC_METADATA_SUFFIX_PAIR_AUDIT_COMPLETE",
        "metadata": args.metadata,
        "rows": len(rows),
        "patients": len(patients),
        "valid_quality_labeled_rows": sum(len(rs) for rs in valid_patients.values()),
        "valid_quality_labeled_patients": len(valid_patients),
        "candidate_complete_suffix_pairs": len(candidate_pairs),
        "candidate_paired_patients": len({patient_id for patient_id, _, _ in candidate_pairs}),
        "candidate_positive_paired_patients": len(positive_patients),
        "minimum_59_positive_patients_met_before_image_audit": len(positive_patients) >= 59,
        "preferred_80_positive_patients_met_before_image_audit": len(positive_patients) >= 80,
        "grade_counts_candidate_pairs": dict(sorted(Counter(grade for _, _, grade in candidate_pairs).items())),
        "suffix_pattern_note": (
            "Assumes same-eye candidate pairs are patient.1/patient.2 right and "
            "patient.3/patient.4 left; anatomy still requires visual or official audit."
        ),
        "next_step": (
            "Download credentialed images from PhysioNet, audit suffix-to-anatomy mapping, "
            "then run run_untouched_certification_package.py with the audited suffix map."
        ),
    }
    Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
