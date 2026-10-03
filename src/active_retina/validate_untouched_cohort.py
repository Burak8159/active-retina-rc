"""Validate that a cohort is eligible for untouched certification.

This is a guardrail, not a statistics tool. It fails closed unless the cohort
is explicitly marked untouched, has enough positive patients, has complete
paired views, and does not overlap with known development identifiers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


KNOWN_DEVELOPMENT_MANIFESTS = [
    Path("results/deepdrid_manifest.csv"),
    Path("results/deepdrid_online_certification_manifest.csv"),
    Path("results/deepdrid_swapped_mapping_manifest.csv"),
    Path("results/TEST_FIXTURE_mfiddr_canonical_manifest.csv"),
]

REQUIRED_COLUMNS = {
    "patient_id",
    "eye_id",
    "image_id",
    "field",
    "referable_dr",
    "image_path",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def known_ids() -> dict[str, set[str]]:
    ids = {"patient_ids": set(), "eye_ids": set(), "image_ids": set()}
    for path in KNOWN_DEVELOPMENT_MANIFESTS:
        if not path.exists():
            continue
        for row in read_csv(path):
            ids["patient_ids"].add(str(row.get("patient_id", "")))
            ids["eye_ids"].add(str(row.get("eye_id", "")))
            ids["image_ids"].add(str(row.get("image_id", "")))
    return ids


def load_attestation(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate(manifest: Path, attestation: dict[str, Any]) -> dict[str, Any]:
    rows = read_csv(manifest)
    columns = set(rows[0]) if rows else set()
    missing_columns = sorted(REQUIRED_COLUMNS - columns)
    by_eye: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_images: list[str] = []
    for row in rows:
        by_eye[row["eye_id"]].append(row)
        by_patient[row["patient_id"]].append(row)
        if not Path(row["image_path"]).exists():
            missing_images.append(row["image_path"])

    complete_eyes = {
        eye_id: eye_rows
        for eye_id, eye_rows in by_eye.items()
        if {"macula", "optic_disc"}.issubset({r["field"] for r in eye_rows})
    }
    positive_patients = {
        patient_id
        for patient_id, patient_rows in by_patient.items()
        if any(int(r["referable_dr"]) == 1 for r in patient_rows)
    }
    known = known_ids()
    patient_overlap = sorted(set(by_patient) & known["patient_ids"])[:50]
    eye_overlap = sorted(set(by_eye) & known["eye_ids"])[:50]
    image_overlap = sorted({r["image_id"] for r in rows} & known["image_ids"])[:50]

    attested_untouched = attestation.get("untouched_for_active_retina_certification") is True
    attested_no_training = attestation.get("not_used_for_training_or_threshold_selection") is True
    attested_no_prior_inspection = attestation.get("not_inspected_before_freeze") is True
    enough_positives = len(positive_patients) >= 59
    preferred_positives = len(positive_patients) >= 80
    no_overlap = not patient_overlap and not eye_overlap and not image_overlap
    complete = len(complete_eyes) == len(by_eye)
    images_present = not missing_images
    schema_ok = not missing_columns

    eligible = all(
        [
            rows,
            schema_ok,
            complete,
            images_present,
            enough_positives,
            no_overlap,
            attested_untouched,
            attested_no_training,
            attested_no_prior_inspection,
        ]
    )
    return {
        "status": "UNTOUCHED_CERTIFICATION_ELIGIBLE" if eligible else "NOT_ELIGIBLE_FOR_UNTOUCHED_CERTIFICATION",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "n_rows": len(rows),
        "n_patients": len(by_patient),
        "n_eyes": len(by_eye),
        "n_complete_paired_eyes": len(complete_eyes),
        "positive_patients": len(positive_patients),
        "minimum_59_positive_patients_met": enough_positives,
        "preferred_80_positive_patients_met": preferred_positives,
        "schema_missing_columns": missing_columns,
        "missing_image_count": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "overlap_with_known_development": {
            "patient_ids": patient_overlap,
            "eye_ids": eye_overlap,
            "image_ids": image_overlap,
        },
        "attestation": {
            "untouched_for_active_retina_certification": attested_untouched,
            "not_used_for_training_or_threshold_selection": attested_no_training,
            "not_inspected_before_freeze": attested_no_prior_inspection,
            "source": attestation.get("source", ""),
            "collector_or_owner": attestation.get("collector_or_owner", ""),
            "attestation_date": attestation.get("attestation_date", ""),
        },
        "blocking_items": [
            *("manifest has no rows" for _ in [] if not rows),
            *([f"missing schema columns: {missing_columns}"] if missing_columns else []),
            *([f"incomplete paired eyes: {len(by_eye) - len(complete_eyes)}"] if not complete else []),
            *([f"missing image files: {len(missing_images)}"] if missing_images else []),
            *([f"positive patients below 59: {len(positive_patients)}"] if not enough_positives else []),
            *(["overlaps with known development data"] if not no_overlap else []),
            *(["attestation does not mark cohort untouched"] if not attested_untouched else []),
            *(["attestation does not confirm no training or threshold selection use"] if not attested_no_training else []),
            *(["attestation does not confirm no prior inspection before freeze"] if not attested_no_prior_inspection else []),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--attestation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    attestation = load_attestation(Path(args.attestation) if args.attestation else None)
    result = validate(Path(args.manifest), attestation)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
