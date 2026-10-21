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
    Path("results/ddr_hf_specificity300_manifest.csv"),
    Path("results/ddr_hf_specificity100_manifest.csv"),
    Path("results/ddr_full_train_manifest_20261011.csv"),
    Path("results/ddr_hybrid_expanded_train_manifest.csv"),
    Path("results/deepdrid_manifest.csv"),
    Path("results/deepdrid_online_certification_manifest.csv"),
    Path("results/deepdrid_swapped_mapping_manifest.csv"),
    Path("results/drtid_untouched_certification/certification_manifest.csv"),
    Path("results/drtid_recovery_candidate_inspected_test_run/manifest.csv"),
    Path("results/idrid_training_manifest_20261011_final.csv"),
    Path("results/idrid_training_manifest_20261011_v2.csv"),
    Path("results/mmrdr_uwf_manifest_20261013.csv"),
    Path("results/combined_mmrdr_qualitygate_expanded_manifest_20261013.csv"),
    Path("results/combined_mmrdr_finetune_dev_20261013/manifest.csv"),
    Path("results/autonomy_route_three_dataset_manifest_20261011.csv"),
    Path("results/autonomy_route_combined_manifest_resplit.csv"),
    Path("results/TEST_FIXTURE_mfiddr_canonical_manifest.csv"),
]

REQUIRED_COLUMNS = {
    "patient_id",
    "eye_id",
    "image_id",
    "field",
    "grade",
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


def canonical_namespace(value: str) -> str:
    normalized = value.casefold().replace("-", "_")
    for dataset in ("deepdrid", "drtid", "mmrdr", "mbrset", "idrid", "ddr", "mfiddr"):
        if normalized.startswith(dataset):
            return dataset
    return normalized


def dataset_namespace(row: dict[str, str], manifest_path: Path, fallback: str = "") -> str:
    explicit = str(row.get("source_dataset", "")).strip()
    if explicit:
        return canonical_namespace(explicit)
    name = str(manifest_path).casefold()
    for dataset in ("deepdrid", "drtid", "mmrdr", "mbrset", "idrid", "ddr", "mfiddr"):
        if dataset in name:
            return "mfiddr_fixture" if dataset == "mfiddr" and "fixture" in name else dataset
    split = str(row.get("source_split", "")).strip()
    if split and split != "untouched_external_certification":
        return canonical_namespace(split)
    if fallback:
        return canonical_namespace(fallback)
    return ""


def known_records() -> tuple[dict[str, set[tuple[str, str]]], set[str]]:
    ids: dict[str, set[tuple[str, str]]] = {
        "patient_ids": set(), "eye_ids": set(), "image_ids": set()
    }
    image_hashes: set[str] = set()
    hashed_paths: set[str] = set()
    for path in KNOWN_DEVELOPMENT_MANIFESTS:
        if not path.exists():
            continue
        for row in read_csv(path):
            namespace = dataset_namespace(row, path)
            if namespace:
                for key in ids:
                    value = str(row.get(key[:-1], "")).strip()
                    if value:
                        ids[key].add((namespace, value))
            image_path = Path(str(row.get("image_path", "")))
            if image_path.is_file():
                resolved = str(image_path.resolve())
                if resolved not in hashed_paths:
                    hashed_paths.add(resolved)
                    image_hashes.add(sha256_file(image_path))
    return ids, image_hashes


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
    manifest_issues: list[str] = []
    seen_image_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        for column in ("patient_id", "eye_id", "image_id", "field", "image_path"):
            if not str(row.get(column, "")).strip():
                manifest_issues.append(f"row {row_number}: blank {column}")
        if str(row.get("field", "")) not in {"macula", "optic_disc"}:
            manifest_issues.append(f"row {row_number}: field must be macula or optic_disc")
        image_id = str(row.get("image_id", "")).strip()
        if image_id and image_id in seen_image_ids:
            manifest_issues.append(f"row {row_number}: duplicate image_id {image_id}")
        seen_image_ids.add(image_id)
        try:
            grade = int(row.get("grade", ""))
            if grade < 0 or grade > 5:
                raise ValueError
            referable = int(row.get("referable_dr", ""))
            if referable not in (0, 1):
                raise ValueError
        except (TypeError, ValueError):
            manifest_issues.append(f"row {row_number}: invalid grade/referable_dr labels")
        patient_id = str(row.get("patient_id", "")).strip()
        eye_id = str(row.get("eye_id", "")).strip()
        if patient_id and eye_id:
            by_eye[eye_id].append(row)
            by_patient[patient_id].append(row)
        image_path = str(row.get("image_path", "")).strip()
        if image_path and not Path(image_path).is_file():
            missing_images.append(image_path)

    complete_eyes = {
        eye_id: eye_rows
        for eye_id, eye_rows in by_eye.items()
        if sorted(r.get("field", "") for r in eye_rows) == ["macula", "optic_disc"]
    }
    for eye_id, eye_rows in by_eye.items():
        if eye_id not in complete_eyes:
            manifest_issues.append(f"eye {eye_id}: expected exactly one macula and one optic_disc row")
        if len({r.get("patient_id", "") for r in eye_rows}) != 1:
            manifest_issues.append(f"eye {eye_id}: maps to multiple patient IDs")
        if len({(r.get("grade", ""), r.get("referable_dr", "")) for r in eye_rows}) != 1:
            manifest_issues.append(f"eye {eye_id}: paired-view labels disagree")
    positive_patients = set()
    for patient_id, patient_rows in by_patient.items():
        try:
            if any(int(r["referable_dr"]) == 1 for r in patient_rows):
                positive_patients.add(patient_id)
        except (KeyError, TypeError, ValueError):
            continue
    known, known_image_hashes = known_records()
    attested_source = str(attestation.get("source", "")).strip()
    namespaces = {dataset_namespace(row, manifest, attested_source) for row in rows}
    namespaces.discard("")
    patient_overlap = sorted(
        {(ns, pid) for ns in namespaces for pid in by_patient} & known["patient_ids"]
    )[:50]
    eye_overlap = sorted(
        {(ns, eid) for ns in namespaces for eid in by_eye} & known["eye_ids"]
    )[:50]
    image_overlap = sorted(
        {
            (dataset_namespace(row, manifest, attested_source), str(row.get("image_id", "")).strip())
            for row in rows
            if dataset_namespace(row, manifest, attested_source) and str(row.get("image_id", "")).strip()
        } & known["image_ids"]
    )[:50]
    candidate_image_hashes = {
        sha256_file(Path(row["image_path"]))
        for row in rows
        if Path(row["image_path"]).is_file()
    }
    duplicate_image_hash_count = len(candidate_image_hashes & known_image_hashes)

    attested_untouched = attestation.get("untouched_for_active_retina_certification") is True
    attested_no_training = attestation.get("not_used_for_training_or_threshold_selection") is True
    attested_no_prior_inspection = attestation.get("not_inspected_before_freeze") is True
    enough_positives = len(positive_patients) >= 59
    preferred_positives = len(positive_patients) >= 80
    no_overlap = not patient_overlap and not eye_overlap and not image_overlap and duplicate_image_hash_count == 0
    complete = len(complete_eyes) == len(by_eye)
    images_present = not missing_images
    schema_ok = not missing_columns and not manifest_issues

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
        "manifest_issue_count": len(manifest_issues),
        "manifest_issue_examples": manifest_issues[:50],
        "missing_image_count": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "overlap_with_known_development": {
            "patient_ids": patient_overlap,
            "eye_ids": eye_overlap,
            "image_ids": image_overlap,
            "exact_image_sha256_count": duplicate_image_hash_count,
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
            *([f"manifest integrity issues: {len(manifest_issues)}"] if manifest_issues else []),
            *([f"incomplete paired eyes: {len(by_eye) - len(complete_eyes)}"] if not complete else []),
            *([f"missing image files: {len(missing_images)}"] if missing_images else []),
            *([f"positive patients below 59: {len(positive_patients)}"] if not enough_positives else []),
            *(["overlaps with known development identifiers or exact image content"] if not no_overlap else []),
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
