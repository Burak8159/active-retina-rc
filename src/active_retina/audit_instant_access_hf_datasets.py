"""Audit instant-access public retinal datasets for the autonomy route.

This script deliberately does not assume that every public Hugging Face dataset
is scientifically usable. It checks access, file layout, likely metadata, and
assigns each dataset to the ActiveRetina role it can honestly support.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


@dataclass(frozen=True)
class DatasetTarget:
    name: str
    repo_id: str
    priority: int
    role: str
    expected_support: list[str]
    cannot_support: list[str]
    notes: str


TARGETS = [
    DatasetTarget(
        name="Retinal DR longitudinal",
        repo_id="usama10/retinal-dr-longitudinal",
        priority=1,
        role="patient_level_longitudinal_pair_evidence",
        expected_support=["patient_ids", "paired_or_longitudinal_images", "domain_robustness"],
        cannot_support=["same_session_multifield_certification", "four_field_planning"],
        notes="Best instant route for repeated-evidence logic if files are public and labels are usable.",
    ),
    DatasetTarget(
        name="BRSET mirror / retina age analysis",
        repo_id="ramankamran/retina-age-analysis",
        priority=1,
        role="quality_anatomy_patient_level_robustness",
        expected_support=["patient_ids", "quality_labels", "anatomy_labels", "external_specificity"],
        cannot_support=["same_eye_multiview_acquisition_unless_view_columns_exist"],
        notes="Useful for quality gate and controlled-input claim.",
    ),
    DatasetTarget(
        name="DR grading Africa",
        repo_id="macular/diabetic-retinopathy-grading-africa",
        priority=2,
        role="external_domain_robustness",
        expected_support=["single_image_external_validation", "population_shift"],
        cannot_support=["multiview_acquisition", "case_level_reacquisition_certification"],
        notes="Good stress test for geographic/domain shift.",
    ),
    DatasetTarget(
        name="DR screening benchmark Africa",
        repo_id="macular/diabetic-retinopathy-screening-benchmark-africa",
        priority=2,
        role="external_screening_robustness",
        expected_support=["single_image_external_validation", "screening_endpoint"],
        cannot_support=["multiview_acquisition"],
        notes="Potential screening-specific external validation.",
    ),
    DatasetTarget(
        name="EyePACS HF",
        repo_id="ctmedtech/EYEPACS",
        priority=3,
        role="large_scale_single_image_training",
        expected_support=["encoder_training", "single_image_generalization"],
        cannot_support=["patient_level_certification_if_no_patient_ids", "multiview_acquisition"],
        notes="Large training diversity, but not enough for the acquisition claim.",
    ),
    DatasetTarget(
        name="DDR HF",
        repo_id="ctmedtech/DDR-dataset",
        priority=3,
        role="single_image_training_and_lesion_stress",
        expected_support=["encoder_training", "single_image_validation"],
        cannot_support=["multiview_acquisition"],
        notes="Already partly used; audit confirms whether larger/full mirror can extend training.",
    ),
    DatasetTarget(
        name="APTOS HF",
        repo_id="OctoMed/Aptos",
        priority=3,
        role="single_image_external_validation",
        expected_support=["single_image_validation", "grade_scale_stress"],
        cannot_support=["patient_level_multiview_acquisition"],
        notes="Useful for single-image external validation only.",
    ),
    DatasetTarget(
        name="IDRiD VQA mirror",
        repo_id="vermayuvraj/idrid-vqa",
        priority=4,
        role="lesion_reasoning_auxiliary",
        expected_support=["lesion_reasoning", "explainability"],
        cannot_support=["primary_binary_certification", "multiview_acquisition"],
        notes="VQA wrapper can support paper explainability, not primary product certification.",
    ),
    DatasetTarget(
        name="RFMiD HF",
        repo_id="ctmedtech/RFMID",
        priority=4,
        role="confounder_specificity",
        expected_support=["multi_disease_specificity", "non_dr_confounders"],
        cannot_support=["dr_grade_certification_unless_labels_map_cleanly"],
        notes="Useful to reduce false referrals from non-DR disease confounders.",
    ),
]


def size_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024 * 1024), 2)


def suffix_counts(files: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for file in files:
        suffix = Path(file).suffix.lower() or "[no_suffix]"
        counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def classify_files(files: list[str]) -> dict[str, Any]:
    lower = [f.lower() for f in files]
    metadata = [
        f
        for f, lf in zip(files, lower)
        if lf.endswith((".csv", ".json", ".jsonl", ".txt", ".tsv", ".xlsx", ".parquet"))
        or "metadata" in lf
        or "label" in lf
    ]
    images = [f for f, lf in zip(files, lower) if lf.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"))]
    readmes = [f for f in files if Path(f).name.lower().startswith("readme")]
    return {
        "n_files": len(files),
        "n_image_files_visible": len(images),
        "n_metadata_like_files": len(metadata),
        "readme_files": readmes[:20],
        "metadata_like_files_sample": metadata[:30],
        "image_files_sample": images[:10],
        "suffix_counts": dict(list(suffix_counts(files).items())[:20]),
    }


def audit_target(api: HfApi, target: DatasetTarget) -> dict[str, Any]:
    row: dict[str, Any] = {"target": asdict(target)}
    try:
        info = api.dataset_info(target.repo_id, files_metadata=True)
        siblings = list(getattr(info, "siblings", []) or [])
        files = [str(s.rfilename) for s in siblings]
        total_size = sum(int(getattr(s, "size", 0) or 0) for s in siblings)
        row.update(
            {
                "status": "ACCESSIBLE",
                "private": bool(getattr(info, "private", False)),
                "downloads": getattr(info, "downloads", None),
                "likes": getattr(info, "likes", None),
                "sha": getattr(info, "sha", None),
                "total_size_mb_visible": size_mb(total_size),
                **classify_files(files),
            }
        )
    except Exception as exc:  # noqa: BLE001 - audit must not stop on one inaccessible dataset.
        row.update({"status": "BLOCKED_OR_INACCESSIBLE", "error": f"{type(exc).__name__}: {exc}"})
    return row


def decision(audit: dict[str, Any]) -> str:
    if audit.get("status") != "ACCESSIBLE":
        return "NO_GO_ACCESS_BLOCKED"
    target = audit["target"]
    supports = set(target["expected_support"])
    n_images = int(audit.get("n_image_files_visible", 0) or 0)
    n_meta = int(audit.get("n_metadata_like_files", 0) or 0)
    if "patient_ids" in supports and n_meta > 0:
        return "GO_HIGH_PRIORITY_AUDIT_AND_PREP"
    if n_images > 0 or n_meta > 0:
        return "GO_SECONDARY_TRAINING_OR_VALIDATION"
    return "CONDITIONAL_METADATA_HIDDEN_NEEDS_DATASET_LIBRARY"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/instant_access_hf_dataset_audit.json")
    parser.add_argument("--matrix-output", default="results/instant_access_hf_dataset_matrix.csv")
    args = parser.parse_args()

    api = HfApi()
    audits = [audit_target(api, target) for target in TARGETS]
    for audit in audits:
        audit["active_retina_decision"] = decision(audit)
    summary = {
        "status": "INSTANT_ACCESS_HF_AUDIT_COMPLETE",
        "n_targets": len(audits),
        "n_accessible": sum(1 for row in audits if row.get("status") == "ACCESSIBLE"),
        "n_high_priority": sum(1 for row in audits if row.get("active_retina_decision") == "GO_HIGH_PRIORITY_AUDIT_AND_PREP"),
        "rows": audits,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = [
        "priority",
        "name",
        "repo_id",
        "status",
        "active_retina_decision",
        "role",
        "total_size_mb_visible",
        "n_files",
        "n_image_files_visible",
        "n_metadata_like_files",
        "notes",
    ]
    matrix = Path(args.matrix_output)
    with matrix.open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in audits:
            target = row["target"]
            writer.writerow(
                {
                    "priority": target["priority"],
                    "name": target["name"],
                    "repo_id": target["repo_id"],
                    "status": row.get("status"),
                    "active_retina_decision": row.get("active_retina_decision"),
                    "role": target["role"],
                    "total_size_mb_visible": row.get("total_size_mb_visible"),
                    "n_files": row.get("n_files"),
                    "n_image_files_visible": row.get("n_image_files_visible"),
                    "n_metadata_like_files": row.get("n_metadata_like_files"),
                    "notes": target["notes"],
                }
            )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
