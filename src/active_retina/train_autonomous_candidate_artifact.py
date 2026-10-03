"""Freeze the next ActiveRetina-RC autonomous-candidate artifact.

This artifact is not a certified autonomous diagnostic product. It freezes the
candidate model family and thresholds selected from development stress testing
so the next step can be a single untouched certification run without repair.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .train_clinical_artifact import FEATURES, MANIFEST, group_by_eye, matrix, read_manifest, sha256_file


ARTIFACT_VERSION = "active-retina-rc-autonomous-candidate-2026-10-03"
OUTPUT = Path("models/active_retina_rc_autonomous_candidate.joblib")
METADATA = Path("models/active_retina_rc_autonomous_candidate.metadata.json")
MODEL_CARD = Path("docs/autonomous_candidate_model_card.md")
EXPANDED_SUMMARY = Path("results/expanded_cohort_development_summary.json")
EXPANDED_TABLE = Path("results/expanded_cohort_development_table.csv")
LOW_THRESHOLD = 0.00025252211213228293
HIGH_THRESHOLD = 0.9


def sha256_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def fit_candidate(x: np.ndarray, y: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=20261003),
    ).fit(x, y)


def main() -> None:
    features = {k: v for k, v in np.load(FEATURES, allow_pickle=False).items()}
    rows = read_manifest(MANIFEST)
    items = group_by_eye(rows, features)
    x_pair, y_ref, _ = matrix(items, ("macula", "optic_disc"))
    policy = {
        "mode": "paired_view_autonomous_candidate",
        "low_threshold": LOW_THRESHOLD,
        "high_threshold": HIGH_THRESHOLD,
        "score_below_or_equal_low": "autonomous_nonreferable_candidate",
        "score_above_or_equal_high": "autonomous_referable_candidate",
        "score_between_thresholds": "defer_to_clinician",
        "selected_from": "expanded development stress test at minimum 80% decided-negative specificity",
    }
    artifact = {
        "version": ARTIFACT_VERSION,
        "status": "FROZEN_AUTONOMOUS_CANDIDATE_NOT_CERTIFIED",
        "encoder": "resnet50.a1_in1k",
        "view_order": {"first": "macula", "second": "optic_disc"},
        "candidate_family": "logistic_c025_balanced",
        "policy": policy,
        "models": {
            "paired_binary": fit_candidate(x_pair, y_ref),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUTPUT)

    metadata = {
        "version": ARTIFACT_VERSION,
        "status": "FROZEN_AUTONOMOUS_CANDIDATE_NOT_CERTIFIED",
        "artifact": str(OUTPUT),
        "training_eyes": len(items),
        "training_positive_eyes": int(y_ref.sum()),
        "training_patients": len({item["patient_id"] for item in items}),
        "candidate_family": "logistic_c025_balanced",
        "encoder": "resnet50.a1_in1k",
        "view_order": artifact["view_order"],
        "policy": policy,
        "manifest_sha256": sha256_file(MANIFEST),
        "feature_cache_sha256": sha256_file(FEATURES),
        "expanded_development_summary_sha256": sha256_file(EXPANDED_SUMMARY) if EXPANDED_SUMMARY.exists() else None,
        "expanded_development_table_sha256": sha256_file(EXPANDED_TABLE) if EXPANDED_TABLE.exists() else None,
        "selection_boundary": (
            "Thresholds were selected on inspected/development evidence. The artifact may only become an "
            "autonomous product after a single no-repair run on an untouched certification cohort."
        ),
        "untouched_certification_requirements": {
            "minimum_decided_positive_patients": 59,
            "preferred_positive_patients": "80-120",
            "maximum_misses": 0,
            "confidence": 0.95,
            "missed_referable_dr_risk_target": 0.05,
            "threshold_or_model_repair_allowed_after_viewing_certification": False,
        },
    }
    metadata["policy_sha256"] = sha256_json(policy)
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    MODEL_CARD.write_text(
        f"""# ActiveRetina-RC Autonomous Candidate Model Card

Status: `FROZEN_AUTONOMOUS_CANDIDATE_NOT_CERTIFIED`

Artifact: `{OUTPUT}`

Version: `{ARTIFACT_VERSION}`

## Intended Computational Role

This artifact freezes the next autonomous-candidate computation package for an
untouched certification run. It is not a certified autonomous diagnostic
product and must not replace clinician review until the certification
requirements in the metadata are satisfied.

## Frozen Policy

- Encoder: `resnet50.a1_in1k`
- Classifier: `logistic_c025_balanced`
- Required views: paired macula/disc DeepDRiD-style fundus embeddings
- Low threshold: `{LOW_THRESHOLD}`
- High threshold: `{HIGH_THRESHOLD}`
- Score <= low: autonomous nonreferable candidate
- Score >= high: autonomous referable candidate
- Between thresholds: defer to clinician

## Development Evidence

Expanded local development stress testing combined 496 patients with 247
positive patients from available DeepDRiD prediction sources. The selected
80% specificity operating point decided 181 positive patients with zero misses,
80.2% decided-negative specificity, 61.3% patient coverage, and a one-sided
95% missed-risk upper bound of 1.64% if the same result were obtained
prospectively.

## Certification Boundary

Because the development and online cohorts have already been inspected, those
results cannot certify autonomous diagnosis. The next required step is a single
run on an untouched cohort with at least 59, preferably 80-120, positive
patients. Zero missed decided-positive patients are required.
""",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
