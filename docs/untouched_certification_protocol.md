# ActiveRetina-RC Untouched Certification Protocol

Status: `REQUIRED_FOR_AUTONOMOUS_DIAGNOSIS`

The frozen autonomous candidate exists, but autonomous diagnosis requires a
new cohort that has not influenced any model, threshold, paper claim, or repair
decision.

## What Counts As Untouched

A cohort is untouched only if all of the following are true:

- The images and labels were not used to train, calibrate, select thresholds, or
  choose the model family.
- The labels were not inspected before the frozen artifact and thresholds were
  locked.
- The cohort has no patient, eye, or image overlap with DeepDRiD development,
  DeepDRiD online, swapped-mapping runs, or MFIDDR fixtures.
- The cohort contains complete paired views with fields named `macula` and
  `optic_disc`.
- The cohort includes at least 59 referable-DR positive patients; 80-120 is the
  preferred range.

## Required Files

1. A certification manifest CSV with columns:

   `patient_id,eye_id,image_id,field,grade,referable_dr,image_path`

2. An attestation JSON:

```json
{
  "untouched_for_active_retina_certification": true,
  "not_used_for_training_or_threshold_selection": true,
  "not_inspected_before_freeze": true,
  "source": "cohort/source name",
  "collector_or_owner": "name or institution",
  "attestation_date": "YYYY-MM-DD"
}
```

## Validation Command

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.validate_untouched_cohort \
  --manifest path/to/certification_manifest.csv \
  --attestation path/to/attestation.json \
  --output results/untouched_cohort_validation.json
```

The status must be `UNTOUCHED_CERTIFICATION_ELIGIBLE`.

## Certification Command

```bash
HF_HOME=.model_cache/huggingface TORCH_HOME=.model_cache/torch \
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.certify_autonomous_candidate \
  --manifest path/to/certification_manifest.csv \
  --feature-cache results/untouched_certification_features.npz \
  --output-predictions results/untouched_certification_predictions.csv \
  --output-summary results/untouched_certification_summary.json
```

The autonomous claim is allowed only if the summary decision is
`GO_AUTONOMOUS_5_PERCENT_95_CONFIDENCE_CERTIFIED`.

## Current Local Inventory

No current local labeled cohort qualifies as untouched:

- DeepDRiD train/validation: used for development and cross-validation.
- DeepDRiD online: already inspected and used for candidate development.
- MFIDDR sample fixture: too small and explicitly fixture-only.
- Historical/exposed test partitions: inspected during repair and excluded from
  confirmatory claims.

Therefore the model is computation-ready, but the autonomous diagnosis claim
requires a new cohort import.

