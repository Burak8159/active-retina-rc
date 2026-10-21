# ActiveRetina-RC Untouched Certification Protocol

Status: `REQUIRED_FOR_RESEARCH_CERTIFICATION`; clinical validation is a separate study.

The frozen paired-view candidate exists, but it is a selective model whose
middle score range defers to a clinician. It is not a full-coverage binary
diagnosis model. The MMRDR-UWF held-out test is a different, single-image model
evaluation and is not paired-view evidence or prospective clinical validation.

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
- For the project's full-binary `80/0/100` research target, every eligible eye
  must receive a binary output, there must be zero missed referable eyes and
  zero missed referable patients, the one-sided 95% patient miss-risk upper
  bound must be at most 5%, and patient-level specificity must be at least 80%.
- Ungradable and inference-failure handling must be specified before the run.
  A defer/abstain counts against 100% binary coverage; it is not a negative
  diagnosis. A forced referral for ungradable input is a referral action, not a
  claim that disease was diagnosed.

## Required Files

1. Either a raw paired-eye CSV with columns:

   `patient_id,eye_id,macula_image_path,optic_disc_image_path,grade,referable_dr`

   Convert it with:

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.prepare_untouched_certification_cohort \
  --input path/to/raw_paired_eye_table.csv \
  --output-manifest results/untouched_certification_manifest.csv \
  --output-summary results/untouched_certification_manifest_summary.json \
  --source-name untouched_external_certification
```

2. Or a certification manifest CSV already in canonical format:

   `patient_id,eye_id,image_id,field,grade,referable_dr,image_path`

3. An attestation JSON:

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

The computational target is met only if the summary decision is
`GO_BINARY_80_0_100_RESEARCH_TARGET_MET`. This result is still a retrospective
research certificate, not clinical validation, a safety guarantee for other
populations, or authorization for autonomous clinical use.

## Current Local Inventory

No current local labeled cohort qualifies as untouched:

- DeepDRiD train/validation: used for development and cross-validation.
- DeepDRiD online: already inspected and used for candidate development.
- MFIDDR sample fixture: too small and explicitly fixture-only.
- Historical/exposed test partitions: inspected during repair and excluded from
  confirmatory claims.
- MMRDR-UWF official test: already scored once with a frozen threshold; five
  positive image rows were missed and specificity was 20.02%. This is a held-out
  dataset test result, not an untouched paired-view certification cohort, and
  it cannot be reused to repair the model and then called confirmatory.

The validation guard now checks patient/eye/image identifiers across the
project's recorded development manifests using dataset namespaces, and checks
exact image-content hashes when the local files are available. It is still not
a substitute for a data custodian's overlap/untouched attestation. Therefore,
no clinical validation claim is currently supported; a new independently
collected cohort and a clinical study are needed.
