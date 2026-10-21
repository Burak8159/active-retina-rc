# Deployable Computation Gate

Status: `CLINICIAN_REVIEWED_SOFTWARE_EXISTS; FULL_BINARY_TARGET_NOT_MET`

ActiveRetina-RC now separates three things that must never be mixed:

1. Clinician-reviewed analysis software
2. Frozen autonomous-candidate computation
3. Final autonomous diagnosis certification

## Ready Now

- Original frozen paired-view autonomous candidate:
  `models/active_retina_rc_autonomous_candidate.joblib`
- Original frozen thresholds:
  - low: `0.00025252211213228293`
  - high: `0.9`
- DRTiD recovery candidate:
  `models/active_retina_rc_drtid_recovery_candidate.joblib`
- DRTiD recovery thresholds:
  - low: `0.2995880110412063`
  - high: `0.6554050430736857`
- One-shot certification runner:
  `src/active_retina/certify_autonomous_candidate.py`
- Untouched-cohort validator:
  `src/active_retina/validate_untouched_cohort.py`
- Generic raw cohort importer:
  `src/active_retina/prepare_untouched_certification_cohort.py`
- mBRSET-specific paired-view review importer:
  `src/active_retina/prepare_mbrset_for_untouched_review.py`
- Clinician-reviewed local/API station:
  `src/active_retina/clinical_station.py`

## Deployment Rule

The software may run images and produce structured reports, but it is not a
clinically validated diagnostic device. The stricter retrospective research
gate can pass only when the certification summary returns:

`GO_BINARY_80_0_100_RESEARCH_TARGET_MET`

The accepted certification run must satisfy all of the following:

- The cohort is attested untouched for this project.
- The cohort was not used for training, model repair, threshold selection, or
  manuscript tuning.
- The cohort has no patient, eye, or image overlap with known development data.
- Every eligible eye has same-eye macula and optic-disc views.
- At least 59 referable-DR positive patients are automatically decided with
  zero misses.
- 100% binary outputs across all eligible eye rows, zero false-negative
  referable eye rows, at least 80% patient-level specificity, and the one-sided
  95% patient missed-risk bound at or below 5%.
- The frozen artifact and thresholds are applied once, with no threshold repair.
- The certification command reports `certification_context=untouched`, not
  `inspected-development`.
- A pass is a retrospective computational result only. Clinical validation
  still requires a separate clinical study and is never inferred from this
  command.

## Current Product State

Some workflow and candidate-scoring code exists, but the current paired-view
candidate defers uncertain cases and therefore is not a full-binary model. The
MMRDR-UWF image model assigned a binary action to every held-out image but had
five misses and 20.02% specificity. Neither artifact is ready to establish the
full-binary clinical claim. Existing local data have also influenced model
development and cannot provide a new confirmatory result.

Current decision:

`GO_CLINICIAN_REVIEWED_COMPUTATIONAL_PRODUCT`

`RECOVERY_CANDIDATE_NOT_CLINICALLY_VALIDATED`

`NO_GO_AUTONOMOUS_DIAGNOSIS_BEFORE_CERTIFICATE`

## Latest Recovery State

The original DeepDRiD-derived candidate failed on DRTiD. After that failure,
DRTiD was used as inspected development evidence. A recovery candidate was
trained on the official DRTiD training split and replayed on the official DRTiD
testing split. The replay satisfies the finite-sample zero-miss arithmetic on
that inspected test split, but the command correctly labels the result:

`DEVELOPMENT_GO_NOT_A_CERTIFICATE`

The next valid autonomous product gate is a no-repair run of the DRTiD recovery
candidate, or a stronger frozen successor, on a different untouched paired-view
cohort.
