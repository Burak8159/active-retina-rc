# Deployable Computation Gate

Status: `DEPLOYABLE_COMPUTATION_READY_CERTIFICATION_PENDING`

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

The software may run real images and produce structured analysis reports, but
the autonomous diagnosis switch remains off unless the certification summary
returns:

`GO_AUTONOMOUS_5_PERCENT_95_CONFIDENCE_CERTIFIED`

The accepted certification run must satisfy all of the following:

- The cohort is attested untouched for this project.
- The cohort was not used for training, model repair, threshold selection, or
  manuscript tuning.
- The cohort has no patient, eye, or image overlap with known development data.
- Every eligible eye has same-eye macula and optic-disc views.
- At least 59 referable-DR positive patients are automatically decided with
  zero misses.
- The frozen artifact and thresholds are applied once, with no threshold repair.
- The certification command reports `certification_context=untouched`, not
  `inspected-development`.

## Current Product State

The computation package is ready for a real certification event. The current
local labeled data are not eligible for final autonomous certification because
they have already influenced development.

Current decision:

`GO_CLINICIAN_REVIEWED_COMPUTATIONAL_PRODUCT`

`RECOVERY_CANDIDATE_READY_FOR_DIFFERENT_UNTOUCHED_CERTIFICATION`

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
