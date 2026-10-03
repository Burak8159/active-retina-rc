# Deployable Computation Gate

Status: `DEPLOYABLE_COMPUTATION_READY_CERTIFICATION_PENDING`

ActiveRetina-RC now separates three things that must never be mixed:

1. Clinician-reviewed analysis software
2. Frozen autonomous-candidate computation
3. Final autonomous diagnosis certification

## Ready Now

- Frozen paired-view autonomous candidate:
  `models/active_retina_rc_autonomous_candidate.joblib`
- Frozen thresholds:
  - low: `0.00025252211213228293`
  - high: `0.9`
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

## Current Product State

The computation package is ready for a real certification event. The current
local labeled data are not eligible for final autonomous certification because
they have already influenced development.

Current decision:

`GO_CLINICIAN_REVIEWED_COMPUTATIONAL_PRODUCT`

`CONDITIONAL_GO_AUTONOMOUS_DIAGNOSIS_AFTER_UNTOUCHED_CERTIFICATION`

`NO_GO_AUTONOMOUS_DIAGNOSIS_BEFORE_CERTIFICATE`

