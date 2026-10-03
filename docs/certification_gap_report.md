# ActiveRetina-RC Certification Gap Report

Status: `CERTIFICATION_DATA_GAP_CONFIRMED`

## What Was Added

The local DeepDRiD v1.1 folder contains the official online challenge evaluation
images and labels. These were converted into a sealed certification-style
manifest and evaluated with the locked clinical artifact without threshold
tuning.

Artifacts:

- `results/deepdrid_online_certification_manifest.csv`
- `results/deepdrid_online_certification_predictions.csv`
- `results/deepdrid_online_certification_summary.json`
- `results/certification_mode_search_online.json`

## Untouched Online Cohort

- Patients: 100
- Complete paired eyes: 200
- Images: 400
- Referable eyes: 82
- Positive patients: 48

Locked artifact result:

- Decided positive patients: 47
- Missed positive patients: 7
- Patient missed-risk point estimate: 14.9%
- Patient missed-risk 95% upper bound: 26.2%
- Decision: `NO_GO_AUTONOMOUS_CERTIFICATION_FAILED_ON_ONLINE_COHORT`

## Best Post-Hoc Zero-Miss Mode

A post-hoc threshold search found a zero-miss mode on the online cohort, but it
is not a prospective certificate and is not a useful diagnostic claim by itself.

- Decided positive patients: 48
- Misses: 0
- 95% upper bound: 6.05%
- Required for 5% certificate: 59 zero-miss decided positive patients
- Shortfall: 11 positive patients

This means even the best post-hoc zero-miss mode cannot satisfy the 95%/5%
patient-level certificate on the current online cohort.

## What Must Happen Next

To make an autonomous 5% missed-DR claim possible:

1. Freeze a clinically meaningful operating mode before looking at the next
   certification cohort.
2. Add at least 11 more positive-patient cases if zero misses continue.
3. Prefer 80-120 positive-patient cases to tolerate uncertainty and avoid a
   brittle one-miss failure.
4. Preserve the current online cohort as already inspected; do not reuse it as a
   fresh prospective certificate after threshold search.
5. Use the next untouched cohort only once.

## Product Decision

Current product remains:

`GO_CLINICIAN_REVIEWED_ACQUISITION_ASSISTANT`

Current autonomous mode remains:

`NO_GO_AUTONOMOUS_DIAGNOSIS`

The obstacle is no longer software completeness. It is prospective statistical
evidence plus model performance under a frozen autonomous operating rule.

