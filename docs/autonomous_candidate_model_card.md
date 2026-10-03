# ActiveRetina-RC Autonomous Candidate Model Card

Status: `FROZEN_AUTONOMOUS_CANDIDATE_NOT_CERTIFIED`

Artifact: `models/active_retina_rc_autonomous_candidate.joblib`

Version: `active-retina-rc-autonomous-candidate-2026-10-03`

## Intended Computational Role

This artifact freezes the next autonomous-candidate computation package for an
untouched certification run. It is not a certified autonomous diagnostic
product and must not replace clinician review until the certification
requirements in the metadata are satisfied.

## Frozen Policy

- Encoder: `resnet50.a1_in1k`
- Classifier: `logistic_c025_balanced`
- Required views: paired macula/disc DeepDRiD-style fundus embeddings
- Low threshold: `0.00025252211213228293`
- High threshold: `0.9`
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
