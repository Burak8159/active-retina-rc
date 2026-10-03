# DRTiD External Certification Report

Status: `NO_GO_AUTONOMOUS_CERTIFICATION_FAILED_ON_DRTID`

DRTiD access was obtained through the returned Google Drive link after the
application form. The archive was downloaded, unpacked, and processed through
the one-shot certification package.

## Cohort

- 3,100 images
- 1,550 paired two-field samples
- 1,550 unique IDs in the grade CSV
- 663 referable-DR positive IDs using grade >= 2
- 887 nonreferable IDs
- Complete macula + optic-disc pairs
- Missing image count: 0

The cohort clears the minimum sample-size requirement for a 95% confidence / 5%
missed-risk certificate.

## Frozen DeepDRiD Candidate Result

Artifact:

`models/active_retina_rc_autonomous_candidate.joblib`

Output:

`results/drtid_untouched_certification/final_deployability_decision.json`

Decision:

`NO_GO_AUTONOMOUS_CERTIFICATION_FAILED`

Key numbers:

- Decided IDs: 739 / 1,550
- Decided positive IDs: 310
- Missed positive IDs: 118
- Missed-risk point estimate: 38.1%
- One-sided 95% missed-risk upper bound: 42.8%
- Decided-negative specificity: 94.4%

This is a real external-domain failure. The frozen DeepDRiD-derived autonomous
candidate does not transfer to DRTiD safely.

## Score Diagnostics

Output:

`results/drtid_external_score_diagnostics.json`

- AUROC: 0.758
- Average precision: 0.735

The failure is not merely a threshold issue. Discrimination is substantially
weaker on DRTiD than on the local DeepDRiD development pool.

## DRTiD-Specific Recovery Diagnostic

Output:

`results/drtid_domain_adaptation_recovery_diagnostic.json`

A DRTiD-specific logistic head trained on the official training split and
evaluated on the official testing split reached:

- Best test AUROC across tested regularization values: 0.812
- Best test average precision: about 0.800

However, it did not produce a usable zero-miss autonomous nonreferable mode on
the already-inspected DRTiD test split. Positive-only referral modes can avoid
misses, but they are not a useful autonomous screening product because they do
not safely clear nonreferable patients.

## Consequence

DRTiD substantially strengthens the paper because it gives a real external
paired-view failure case. It does not finish the autonomous product.

Current decision:

`GO_CLINICIAN_REVIEWED_COMPUTATIONAL_PRODUCT`

`NO_GO_AUTONOMOUS_DIAGNOSIS_ON_DRTID`

`CONTINUE_MODEL_UPGRADE_AND_MBRSET_CERTIFICATION`

