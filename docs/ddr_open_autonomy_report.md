# DDR Immediate-Open Autonomy Report

Date: 2026-10-03

## Purpose

This track tests the strongest autonomy claim that can be attempted with a fully open,
immediately accessible dataset available now. DDR is a single-fundus-image diabetic
retinopathy grading dataset with official train, validation, and test label files.

DDR cannot certify the original paired-view ActiveRetina acquisition claim because it
does not provide same-eye macula/optic-disc paired acquisition episodes. It can,
however, test whether an autonomous single-image referable-DR escalation mode can be
certified on open data.

## Dataset

- Source mirror: `ctmedtech/DDR-dataset`
- Local subset: `external/DDR_hf_cert_subset`
- Manifest: `results/ddr_hf_cert_subset_manifest.csv`
- Rows: 540 images
- Splits: 180 train, 180 validation, 180 test
- Grade balance: 90 images per grade 0, 1, 2, 3, 4, 5
- Referable DR definition: ICDR-like grade >= 2
- Positive images: 360 total, 120 per split
- Missing images: 0

## Locked Evaluation Protocol

1. Train candidate binary heads on DDR train only.
2. Select the autonomous-referable threshold on DDR validation only.
3. Require zero missed referable cases on validation with at least 59 validation
   positive cases.
4. Prefer the zero-miss validation threshold with the highest validation specificity.
5. Evaluate the selected artifact once on DDR test.

The resulting mode is intentionally one-sided:

- `AUTONOMOUS_REFERABLE`: model score is above the threshold.
- `DEFER_TO_CLINICIAN`: model score is below the threshold.

No autonomous nonreferable clearance is claimed here.

## Result

Selected candidate: `extra_trees_depth8`

Validation:

- AUROC: 0.9183
- Average precision: 0.9590
- Threshold: 0.3797
- Positive cases: 120
- Missed positive cases: 0
- 95% upper missed-risk bound: 2.47%
- Specificity: 43.3%
- Referable action rate: 85.6%

Test:

- AUROC: 0.8489
- Average precision: 0.9221
- Positive cases: 120
- Missed positive cases: 0
- 95% upper missed-risk bound: 2.47%
- Specificity: 5.0%
- Precision of autonomous referable action: 67.8%
- Referable action rate: 98.3%

## Decision

`GO_OPEN_DDR_SINGLE_IMAGE_AUTONOMOUS_REFERABLE_CERTIFICATE`

This is a real open-data zero-miss certificate for one-sided autonomous referable
escalation on the DDR test subset:

- 120/120 referable test cases were autonomously referred.
- 0 referable test cases were missed.
- The 95% upper missed-risk bound was 2.47%, below the 5% target.

## Non-Negotiable Boundary

This is **not** a finished full autonomous diagnostic product.

The test specificity is only 5.0%, meaning nearly every nonreferable image is also
sent to autonomous referable escalation. That is safe against missed referable DR in
this open-data subset, but it is not operationally acceptable as a complete screening
replacement.

This result supports:

- Open-data single-image autonomous referable escalation.
- Evidence that the model can be made fail-open for disease capture.
- A paper section on immediate-open autonomy stress testing.

This result does not support:

- Autonomous nonreferable clearance.
- Paired-view acquisition autonomy.
- Full diagnostic replacement.
- Clinical deployment without a stronger specificity-preserving certificate.

## Next Technical Upgrade

The next model objective is no longer "find any zero-miss threshold." That has been
achieved. The next objective is:

> Preserve zero missed referable DR while recovering clinically useful specificity.

The immediate routes are:

1. Train on the full DDR train split rather than the 30-per-grade subset.
2. Add stronger feature models or fine-tuned retinal encoders.
3. Add explicit image-quality and uncertainty deferral so uncertain nonreferable
   cases do not become automatic false-positive referrals.
4. Validate on a separate untouched open or credentialed cohort after the operating
   mode is frozen.

