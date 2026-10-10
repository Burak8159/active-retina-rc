# ClementP EfficientNet B5 Extra-Signal Search

Status: `DEVELOPMENT_ONLY_TEST_EXPOSED`; prior test-selected headline is superseded.

This run tests whether the public `ClementP/FundusDRGrading-tf_efficientnet_b5` score helps ActiveRetina when used as an additional ensemble signal. It does not treat the model as a standalone certified classifier.

## Baseline

- Test images: `1435`
- Test positives: `946`
- False-negative clearances: `0`
- Specificity among diagnostic decisions: `94.7%`
- Diagnostic decision rate: `43.8%`
- Negative-clearance rate: `14.7%`

## Augmented Result

- Candidate: `top2_mean_augmented_ranknorm`
- Validation-selected low threshold: `0.075`
- Validation-selected high threshold: `0.55`
- Test false-negative clearances: `0`
- Test specificity among diagnostic decisions: `92.3%`
- Test diagnostic decision rate: `52.5%`
- Test insufficient-evidence/reacquire rate: `47.5%`
- Test negative-clearance rate: `22.1%`
- Test referable autonomous referral rate: `67.3%`

## Corrected Validation-Selected Random-Mixture Result

- Candidate: `random_weight_3992_clem0.60_conv0.07_mlp_0.24_resn0.06`
- Candidate and thresholds selected using validation only; full table: `results/autonomy_extra_signal_validation_audit_20261011_clean/summary.json`
- Validation-selected low threshold: `0.371136`
- Validation-selected high threshold: `0.628759`
- Test false-negative clearances: `0`
- Test specificity among diagnostic decisions: `83.0%`
- Test diagnostic decision rate: `72.6%`
- Test insufficient-evidence/reacquire rate: `27.4%`
- Test negative-clearance rate: `33.9%`
- Test referable autonomous referral rate: `89.0%`

The test split is exposed from previous searches, so these values are diagnostic only. If every insufficient-evidence case is forced to `REFER`, binary specificity is `166/489 = 33.9%` with zero observed misses. It does not meet binary `80/0/100`.

## Correction to the former headline

The former 90.8% / 66.8% result came from choosing the mixture with the strongest test metrics among 5,000 random mixtures. That is test-set selection and cannot support an independent held-out claim. The search implementation now ranks candidates using validation only; historical test metrics remain exposed development evidence.

The public EfficientNet signal remains a plausible contributor to a three-output development frontier:

`nonreferable / referable / insufficient-evidence-reacquire`

The corrected operating point is not every-image binary diagnosis, not a certification result, and not clinical-readiness evidence. All cases emit an action, but 27.4% are `INSUFFICIENT_EVIDENCE/REACQUIRE`; that is not binary diagnostic coverage.

Frontier figure: `results/autonomy_extra_signal_clementp_efficientnet_b5/frontier.svg`
