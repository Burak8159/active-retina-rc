# ClementP EfficientNet B5 Extra-Signal Search

Status: `GO_EXTRA_SIGNAL_IMPROVES_AUTONOMY`

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

## Augmented Random-Mixture Max-Coverage Result

- Candidate: `random_weight_1067_clem0.53_conv0.14_dino0.07_feat0.09_mlp_0.09_ranknorm`
- Validation-selected low threshold: `0.129223`
- Validation-selected high threshold: `0.4625`
- Test false-negative clearances: `0`
- Test specificity among diagnostic decisions: `90.3%`
- Test diagnostic decision rate: `66.7%`
- Test insufficient-evidence/reacquire rate: `33.3%`
- Test negative-clearance rate: `38.0%`
- Test referable autonomous referral rate: `79.4%`

## Augmented Random-Mixture Conservative Result

- Candidate: `random_weight_0160_clem0.20_dino0.46_feat0.11_mlp_0.13_mlp_0.07_ranknorm`
- Validation-selected low threshold: `0.09375`
- Validation-selected high threshold: `0.55`
- Test false-negative clearances: `0`
- Test specificity among diagnostic decisions: `93.1%`
- Test diagnostic decision rate: `54.4%`
- Test insufficient-evidence/reacquire rate: `45.6%`
- Test negative-clearance rate: `27.6%`
- Test referable autonomous referral rate: `67.2%`

## Interpretation

The public EfficientNet model failed as a standalone policy, but it adds useful signal diversity to the existing score family. The result improves the product-relevant autonomy frontier under the current three-output contract:

`nonreferable / referable / insufficient-evidence-reacquire`

This is still not every-image binary diagnosis. It is a stronger completed autonomous triage subsystem with higher diagnostic coverage. The max-coverage mode reaches 66.7% diagnostic coverage at 90.3% specificity; the conservative mode trades coverage for a 93.1% specificity margin.

Frontier figure: `results/autonomy_extra_signal_clementp_efficientnet_b5/frontier.svg`
