# Every-Image Autonomy Status

Target product claim:

```text
Every image receives an autonomous CLEAR or REFER decision,
zero referable-DR images are cleared,
and specificity is at least 90%.
```

Current status:

```text
NO_GO_EVERY_IMAGE_AUTONOMY
```

## What is already achieved

The project has a real selective autonomous subsystem:

- policy: `ActiveRetina-DDR-augmented-selective-fusion-v2`
- action space: `CLEAR / REFER / ABSTAIN`
- held-out DDR false-negative clearances: `0`
- held-out specificity among automatic decisions: `90.8%`
- held-out automatic coverage: `66.8%`
- held-out insufficient-evidence/reacquire rate: `33.2%`

This is a valid high-confidence autonomous subsystem, but it is not every-image diagnosis because it abstains on `33.2%` of held-out images. A stricter specificity-margin mode remains available with `94.7%` specificity and `46.3%` coverage.

## What failed for every-image autonomy

Four direct repair attempts were run.

1. Full-coverage meta-model over current score family:
   - best zero-miss test result: `0` false negatives
   - specificity: `1.4%`
   - conclusion: forcing every image with the current score family collapses specificity.

2. Optimistic oracle threshold analysis:
   - best candidate: `median_all`
   - even with test-label knowledge, zero-miss specificity ceiling: `17.4%`
   - distance from 90% specificity: `72.6` percentage points
   - conclusion: current fused scores do not contain enough separability for every-image 90/0.

3. Dedicated clearance ranker on regenerated DDR-300 ResNet features:
   - validation: zero misses, `21.6%` nonreferable clearance specificity
   - test: `8` referable images incorrectly cleared
   - test nonreferable clearance specificity: `12.9%`
   - conclusion: hard-case feature training did not generalize to safe every-image clearance.

4. Augmented full-binary score-family search:
   - added a public EfficientNet fundus signal and searched augmented forced-binary candidates
   - best current held-out result: `4` false negatives and `45.8%` specificity
   - conclusion: augmented scores improve selective autonomy but still do not support every-image binary diagnosis.

## Product interpretation

The safe product boundary is currently:

```text
GO_SELECTIVE_AUTONOMOUS_CLEAR_REFER_WITH_ABSTAIN
NO_GO_EVERY_IMAGE_AUTONOMOUS_DIAGNOSIS
```

Every-image binary autonomy cannot be honestly unlocked by thresholding, report wording, or the score-level families tested so far. It requires one of:

1. a stronger image model trained end-to-end on hard cases with sufficient compute,
2. additional independent evidence per case, such as paired/multi-view images or image-quality gated reacquisition,
3. a narrower product claim where ambiguous cases are not forced into autonomous clearance,
4. or a new untouched cohort showing that a frozen stronger model actually reaches zero misses and at least 90% specificity.

## Next executable route

The next non-circular technical move is:

```text
train end-to-end hard-case model on GPU
  -> optimize grade-aware referable boundary and conservative clearance
  -> select threshold on validation only
  -> test once
  -> accept only if false clearances = 0 and specificity >= 90%
```

The local CPU feature-cache route has been exhausted for this target.

## New executable trainer

Added:

```text
src/active_retina/train_every_image_autonomy.py
```

This script trains the direct every-image autonomy target:

- 6-class DR-grade head
- referable score = probability of grade `>= 2`
- validation-only zero-miss threshold selection
- frozen threshold applied once to test
- `GO_EVERY_IMAGE_AUTONOMY_90_0` only if test false-negative clearances are `0` and specificity is at least `90%`

A local offline smoke test passed mechanically using random initialization and a tiny subset:

```text
results/every_image_autonomy_smoke_resnet50/summary.json
```

That smoke test is not evidence for the model. It only verifies that the training/evaluation package runs.

RTX instructions:

```text
docs/rtx_every_image_autonomy_runbook.md
```
