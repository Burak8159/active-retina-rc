# Zero-Miss / 90% Specificity Recovery Status

Status: **not achieved with current DDR-only open-access model attempts**.

The requested autonomous operating point is:

- zero referable-DR misses,
- at least 90% specificity among nonreferable images,
- validation-selected threshold/rule,
- no test-set threshold repair.

## Attempts Completed

### Cached-feature probability models

Prior frozen-feature classifiers and MLP heads did not produce a 90/0 operating
region. The best zero-miss modes stayed in low-specificity refer-all or
near-refer-all regimes.

### Clearance ranker

Script: `src/active_retina/train_clearance_ranker.py`

Result: `results/clearance_ranker_specificity300_ensemble_h768/summary.json`

- Validation: 20.2% zero-miss nonreferable clearance specificity.
- Test: 11.0% specificity with 2 false nonreferable clearances.
- Decision: `NO_GO_CLEARANCE_RANKER_90_SPECIFICITY_ZERO_MISS`.

### Feature-geometry clearance

Script: `src/active_retina/feature_geometry_clearance_search.py`

Result: `results/geometry_clearance_specificity300_ensemble/summary.json`

- Validation: 11.3% zero-miss nonreferable clearance specificity.
- Test: 9.6% specificity with 13 false nonreferable clearances.
- Decision: `NO_GO_FEATURE_GEOMETRY_90_SPECIFICITY_ZERO_MISS`.

### Image fine-tuning

Script upgraded: `src/active_retina/train_ddr_finetuned_autonomy.py`

Added:

- high-resolution transform override,
- boundary-case weighted sampling,
- boundary-case weighted loss,
- ordinal MSE severity-regression mode.

Aborted runs:

- boundary-weighted binary ResNet-50, 448px, layer3/layer4/fc,
- boundary-weighted ordinal-MSE ResNet-50, 448px, layer3/layer4/fc.

Reason for abort: validation remained in the refer-all / zero-specificity regime
after the first useful validation checkpoints, so continuing would waste GPU time.

## Current Technical Interpretation

The blocking failure is not threshold search. The current DDR-only signal does
not yet separate nonreferable grade 0/1 images from referable grade 2+ images
well enough to support 90% specificity with zero missed referable cases.

In particular, grade 1 and grade 2 form the dangerous boundary. Any model that
clears most grade-1 images must avoid clearing visually similar grade-2 cases.
The current cached features, conservative geometry, and first high-resolution
fine-tuning attempts do not achieve that.

## Next Concrete Route

The next credible route is not another threshold/rule. It is a stronger DR
foundation signal:

1. use an EyePACS-trained or retinal-contrastive pretrained checkpoint,
2. train at 448-512px,
3. use ordinal severity loss,
4. ensemble multiple independently trained checkpoints,
5. select a zero-miss validation threshold once,
6. test once.

This is the route supported by published DR training-setting evidence: higher
resolution, ordinal/objective choice, and ensembling are the major performance
drivers for DR grading.
