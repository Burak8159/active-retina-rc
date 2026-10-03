# DDR Specificity-Recovery Status

Date: 2026-10-03

## Goal

Move beyond a zero-miss but near-refer-all DDR autonomy mode. The target is:

- zero missed referable DR on held-out test,
- one-sided 95% missed-risk upper bound below 5%,
- clinically useful specificity, currently tracked with a 70% target.

## What Improved

The DDR autonomy runner now separates three concepts:

1. **Validation-only artifact selection**: model family and threshold are selected without using test performance.
2. **Robust threshold candidates**: thresholds can be selected from safety-margined positive-score floors instead of only from brittle observed validation edges.
3. **Specificity-recovery reporting**: a separate report classifies runs as:
   - `GO_FULLER_SINGLE_IMAGE_AUTONOMY`
   - `NO_GO_SPECIFICITY`
   - `NO_GO_MISSED_RISK`

## Current Results

Best safety-preserving run after the larger DDR subset and stronger encoders:

- Run: `results/ddr_open_autonomy_specificity100_ensemble/summary.json`
- Candidate: three-encoder ensemble features with `logistic_c01_balanced`
- Test positives: 371
- Missed positives: 0
- One-sided 95% missed-risk upper bound: 0.80%
- AUROC: 0.889
- Average precision: 0.941
- Specificity: 10.5%
- Decision: `NO_GO_SPECIFICITY`

Diagnostic validation-edge thresholding nudged this to 11.0% specificity:

- Run: `results/ddr_open_autonomy_specificity100_ensemble_observed_thresholds/summary.json`
- Test positives: 371
- Missed positives: 0
- One-sided 95% missed-risk upper bound: 0.80%
- Specificity: 11.0%
- Boundary: diagnostic threshold search, not the preferred deployable artifact.

The larger open-data run used:

- 1,718 DDR images
- 600 train images
- 547 validation images
- 571 test images
- 371 referable-positive test images
- 0 missing images

Backbone comparison:

| Encoder/features | Test AUROC | Test AP | Misses | 95% UCB | Specificity | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ResNet-50 | 0.849 | 0.915 | 0/371 | 0.80% | 4.0% | `NO_GO_SPECIFICITY` |
| ConvNeXt-Tiny | 0.861 | 0.923 | 0/371 | 0.80% | 5.0% | `NO_GO_SPECIFICITY` |
| DINO ViT-B/16 | 0.871 | 0.924 | 0/371 | 0.80% | 8.0% | `NO_GO_SPECIFICITY` |
| Concatenated ensemble | 0.889 | 0.941 | 0/371 | 0.80% | 10.5% | `NO_GO_SPECIFICITY` |
| Concatenated ensemble, observed-threshold diagnostic | 0.889 | 0.941 | 0/371 | 0.80% | 11.0% | `NO_GO_SPECIFICITY` |

Post-hoc test-oracle check:

- Even when cheating by selecting the threshold directly on the test scores, the best zero-miss specificity among the current shallow-head candidates was only 11.5%.
- This means the current feature/head family is the blocker, not just validation threshold selection.

Fine-tuning check:

- A two-epoch ResNet-50 fine-tune on the larger DDR train split completed.
- It preserved zero misses only by referring every test image.
- Test specificity was 0.0%.
- This failed the autonomy target and showed that naive short fine-tuning is worse than the frozen ensemble.

RTX feature-head checks:

- The RTX 3050 Ti laptop path is operational with CUDA PyTorch.
- A 1024-hidden MLP preserved zero misses but reached only 7.0% test specificity.
- A stronger-regularized 512-hidden MLP improved validation specificity to 20.0%, but test specificity was 8.5%.
- Conclusion: shallow nonlinear heads do not solve the specificity problem on the current frozen features.

DRTiD recovery check:

- DRTiD is now inspected development data, not an untouched certificate.
- The recovery search produced a narrow safety-controlled triage mode:
  - 65 decided positive patients,
  - 0 missed positives,
  - 95% missed-risk upper bound 4.50%,
  - 70.0% specificity among decided negative patients,
  - 15.5% patient coverage.
- This is useful for the product as a limited autonomous sub-mode, but it is not full all-case autonomous diagnosis.

Larger DDR specificity300 check:

- The larger DDR slice is complete:
  - 4,238 images,
  - 2,723 referable-positive images,
  - 0 missing images,
  - 1,618 train / 1,185 validation / 1,435 test images.
- RTX CUDA feature extraction was used because Mac CPU/MPS extraction was too slow.
- ResNet-50 on the larger slice:
  - 946/946 test positives captured,
  - 95% upper missed-risk bound 0.32%,
  - specificity 6.5%.
- DINO ViT on the larger slice:
  - AUROC 0.939 and AP 0.968,
  - specificity 13.5%,
  - but 1 missed positive, so it is not a zero-miss artifact.
- ConvNeXt-Tiny on the larger slice:
  - specificity 2.5% with 4 missed positives.
- Three-encoder ensemble on the larger slice:
  - 946/946 test positives captured,
  - 95% upper missed-risk bound 0.32%,
  - specificity 8.4%.
- Conclusion: adding more immediate DDR images improved statistical power and representation metrics, but it still did not recover clinically useful all-case autonomous specificity.

Partial expanded-train shortcut:

- Used extra DDR train images that had downloaded before the full 100-per-grade run completed.
- It worsened discrimination and specificity because the partial train folder was not yet balanced.
- Decision: failed as a shortcut; do not use as the main evidence.

## Current Decision

`NO_GO_SPECIFICITY`

The project now has a stronger zero-miss open-data disease-capture certificate, but not a useful full autonomous diagnostic mode. The larger subset and stronger pretrained encoders improved discrimination and the finite-sample risk bound, but they did not recover enough autonomous nonreferable clearance.

Generated summary artifacts:

- `results/ddr_autonomy_summary/runs.csv`
- `results/ddr_autonomy_summary/decision.json`
- `results/ddr_autonomy_summary/specificity_recovery.svg`
- `results/ddr_autonomy_summary/risk_specificity_frontier.svg`

The next step is not more threshold tuning on these same frozen embeddings or naive short fine-tuning. The next step is better actual model learning:

1. fine-tune a retinal encoder with a real validation-monitored schedule rather than two quick epochs,
2. add image-quality and lesion-aware auxiliary heads,
3. keep DDR validation for threshold selection,
4. run DDR test once under the same zero-miss plus specificity target.

## Next Run

The balanced DDR specificity100 download completed successfully. To reproduce the current best run:

```bash
PYTHONPATH=src HF_HOME=.model_cache/huggingface TORCH_HOME=.model_cache/torch \
.venv-active-retina/bin/python -m active_retina.ddr_open_autonomy \
  --manifest results/ddr_hf_specificity100_manifest.csv \
  --feature-cache results/ddr_hf_specificity100_ensemble_features.npz \
  --output-dir results/ddr_open_autonomy_specificity100_ensemble \
  --artifact models/active_retina_rc_ddr_specificity100_ensemble.joblib \
  --batch-size 16 \
  --threshold-safety-factors 0.9,0.75,0.5,0.25
```

The next new run should be a fine-tuned encoder experiment rather than another shallow-head sweep:

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.train_ddr_finetuned_autonomy \
  --manifest results/ddr_hf_specificity100_manifest.csv \
  --output-dir results/ddr_finetuned_autonomy \
  --artifact models/active_retina_rc_ddr_finetuned_autonomy.pt
```

Expected purpose: determine whether task-specific representation learning can preserve zero misses while materially improving specificity.
