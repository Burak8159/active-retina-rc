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

Best safety-preserving run:

- Run: `results/ddr_open_autonomy_cert_subset/summary.json`
- Candidate: `extra_trees_depth8`
- Test positives: 120
- Missed positives: 0
- One-sided 95% missed-risk upper bound: 2.47%
- Specificity: 5.0%
- Decision: `NO_GO_SPECIFICITY`

Regularized logistic grid:

- Improved specificity as high as 20.0% in one run.
- Failed the zero-miss test requirement with 4 missed positives.
- Decision: `NO_GO_MISSED_RISK`

Partial expanded-train shortcut:

- Used extra DDR train images that had downloaded before the full 100-per-grade run completed.
- It worsened discrimination and specificity because the partial train folder was not yet balanced.
- Decision: failed as a shortcut; do not use as the main evidence.

## Current Decision

`NO_GO_SPECIFICITY`

The project now has a real zero-miss open-data safety pass, but not a useful full autonomous diagnostic mode. The next step is not more threshold tuning on the same 540-image subset. The next step is a balanced larger DDR training/validation/test subset or full DDR run.

## Next Run

Resume the balanced DDR download when network time is available:

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.prepare_ddr_hf \
  --local-dir external/DDR_hf_specificity100 \
  --manifest-output results/ddr_hf_specificity100_manifest.csv \
  --summary-output results/ddr_hf_specificity100_summary.json \
  --download-images \
  --limit-per-grade-per-split 100
```

Then run:

```bash
PYTHONPATH=src HF_HOME=.model_cache/huggingface TORCH_HOME=.model_cache/torch \
.venv-active-retina/bin/python -m active_retina.ddr_open_autonomy \
  --manifest results/ddr_hf_specificity100_manifest.csv \
  --feature-cache results/ddr_hf_specificity100_features.npz \
  --output-dir results/ddr_open_autonomy_specificity100 \
  --artifact models/active_retina_rc_ddr_specificity100.joblib \
  --batch-size 16 \
  --threshold-safety-factors 0.9,0.75,0.5,0.25
```

Expected purpose: determine whether more balanced training and validation data can preserve zero misses while materially improving specificity.

