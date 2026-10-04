# RTX Every-Image Autonomy Runbook

Goal:

```text
GO_EVERY_IMAGE_AUTONOMY_90_0
```

Acceptance gate:

- every DDR test image receives `CLEAR` or `REFER`
- test false-negative clearances = `0`
- test specificity >= `90%`
- threshold selected on validation only

## Why RTX is required

The CPU/cached-feature path has been exhausted for the every-image target. The next valid move is end-to-end image training with a grade-aware head and hard boundary weighting.

The executable trainer is:

```text
src/active_retina/train_every_image_autonomy.py
```

It trains a 6-grade DR classifier, converts grade probabilities into referable risk, selects the most specific zero-miss threshold on validation, and tests once.

## Recommended RTX command

From the project root on the RTX machine:

```powershell
$env:PYTHONPATH="src"
$env:TORCH_HOME=".model_cache\torch"
$env:HF_HOME="external\model_cache"
$env:HF_HUB_ENABLE_HF_TRANSFER="0"

.venv-active-retina\Scripts\python.exe -m active_retina.train_every_image_autonomy `
  --manifest results/ddr_hf_specificity300_manifest.csv `
  --output-dir results/every_image_autonomy_rtx_convnext_tiny_gradeaware `
  --artifact models/every_image_autonomy_rtx_convnext_tiny_gradeaware.pt `
  --model-name convnext_tiny.in12k_ft_in1k `
  --image-size 320 `
  --epochs 30 `
  --batch-size 16 `
  --num-workers 2 `
  --lr 2e-5 `
  --weight-decay 2e-4 `
  --grade2-boost 18 `
  --grade1-boost 5 `
  --boundary-margin-weight 1.0 `
  --device cuda `
  --use-amp
```

If the RTX runs out of memory:

```text
image-size 256, batch-size 8
```

## Stronger second command

If the first run does not reach the gate, run a higher boundary-pressure version:

```powershell
$env:PYTHONPATH="src"
$env:TORCH_HOME=".model_cache\torch"
$env:HF_HOME="external\model_cache"

.venv-active-retina\Scripts\python.exe -m active_retina.train_every_image_autonomy `
  --manifest results/ddr_hf_specificity300_manifest.csv `
  --output-dir results/every_image_autonomy_rtx_convnext_tiny_boundary2 `
  --artifact models/every_image_autonomy_rtx_convnext_tiny_boundary2.pt `
  --model-name convnext_tiny.in12k_ft_in1k `
  --image-size 320 `
  --epochs 45 `
  --batch-size 16 `
  --num-workers 2 `
  --lr 1e-5 `
  --weight-decay 5e-4 `
  --grade2-boost 24 `
  --grade1-boost 8 `
  --boundary-margin-weight 2.0 `
  --device cuda `
  --use-amp
```

## How to read the result

Open:

```text
results/every_image_autonomy_rtx_convnext_tiny_gradeaware/summary.json
```

or:

```text
results/every_image_autonomy_rtx_convnext_tiny_boundary2/summary.json
```

The only successful state is:

```json
"status": "GO_EVERY_IMAGE_AUTONOMY_90_0"
```

Anything else remains a failed autonomous-diagnosis candidate.
