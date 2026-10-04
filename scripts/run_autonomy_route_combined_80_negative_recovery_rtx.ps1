Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "C:\Users\bataa\active_retina"
$env:PYTHONPATH = "src"
$env:HF_HOME = "external/model_cache"
$env:TORCH_HOME = ".model_cache/torch"

.\.venv-active-retina\Scripts\python.exe -m active_retina.train_every_image_autonomy `
  --manifest results/autonomy_route_combined_manifest_resplit.csv `
  --output-dir results/autonomy_route_combined_80_negative_recovery_rtx `
  --artifact artifacts/autonomy_route_combined_80_negative_recovery_rtx.pt `
  --model-name tf_efficientnet_b0.ns_jft_in1k `
  --image-size 224 `
  --epochs 12 `
  --batch-size 24 `
  --num-workers 2 `
  --lr 6e-5 `
  --weight-decay 8e-4 `
  --target-specificity 0.80 `
  --grade0-boost 16 `
  --grade1-boost 8 `
  --grade2-boost 5 `
  --boundary-margin-weight 0.5 `
  --binary-weight 0.5 `
  --rank-weight 0.5 `
  --hard-boundary-weight 12 `
  --positive-margin 1.5 `
  --negative-margin 4.0 `
  --hard-positive-weight 4 `
  --hard-negative-weight 24 `
  --hard-grade2-weight 1.5 `
  --hard-grade1-weight 3 `
  --hard-top-fraction 0.50 `
  --device cuda `
  --use-amp
