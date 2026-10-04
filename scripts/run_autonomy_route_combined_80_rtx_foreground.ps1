Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "C:\Users\bataa\active_retina"
$env:PYTHONPATH = "src"
$env:HF_HOME = "external/model_cache"
$env:TORCH_HOME = ".model_cache/torch"

.\.venv-active-retina\Scripts\python.exe -m active_retina.train_every_image_autonomy `
  --manifest results/autonomy_route_combined_manifest_resplit.csv `
  --output-dir results/autonomy_route_combined_80_rtx `
  --artifact artifacts/autonomy_route_combined_80_rtx.pt `
  --model-name tf_efficientnet_b0.ns_jft_in1k `
  --image-size 384 `
  --epochs 8 `
  --batch-size 12 `
  --num-workers 2 `
  --lr 3e-5 `
  --weight-decay 5e-4 `
  --target-specificity 0.80 `
  --grade0-boost 8 `
  --grade1-boost 5 `
  --grade2-boost 8 `
  --boundary-margin-weight 1 `
  --binary-weight 1 `
  --rank-weight 1 `
  --hard-boundary-weight 6 `
  --positive-margin 2.5 `
  --negative-margin 2.5 `
  --hard-positive-weight 8 `
  --hard-negative-weight 10 `
  --hard-grade2-weight 2 `
  --hard-grade1-weight 2 `
  --hard-top-fraction 0.35 `
  --device cuda `
  --use-amp
