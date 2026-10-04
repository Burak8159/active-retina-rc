Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "C:\Users\bataa\active_retina"
$env:PYTHONPATH = "src"
$env:HF_HOME = "external/model_cache"
$env:TORCH_HOME = ".model_cache/torch"

.\.venv-active-retina\Scripts\python.exe -m active_retina.train_every_image_autonomy `
  --manifest results/autonomy_route_combined_manifest_resplit.csv `
  --output-dir results/autonomy_route_combined_80_rtx_fg_smoke `
  --artifact artifacts/autonomy_route_combined_80_rtx_fg_smoke.pt `
  --model-name resnet18 `
  --image-size 224 `
  --epochs 1 `
  --batch-size 16 `
  --num-workers 0 `
  --lr 1e-4 `
  --target-specificity 0.80 `
  --max-train 160 `
  --max-valid 100 `
  --max-test 100 `
  --grade0-boost 8 `
  --grade1-boost 5 `
  --grade2-boost 8 `
  --boundary-margin-weight 1 `
  --binary-weight 1 `
  --rank-weight 1 `
  --hard-boundary-weight 6 `
  --hard-positive-weight 8 `
  --hard-negative-weight 10 `
  --hard-top-fraction 0.35 `
  --device cuda `
  --use-amp
