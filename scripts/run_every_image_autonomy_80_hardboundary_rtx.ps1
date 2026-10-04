$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "src"
$env:TORCH_HOME = ".model_cache\torch"
$env:HF_HOME = "external\model_cache"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"

if (-not (Test-Path ".venv-active-retina\Scripts\python.exe")) {
  Write-Error "Missing .venv-active-retina. Create/restore the project virtual environment first."
}

.venv-active-retina\Scripts\python.exe -m active_retina.train_every_image_autonomy `
  --manifest results/ddr_hf_specificity300_manifest.csv `
  --output-dir results/every_image_autonomy_rtx_resnet50_hardboundary_80 `
  --artifact models/every_image_autonomy_rtx_resnet50_hardboundary_80.pt `
  --model-name resnet50.a1_in1k `
  --image-size 320 `
  --epochs 20 `
  --batch-size 12 `
  --num-workers 2 `
  --lr 2e-5 `
  --weight-decay 3e-4 `
  --grade0-boost 2.5 `
  --grade1-boost 6 `
  --grade2-boost 24 `
  --boundary-margin-weight 1.0 `
  --binary-weight 1.0 `
  --rank-weight 2.5 `
  --rank-margin 2.5 `
  --hard-boundary-weight 2.0 `
  --positive-margin 2.5 `
  --negative-margin 1.0 `
  --hard-positive-weight 14 `
  --hard-negative-weight 1.5 `
  --hard-grade2-weight 2.5 `
  --hard-grade1-weight 2.0 `
  --target-specificity 0.80 `
  --device cuda `
  --use-amp

$summaryPath = "results\every_image_autonomy_rtx_resnet50_hardboundary_80\summary.json"
if (Test-Path $summaryPath) {
  Get-Content $summaryPath
} else {
  Write-Error "Training finished without writing $summaryPath"
}
