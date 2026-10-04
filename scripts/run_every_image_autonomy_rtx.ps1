$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "src"
$env:TORCH_HOME = ".model_cache\torch"
$env:HF_HOME = "external\model_cache"

if (-not (Test-Path ".venv-active-retina\Scripts\python.exe")) {
  Write-Error "Missing .venv-active-retina. Create/restore the project virtual environment first."
}

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

$summaryPath = "results\every_image_autonomy_rtx_convnext_tiny_gradeaware\summary.json"
if (Test-Path $summaryPath) {
  Get-Content $summaryPath
} else {
  Write-Error "Training finished without writing $summaryPath"
}
