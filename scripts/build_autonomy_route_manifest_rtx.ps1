Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "C:\Users\bataa\active_retina"
$env:PYTHONPATH = "src"

.\.venv-active-retina\Scripts\python.exe -m active_retina.build_autonomy_route_manifest `
  --output results/autonomy_route_combined_manifest_rtx.csv `
  --summary-output results/autonomy_route_combined_manifest_rtx_summary.json

Get-Content results\autonomy_route_combined_manifest_rtx_summary.json
