#!/usr/bin/env bash
set -euo pipefail

cd /c/Users/bataa/active_retina

for cache in \
  ddr_hf_specificity300_features.npz \
  ddr_hf_specificity300_vit_dino_features.npz \
  ddr_hf_specificity300_convnext_tiny_features.npz \
  ddr_hf_specificity300_resnet_dino_features.npz \
  ddr_hf_specificity300_ensemble_features.npz
do
  stem="${cache%.npz}"
  PYTHONPATH=src .venv-active-retina/Scripts/python.exe -m active_retina.feature_geometry_clearance_search \
    --manifest results/ddr_hf_specificity300_manifest.csv \
    --feature-cache "results/${cache}" \
    --output-dir "results/geometry_clearance_sweep_${stem}_raw" \
    --target-specificity 0.90

  PYTHONPATH=src .venv-active-retina/Scripts/python.exe -m active_retina.feature_geometry_clearance_search \
    --manifest results/ddr_hf_specificity300_manifest.csv \
    --feature-cache "results/${cache}" \
    --output-dir "results/geometry_clearance_sweep_${stem}_l2" \
    --target-specificity 0.90 \
    --l2-normalize
done
