#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export HF_HOME="${HF_HOME:-.model_cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-.model_cache/torch}"

MANIFEST="${MANIFEST:-results/ddr_hf_specificity100_manifest.csv}"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[ActiveRetina] CUDA check"
.venv-active-retina/bin/python - <<'PY'
import torch
print({"cuda": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"})
PY

echo "[ActiveRetina] Fast head-only ConvNeXt run"
.venv-active-retina/bin/python -m active_retina.train_ddr_finetuned_autonomy \
  --manifest "$MANIFEST" \
  --output-dir "results/rtx3050_convnext_head_${STAMP}" \
  --artifact "models/active_retina_rc_rtx3050_convnext_head_${STAMP}.pt" \
  --model-name convnext_tiny.fb_in22k_ft_in1k \
  --epochs 20 \
  --batch-size 24 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --device cuda \
  --use-amp \
  --trainable-name-fragments head,classifier,fc \
  --threshold-safety-factors 0.95,0.9,0.8,0.7,0.5,0.25 \
  --min-specificity 0.70

echo "[ActiveRetina] Fine-tune ConvNeXt all parameters"
.venv-active-retina/bin/python -m active_retina.train_ddr_finetuned_autonomy \
  --manifest "$MANIFEST" \
  --output-dir "results/rtx3050_convnext_full_${STAMP}" \
  --artifact "models/active_retina_rc_rtx3050_convnext_full_${STAMP}.pt" \
  --model-name convnext_tiny.fb_in22k_ft_in1k \
  --epochs 30 \
  --batch-size 16 \
  --lr 2e-5 \
  --weight-decay 1e-4 \
  --device cuda \
  --use-amp \
  --threshold-safety-factors 0.95,0.9,0.8,0.7,0.5,0.25 \
  --min-specificity 0.70

echo "[ActiveRetina] Fine-tune DINO ViT all parameters"
.venv-active-retina/bin/python -m active_retina.train_ddr_finetuned_autonomy \
  --manifest "$MANIFEST" \
  --output-dir "results/rtx3050_vit_dino_full_${STAMP}" \
  --artifact "models/active_retina_rc_rtx3050_vit_dino_full_${STAMP}.pt" \
  --model-name vit_base_patch16_224.dino \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --device cuda \
  --use-amp \
  --threshold-safety-factors 0.95,0.9,0.8,0.7,0.5,0.25 \
  --min-specificity 0.70

echo "[ActiveRetina] Done. Inspect results/rtx3050_*_${STAMP}/summary.json"
