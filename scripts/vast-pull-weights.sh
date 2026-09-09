#!/usr/bin/env bash
# Download Wan Diffusers weights onto the Vast instance (not the Mac).
set -euo pipefail

WEIGHTS="${WEIGHTS_ROOT:-/workspace/weights}"
mkdir -p "$WEIGHTS"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  pip install -q huggingface_hub
fi

hf_download() {
  local repo="$1"
  local dest="$2"
  shift 2
  echo "=== $repo -> $dest ==="
  mkdir -p "$dest"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" --local-dir "$dest" "$@"
  else
    huggingface-cli download "$repo" --local-dir "$dest" "$@"
  fi
}

# Runnable on RTX 4090 24GB (bf16).
hf_download Wan-AI/Wan2.1-T2V-1.3B-Diffusers "$WEIGHTS/Wan2.1-T2V-1.3B-Diffusers"

# CLIP ViT-H for I2V (needed even when the 14B DiT cannot fit in 24GB).
hf_download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers "$WEIGHTS/Wan2.1-I2V-14B-480P-Diffusers" \
  --include "image_encoder/*" \
  --include "image_processor/*" \
  --include "model_index.json"

# Full I2V 14B (~45GB). Generate will OOM on 24GB; layout/CLIP still useful.
if [ "${PULL_I2V_FULL:-0}" = "1" ]; then
  hf_download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers "$WEIGHTS/Wan2.1-I2V-14B-480P-Diffusers"
fi

# A14B dual 14B experts (~70GB) — skip unless disk and VRAM can take it.
if [ "${PULL_A14B:-0}" = "1" ]; then
  hf_download Wan-AI/Wan2.2-T2V-A14B-Diffusers "$WEIGHTS/Wan2.2-T2V-A14B-Diffusers"
fi

du -sh "$WEIGHTS"/* 2>/dev/null || true
df -h /workspace | tail -1
