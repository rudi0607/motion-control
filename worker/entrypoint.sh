#!/usr/bin/env bash
# Downloads model weights once (into the /models volume), then serves the API fully offline.
set -euo pipefail

: "${WORKER_TOKEN:?Set WORKER_TOKEN to a long random secret}"
BACKEND="${WAN_BACKEND:-vace-1.3b}"

if [ "$BACKEND" = "animate-14b" ]; then
  CKPT="${WAN_CKPT_DIR:-/models/Wan2.2-Animate-14B}"
  if [ ! -d "$CKPT/process_checkpoint" ]; then
    echo "One-time download: Wan-AI/Wan2.2-Animate-14B -> $CKPT (~100 GB)"
    huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir "$CKPT"
  fi
else
  CKPT="${VACE_CKPT_DIR:-/models/Wan2.1-VACE-1.3B}"
  if [ ! -f "$CKPT/config.json" ]; then
    echo "One-time download: Wan-AI/Wan2.1-VACE-1.3B -> $CKPT"
    huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir "$CKPT"
  fi
  if [ ! -d /models/VACE-Annotators/pose ]; then
    echo "One-time download: VACE pose annotators -> /models/VACE-Annotators/pose"
    huggingface-cli download ali-vilab/VACE-Annotators --include "pose/*" --local-dir /models/VACE-Annotators
  fi
  # VACE resolves annotator paths relative to its repo root.
  mkdir -p "${VACE_REPO:-/opt/VACE}/models"
  ln -sfn /models/VACE-Annotators "${VACE_REPO:-/opt/VACE}/models/VACE-Annotators"
fi

# Face-lock models (identity detection, swap and restoration), ~1 GB, downloaded once.
FL="${FACELOCK_MODELS_DIR:-/models/facelock}"
FL_BASE="https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0"
mkdir -p "$FL"
for m in scrfd_2.5g.onnx arcface_w600k_r50.onnx inswapper_128.onnx gfpgan_1.4.onnx; do
  if [ ! -s "$FL/$m" ]; then
    echo "One-time download: $m -> $FL"
    curl -fL --retry 3 -o "$FL/$m.part" "$FL_BASE/$m" && mv "$FL/$m.part" "$FL/$m"
  fi
done

# From here on nothing may reach the internet: a hidden download fails loudly instead.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
