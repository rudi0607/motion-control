#!/usr/bin/env bash
# Downloads Wan2.2-Animate-14B weights on first boot (into a persistent volume), then serves the API.
set -euo pipefail

: "${WORKER_TOKEN:?Set WORKER_TOKEN to a long random secret}"
CKPT="${WAN_CKPT_DIR:-/models/Wan2.2-Animate-14B}"

if [ ! -d "$CKPT/process_checkpoint" ]; then
  echo "Downloading Wan-AI/Wan2.2-Animate-14B to $CKPT (one-time, large download)…"
  mkdir -p "$CKPT"
  huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir "$CKPT"
fi

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
