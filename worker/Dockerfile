# GPU worker. Needs an NVIDIA GPU host with the NVIDIA container toolkit (Linux, or Windows via WSL2 + Docker Desktop).
#   docker build --build-arg WAN_BACKEND=vace-1.3b   -t motion-worker worker/   # 8–12 GB GPUs (default)
#   docker build --build-arg WAN_BACKEND=animate-14b -t motion-worker worker/   # 48–80 GB GPUs
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG WAN_BACKEND=vace-1.3b
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Pinned upstream commits for reproducible builds.
ARG WAN22_COMMIT=1ea34ff48f87168174e12956e200b1d908b1c5ff
ARG WAN21_COMMIT=9737cba9c1c3c4d04b33fcad41c111989865d315
ARG VACE_COMMIT=48eb44f1c4be87cc65a98bff985a26976841e9f3

# flash-attn is optional; skip it if no wheel matches so the build doesn't fail.
RUN set -eux; \
    if [ "$WAN_BACKEND" = "animate-14b" ]; then \
      git clone https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2 && git -C /opt/Wan2.2 checkout "$WAN22_COMMIT"; \
      grep -v '^flash_attn' /opt/Wan2.2/requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt; \
      pip install -r /opt/Wan2.2/requirements_animate.txt; \
    else \
      git clone https://github.com/Wan-Video/Wan2.1.git /opt/Wan2.1 && git -C /opt/Wan2.1 checkout "$WAN21_COMMIT"; \
      git clone https://github.com/ali-vilab/VACE.git /opt/VACE && git -C /opt/VACE checkout "$VACE_COMMIT"; \
      grep -v '^flash_attn' /opt/Wan2.1/requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt; \
      pip install -r /opt/VACE/requirements/annotator.txt; \
    fi; \
    pip install flash-attn --no-build-isolation || echo "flash-attn unavailable; using PyTorch SDPA"

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py facelock.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV WAN_BACKEND=${WAN_BACKEND} \
    WAN_REPO=/opt/Wan2.2 WAN_CKPT_DIR=/models/Wan2.2-Animate-14B \
    WAN21_REPO=/opt/Wan2.1 VACE_REPO=/opt/VACE VACE_CKPT_DIR=/models/Wan2.1-VACE-1.3B \
    PYTHONPATH=/opt/VACE \
    FACELOCK_MODELS_DIR=/models/facelock \
    WORKER_DATA_DIR=/data/jobs PORT=8000
VOLUME ["/models", "/data"]
EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
