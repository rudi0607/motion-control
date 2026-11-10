# GPU worker for Wan2.2-Animate-14B. Needs an NVIDIA GPU host with the NVIDIA container toolkit.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Pinned upstream commit for reproducible builds.
ARG WAN_COMMIT=1ea34ff48f87168174e12956e200b1d908b1c5ff
RUN git clone https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2 && \
    git -C /opt/Wan2.2 checkout "$WAN_COMMIT"

WORKDIR /opt/Wan2.2
# flash-attn is optional; skip it if no wheel matches so the build doesn't fail.
RUN grep -v '^flash_attn' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt && \
    pip install -r requirements_animate.txt && \
    (pip install flash-attn --no-build-isolation || echo "flash-attn unavailable; using PyTorch SDPA")

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV WAN_REPO=/opt/Wan2.2 WAN_CKPT_DIR=/models/Wan2.2-Animate-14B WORKER_DATA_DIR=/data/jobs PORT=8000
VOLUME ["/models", "/data"]
EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
