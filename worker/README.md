# Self-hosted GPU worker (Wan2.2-Animate-14B)

This worker runs the open-source [Wan2.2-Animate-14B](https://github.com/Wan-Video/Wan2.2) model on your own GPU. It doesn't use any third-party AI API, and your character images and videos stay on your machine.

For each job it runs the two official upstream steps:

1. `wan/modules/animate/preprocess/preprocess_data.py`: extracts pose and face from the reference video and retargets the skeleton to your character's body shape (`--retarget_flag --use_flux`), at 30 fps.
2. `generate.py --task animate-14B`: diffusion that animates the character image with the extracted motion. Output is 720p at 30 fps.

The web app sends files to the worker, polls it, downloads the result, then runs its own steps: frame-rate and resolution conversion, reference-audio mux, quality checks and final MP4 render.

## GPU requirements

| GPU memory | Setting | Notes |
|---|---|---|
| 80 GB (A100/H100) | default | Recommended |
| 48 GB (L40S, RTX 6000 Ada, A6000) | `WAN_LOW_VRAM=true` | Turns on model offload, dtype conversion and T5 on CPU. Slower. |
| < 48 GB | not supported | |
| Multiple GPUs | `WAN_NUM_GPUS=N` | Uses FSDP + Ulysses sequence parallelism |

You need roughly 100 GB of disk for the weights (`Wan-AI/Wan2.2-Animate-14B`, downloaded on first boot into the `/models` volume) and 32 GB+ of system RAM.

## Run with Docker (any GPU server: RunPod, Lambda, Vast, your own PC)

```bash
export WORKER_TOKEN=$(openssl rand -hex 32)
docker build -t motion-worker worker/
docker run --gpus all -p 8000:8000 --shm-size 16g \
  -e WORKER_TOKEN=$WORKER_TOKEN \
  -v motion-models:/models -v motion-jobs:/data motion-worker
```

Then start the web app with:

```bash
SELFHOSTED_WORKER_URL=http://<gpu-host>:8000 SELFHOSTED_WORKER_TOKEN=$WORKER_TOKEN npm start
```

To run both on one GPU machine, use `docker compose up -d --build` from the repo root.

Check health with `curl -H "Authorization: Bearer $WORKER_TOKEN" http://<gpu-host>:8000/v1/health`. It reports the GPU, missing weights, or a missing Wan repo. The web app shows the same status in its header.

## API

Every endpoint requires `Authorization: Bearer $WORKER_TOKEN`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/health` | GPU, weights and queue status |
| POST | `/v1/files` | multipart `file`, returns `{file_id}` |
| POST | `/v1/jobs` | `{image_file, video_file, resolution: "720p"\|"480p", refert_num: 1\|5, sample_steps, seed?}` |
| GET | `/v1/jobs/{id}` | `status` (queued → preprocessing → generating → succeeded/failed/canceled), `queue_position`, logs, error |
| GET | `/v1/jobs/{id}/output` | Generated MP4 |
| POST | `/v1/jobs/{id}/cancel` | Kills the running step |

Jobs run one at a time per worker, because one 14B model fills a GPU. Job state is saved to `/data`, so queued jobs survive a restart.

## Environment

`WORKER_TOKEN` (required), `WAN_REPO`, `WAN_CKPT_DIR`, `WORKER_DATA_DIR`, `WAN_LOW_VRAM`, `WAN_NUM_GPUS`, `WAN_USE_FLUX` (default true), `WAN_RETARGET` (default true), `WORKER_JOB_TIMEOUT_SEC` (default 3 h), `PORT`.

## Tests

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt pytest httpx
.venv/bin/python -m pytest -q
```

The tests replace the two Wan scripts with stubs in `test_stubs/`. They check the API, auth, queue, error reporting and the exact command-line flags, but they don't run the real model.
