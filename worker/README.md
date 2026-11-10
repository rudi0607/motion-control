# Self-hosted GPU worker

This worker runs open-source Wan video models on your own GPU. It doesn't use any third-party AI API, and your character images and videos stay on your machine. After the one-time weight download it runs offline (`HF_HUB_OFFLINE=1`), so any hidden download attempt fails loudly instead of quietly reaching the internet.

Pick one backend with `WAN_BACKEND`, which is both a build arg and an env var.

### `vace-1.3b` (default, for 8–12 GB GPUs such as an RTX 3050 8 GB)

1. `ffmpeg` resamples the reference to 16 fps, capped at `VACE_FRAME_NUM` frames (default 81 ≈ 5 s, must be 4n+1).
2. `vace/vace_preproccess.py --task pose` (VACE DWPose annotator) renders a pose-skeleton video.
3. Wan2.1 `generate.py --task vace-1.3B --src_video <pose> --src_ref_images <character> --offload_model True --t5_cpu` at 480p (832×480 or 480×832).

VRAM: the 1.3B family is documented at about 8.19 GB. **On an 8 GB card that is right at the limit and hasn't been tested here.** If you hit CUDA out-of-memory, lower `VACE_FRAME_NUM` to 49 or 33. Speed on a 3050 is roughly several minutes to tens of minutes per clip, depending on steps. You need about 30 GB of disk for the weights (`Wan-AI/Wan2.1-VACE-1.3B` plus the VACE pose annotators) and 32 GB of system RAM recommended (T5 runs on the CPU).

### `animate-14b` (for 48–80 GB GPUs)

1. `wan/modules/animate/preprocess/preprocess_data.py`: extracts pose and face from the reference video and retargets the skeleton to your character's body shape (`--retarget_flag --use_flux`), at 30 fps.
2. `generate.py --task animate-14B`: animates the character image with the extracted motion. Output is 720p at 30 fps.

The web app sends files to the worker, polls it, downloads the result, then runs its own steps: frame-rate and resolution conversion, reference-audio mux, quality checks and final MP4 render.

## GPU requirements (animate-14b)

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
docker build --build-arg WAN_BACKEND=vace-1.3b -t motion-worker worker/   # or animate-14b
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
| POST | `/v1/jobs` | `{image_file, video_file, backend?, resolution: "720p"\|"480p", refert_num: 1\|5, sample_steps, seed?}`. Returns 409 if `backend` differs from the worker's |
| GET | `/v1/jobs/{id}` | `status` (queued → preprocessing → generating → succeeded/failed/canceled), `queue_position`, logs, error |
| GET | `/v1/jobs/{id}/output` | Generated MP4 |
| POST | `/v1/jobs/{id}/cancel` | Kills the running step |

Jobs run one at a time per worker, because one model fills the GPU. Job state is saved to `/data`, so queued jobs survive a restart.

## Environment

`WORKER_TOKEN` (required), `WAN_BACKEND`, `VACE_FRAME_NUM`, `WAN21_REPO`, `VACE_REPO`, `VACE_CKPT_DIR`, `WAN_REPO`, `WAN_CKPT_DIR`, `WORKER_DATA_DIR`, `WAN_LOW_VRAM`, `WAN_NUM_GPUS`, `WAN_USE_FLUX` (default true), `WAN_RETARGET` (default true), `WORKER_JOB_TIMEOUT_SEC` (default 3 h), `PORT`.

## Tests

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt pytest httpx
.venv/bin/python -m pytest -q
```

The tests replace the upstream scripts (Wan2.2, Wan2.1, VACE) with stubs in `test_stubs/`. They check the API, auth, queue, error reporting and the exact command-line flags, but they don't run the real model.
