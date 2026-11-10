# Motion Control Studio

Animate an **existing AI character image** with the motion from any **reference video**. The character image is the only identity source; the reference video is used purely as a motion source.

**Fully self-hosted.** Generation runs open-source Wan models on **your own GPU**. No third-party AI API is used, and every model the browser needs (MediaPipe pose, hand and face) is served from this app. After the one-time weight download, the whole stack runs offline. Nothing is simulated: if the GPU worker is missing or fails, the UI shows the real error.

## Models (pick the one that fits your GPU)

| Model | GPU | Output | Notes |
|---|---|---|---|
| **Wan 2.1 VACE 1.3B** (`WAN_BACKEND=vace-1.3b`, default) | 8–12 GB, e.g. RTX 3050/3060 | 480p native, 16 fps (interpolated to your fps), up to 60 s | Pose skeleton drives your character. Long videos are generated in ~5 s parts and crossfaded. Face Lock re-applies your character's face. |
| **Wan 2.2 Animate 14B** (`WAN_BACKEND=animate-14b`) | 48–80 GB | 720p, 30 fps, up to 30 s | Best quality: full body, face and expression transfer with retargeting |

The worker runs one backend at a time, and the app enables only the model that matches it.

### Face Lock (both backends)

After generation, every frame goes through a face identity pass: SCRFD detection, then the ArcFace identity from your character image, then the inswapper face replacement and an optional GFPGAN restoration blend at 60% (full GFPGAN airbrushes skin). Keypoints are smoothed over time to avoid jitter. The same ArcFace model then measures each frame against your character, and the result appears as the **Face identity (ArcFace)** quality check, with the frames below the same-identity threshold flagged.

The face-lock models (insightface `inswapper_128`, `buffalo_l` ArcFace, GFPGAN) are about 1 GB and are downloaded once. **insightface model weights are licensed for non-commercial research use only**, so check the licenses before using the output commercially.

## Setup on one PC (single user)

Requirements: an NVIDIA GPU, a recent NVIDIA driver, Docker with the NVIDIA Container Toolkit, and 32 GB of system RAM recommended (the T5 text encoder runs on the CPU in low-VRAM mode). On Windows, use **Docker Desktop with the WSL2 backend**; the GPU is passed through automatically.

```bash
export WORKER_TOKEN=$(openssl rand -hex 32)      # PowerShell: $env:WORKER_TOKEN = [guid]::NewGuid()
docker compose up -d --build                     # app on http://localhost:3000
docker compose logs -f worker                    # first boot downloads the weights once
```

The app header turns green ("Self-hosted GPU ready · <your GPU>") once the worker is healthy. Out of memory on an 8 GB card? Set `VACE_FRAME_NUM=49` (≈3 s parts) in `.env` and restart. Long videos still work; they are just made from more parts. See [worker/README.md](worker/README.md) for details.

### Local development (app only)

```bash
npm install
cp .env.example .env   # SELFHOSTED_WORKER_URL / SELFHOSTED_WORKER_TOKEN
export $(grep -v '^#' .env | xargs)
npm run dev            # http://localhost:3000 (Express + Vite)
```

Production: `npm run build && npm start`. Requires `ffmpeg`/`ffprobe` on `PATH` and Node ≥ 20.

## Pipeline

1. **Analyze reference video**: ffprobe, then normalise to H.264/AAC MP4 (and trim to the model limit if you allow it).
2. **Extract pose, face and hand landmarks**: MediaPipe runs frame by frame in the browser, with a live overlay, from locally served models. This step is for preview, pre-flight checks and QC; the worker does its own pose extraction for the model.
3. **Identity reference**: detect the character's face and use the character image as the model's only appearance reference.
4. **Generate on your GPU**: upload to the worker, where the official upstream scripts run (VACE pose annotator + Wan2.1 `generate.py`, or Wan2.2-Animate preprocess + `generate.py`). Queue position and logs stream live to the UI.
5. **Temporal processing (ffmpeg)**: motion-compensated frame interpolation to your fps, resolution scaling, exact reference duration, and reference audio muxed in.
6. **Quality checks**:
   - ffmpeg (server): duration, A/V sync, brightness flicker, sudden jumps, frozen motion, and motion timing vs the reference.
   - MediaPipe (browser): face-geometry identity vs the character image, landmark jitter, hand plausibility, bone-length stability, joint-angle motion accuracy, blinking, mouth motion, and expression liveliness.
7. **Refine**: flicker triggers a deflicker re-render and a re-check. Structural problems can't be fixed locally, so the app offers **Regenerate**.
8. **Render final MP4**: H.264 High, faststart, with a download button.

## Honest limitations

- On an 8 GB GPU the body comes from VACE-1.3B at 480p. Face Lock makes the identity match your character (measured), but the face is still 480p detail (a 128 px swap plus restoration). It is not 1080p Kling-level skin detail.
- Long videos are joined from parts. The pose stays continuous, but clothing or background can shift slightly at the joins (the Motion continuity check flags them). Every part costs several minutes on a 3050, so 60 s takes a long time.
- **Motion Strength** isn't exposed by the models. The control is shown disabled.
- **2K/4K** output is a Lanczos upscale of the native output. It adds pixels, not detail.
- The automated checks are heuristics on sampled frames and can't guarantee every frame is correct. Flagged time ranges link to the player.
- Lip sync comes from the reference performer's mouth motion, not from the audio waveform.

## Scripts

- `npm test`: Vitest (settings, worker payloads, validation, real ffmpeg render + QC)
- `npm run test:worker`: pytest for the GPU worker (API, queue, auth, both backends' command lines, using stub model scripts)
- `npm run typecheck`, `npm run build`
