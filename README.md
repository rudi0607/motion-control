# Motion Control Studio

Animate an **existing AI character image** with the motion from any **reference video**. The character image is the only identity source; the reference video is used purely as a motion source.

Generation runs on real hosted motion-transfer models. Nothing is simulated: if no provider key is configured, or a provider fails, the UI shows the actual error.

## Models

| Model | Provider | Identity | Notes |
|---|---|---|---|
| Kling v3 Pro Motion Control | fal.ai (`fal-ai/kling-video/v3/pro/motion-control`) | Facial element binding (`@Element1`) from an automatic face crop + optional extra angles | Orientation `video` (≤30 s, complex motion) or `image` (≤10 s, follows camera motion; no element binding) |
| Kling v3 Standard Motion Control | fal.ai | Same as Pro | Cheaper |
| Wan 2.2 Animate Move | fal.ai (`fal-ai/wan/v2.2-14b/animate/move`) | Character image | Seeded, 720p, no text prompt |
| Wan 2.2 Animate | Replicate (`wan-video/wan-2.2-animate-animation`) | Character image | Generates natively at 5–60 fps |

## Setup

```bash
npm install
cp .env.example .env   # then set FAL_KEY and/or REPLICATE_API_TOKEN
export $(grep -v '^#' .env | xargs)
npm run dev            # http://localhost:3000 (Express + Vite)
```

Production: `npm run build && npm start`. Requires `ffmpeg`/`ffprobe` on `PATH` and Node ≥ 20.

## Pipeline

1. **Analyze reference video**: ffprobe, then normalise to H.264/AAC MP4 (and trim to the model limit if you allow it).
2. **Extract pose, face and hand landmarks**: MediaPipe Pose/Hand/Face Landmarker runs frame by frame in the browser, with a live overlay. The models still do their own internal motion extraction; this step is for visualisation, pre-flight checks and QC.
3. **Lock identity**: detect the character's face, crop it, and bind it as a Kling facial element. Identity-preservation prompt guidance is added for models that accept prompts.
4. **Upload and generate**: files go to provider storage, and the job is submitted to the provider queue. Live queue position and logs stream to the UI over SSE. In-flight jobs resume polling after a server restart.
5. **Temporal processing (ffmpeg)**: frame-rate conversion (motion-compensated interpolation when Motion Smoothing is on), resolution scaling, duration matched to the reference, and reference audio muxed in if needed.
6. **Quality checks**:
   - ffmpeg (server): duration match, A/V sync, brightness flicker, sudden jumps, frozen motion, and motion-timing correlation, each compared against the reference.
   - MediaPipe (browser): face-geometry identity consistency against the character image, facial landmark jitter, hand/finger plausibility, 3D bone-length stability, joint-angle motion accuracy, blinking, mouth motion, and expression liveliness.
7. **Refine**: if flicker is flagged, the clip is re-rendered with a temporal deflicker filter and re-checked. Structural problems such as jumps or frozen segments can't be fixed locally, so the app offers **Regenerate**.
8. **Render final MP4**: H.264 High, yuv420p, faststart, and a download button.

## Honest limitations

- **Motion Strength** isn't exposed by any connected model API. The control is shown disabled.
- **2K/4K** output is Lanczos-upscaled from the model's native resolution (≤1080p). It adds pixels, not generated detail.
- Detail, background and scene options are **prompt guidance** (Kling only). Wan endpoints take no prompt.
- Automated checks are heuristics on sampled frames. They can't prove that every frame is perfect. For example, the 21-point hand model can't count fingers, and face geometry is not face recognition. Flagged time ranges link to the player for manual review.
- Lip sync comes from the reference performer's mouth motion. It isn't driven by an audio waveform.

## Scripts

- `npm test`: Vitest (settings, provider payload mapping, validation, and real ffmpeg render + QC on synthetic clips)
- `npm run typecheck`
- `npm run build`
