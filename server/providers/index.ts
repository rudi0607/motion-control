import { lastHealth, selfhostedAdapter } from "./selfhosted.js";
import type { ModelSpec, ProviderAdapter, ProviderId } from "./types.js";

export const adapters: Record<ProviderId, ProviderAdapter> = { selfhosted: selfhostedAdapter };
export const REQUIRED_ENV: Record<ProviderId, string> = {
  selfhosted: "SELFHOSTED_WORKER_URL + SELFHOSTED_WORKER_TOKEN",
};
export const PROVIDER_NAME: Record<ProviderId, string> = { selfhosted: "your GPU worker" };

const NO_MOTION_STRENGTH = {
  supported: false,
  how: "Not exposed by this model's API — the output always follows the full reference motion.",
};

/** Uses the worker's reported limit when it runs this model's backend. */
const workerLimit = (backend: ModelSpec["workerBackend"], fallback: number) => () =>
  lastHealth?.backend === backend && typeof lastHealth.max_duration_sec === "number" ? lastHealth.max_duration_sec : fallback;

export const MODELS: ModelSpec[] = [
  {
    id: "wan-2.1-vace-selfhosted",
    provider: "selfhosted",
    workerBackend: "vace-1.3b",
    label: "Wan 2.1 VACE 1.3B (your GPU · 8–12 GB)",
    endpoint: "worker:/v1/jobs (Wan2.1-VACE-1.3B pose-to-video)",
    description:
      "Low-VRAM local model for consumer GPUs such as an RTX 3050/3060. Follows the reference body pose; long videos are generated in ~5 s parts and crossfaded. Face Lock then re-applies your character's exact face to every frame. 480p native.",
    pricing: "Free · runs on your GPU",
    minDurationSec: 1,
    maxDurationSec: workerLimit("vace-1.3b", 60),
    nativeResolution: "480p @ 16 fps",
    capabilities: {
      identityElement: { supported: false, how: "Identity comes from the character image as a VACE reference image." },
      cameraMotion: { supported: false, how: "Only the body pose is transferred; the camera is static." },
      audio: { supported: true, how: "Reference audio is muxed back in by ffmpeg after generation." },
      seed: { supported: true, how: "Seed is passed to generate.py --base_seed." },
      nativeFps: { supported: false, how: "Generates at 16 fps; Motion Smoothing interpolates to the chosen rate." },
      motionStrength: NO_MOTION_STRENGTH,
      qualityControls: { supported: true, how: "Maximum: 40 sampling steps. Balanced: 25 steps (faster on small GPUs)." },
      promptGuidance: { supported: false, how: "A fixed identity-preservation prompt is used." },
    },
  },
  {
    id: "wan-2.2-animate-selfhosted",
    provider: "selfhosted",
    workerBackend: "animate-14b",
    label: "Wan 2.2 Animate 14B (your GPU · 48–80 GB)",
    endpoint: "worker:/v1/jobs (Wan2.2-Animate-14B)",
    description:
      "Highest-quality local model: full body, face and expression transfer with retargeting to your character's body shape. Needs a 48–80 GB GPU.",
    pricing: "Free · runs on your GPU",
    minDurationSec: 1,
    maxDurationSec: workerLimit("animate-14b", 60),
    nativeResolution: "720p @ 30 fps",
    capabilities: {
      identityElement: { supported: false, how: "Identity comes from the character image (Wan-Animate reference encoder)." },
      cameraMotion: { supported: false, how: "Keeps the character image framing; camera motion is not transferred." },
      audio: { supported: true, how: "Reference audio is muxed back in by ffmpeg after generation." },
      seed: { supported: true, how: "Seed is passed to generate.py --base_seed." },
      nativeFps: { supported: false, how: "Generates at 30 fps; other rates are converted by ffmpeg." },
      motionStrength: NO_MOTION_STRENGTH,
      qualityControls: { supported: true, how: "Maximum: refert_num=5 and 30 sampling steps. Balanced: refert_num=1 and 20 steps." },
      promptGuidance: { supported: false, how: "Wan-Animate uses its built-in prompt; detail and scene options have no effect." },
    },
  },
];

export function getModel(id: string): ModelSpec | undefined {
  return MODELS.find((m) => m.id === id);
}
