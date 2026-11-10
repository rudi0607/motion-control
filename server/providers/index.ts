import { falAdapter } from "./fal.js";
import { replicateAdapter } from "./replicate.js";
import { selfhostedAdapter } from "./selfhosted.js";
import type { ModelSpec, ProviderAdapter, ProviderId } from "./types.js";

export const adapters: Record<ProviderId, ProviderAdapter> = { fal: falAdapter, replicate: replicateAdapter, selfhosted: selfhostedAdapter };
export const REQUIRED_ENV: Record<ProviderId, string> = {
  fal: "FAL_KEY",
  replicate: "REPLICATE_API_TOKEN",
  selfhosted: "SELFHOSTED_WORKER_URL + SELFHOSTED_WORKER_TOKEN",
};
export const PROVIDER_NAME: Record<ProviderId, string> = { fal: "fal.ai", replicate: "Replicate", selfhosted: "your self-hosted worker" };

const NO_MOTION_STRENGTH = {
  supported: false,
  how: "Not exposed by this model's API — the output always follows the full reference motion.",
};

const kling = (tier: "pro" | "standard"): ModelSpec => ({
  id: `kling-v3-${tier}`,
  provider: "fal",
  label: `Kling v3 ${tier === "pro" ? "Pro" : "Standard"} Motion Control`,
  endpoint: `fal-ai/kling-video/v3/${tier}/motion-control`,
  description:
    tier === "pro"
      ? "Highest quality. Supports facial element binding for stronger identity preservation, expressions and lip movement from the reference."
      : "Cost-effective Kling motion transfer with facial element binding.",
  pricing: tier === "pro" ? "Per-second, see fal.ai pricing" : "≈ $0.126 / second of output (fal.ai)",
  minDurationSec: 3,
  maxDurationSec: (s) => (s.cameraMotionTransfer ? 10 : 30),
  nativeResolution: tier === "pro" ? "Up to 1080p (provider-decided)" : "720p (provider-decided)",
  capabilities: {
    identityElement: {
      supported: true,
      how: "Binds a facial element (face crop + reference views) as @Element1. Only available when Camera Motion Transfer is off (orientation = video).",
    },
    cameraMotion: {
      supported: true,
      how: "On → character_orientation=image (follows camera moves, max 10 s). Off → orientation=video (complex body motion, max 30 s).",
    },
    audio: { supported: true, how: "keep_original_sound keeps the reference audio track." },
    seed: { supported: false, how: "Seed not exposed; regenerate for a new sample." },
    nativeFps: { supported: false, how: "Model decides frame rate; conversion done by ffmpeg." },
    motionStrength: NO_MOTION_STRENGTH,
    qualityControls: { supported: false, how: "Detail options are applied as prompt guidance." },
    promptGuidance: { supported: true, how: "Identity, detail and scene options are sent as prompt guidance." },
  },
});

export const MODELS: ModelSpec[] = [
  {
    id: "wan-2.2-animate-selfhosted",
    provider: "selfhosted",
    label: "Wan 2.2 Animate 14B (self-hosted GPU)",
    endpoint: "worker:/v1/jobs (Wan2.2-Animate-14B)",
    description:
      "Open-source Wan2.2-Animate running on your own GPU worker (see worker/). No third-party API, and your images and videos never leave your infrastructure.",
    pricing: "Your GPU cost only",
    minDurationSec: 1,
    maxDurationSec: () => Number(process.env.SELFHOSTED_MAX_DURATION_SEC ?? 30),
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
  kling("pro"),
  kling("standard"),
  {
    id: "wan-2.2-animate-fal",
    provider: "fal",
    label: "Wan 2.2 Animate Move (fal.ai)",
    endpoint: "fal-ai/wan/v2.2-14b/animate/move",
    description: "Open-weight Wan-Animate model: body + facial motion from the video, appearance from the image. Seeded and reproducible.",
    pricing: "Per-second, see fal.ai pricing",
    minDurationSec: 1,
    maxDurationSec: () => 30,
    nativeResolution: "720p",
    capabilities: {
      identityElement: { supported: false, how: "Identity comes from the character image only (no element binding)." },
      cameraMotion: { supported: false, how: "Keeps the character image framing; camera motion is not transferred." },
      audio: { supported: true, how: "Reference audio is muxed back in by ffmpeg after generation." },
      seed: { supported: true, how: "Seed is passed through for reproducible results." },
      nativeFps: { supported: false, how: "Frame rate converted by ffmpeg." },
      motionStrength: NO_MOTION_STRENGTH,
      qualityControls: { supported: true, how: "Motion Accuracy = Maximum uses 30 inference steps and maximum video quality." },
      promptGuidance: { supported: false, how: "This endpoint takes no text prompt; detail and scene options have no effect." },
    },
  },
  {
    id: "wan-2.2-animate-replicate",
    provider: "replicate",
    label: "Wan 2.2 Animate (Replicate)",
    endpoint: "wan-video/wan-2.2-animate-animation",
    description: "Wan-Animate on Replicate. Generates natively at the selected frame rate and can merge the reference audio.",
    pricing: "Per-second GPU time, see Replicate pricing",
    minDurationSec: 1,
    maxDurationSec: () => 30,
    nativeResolution: "720p",
    capabilities: {
      identityElement: { supported: false, how: "Identity comes from the character image only." },
      cameraMotion: { supported: false, how: "Camera motion is not transferred." },
      audio: { supported: true, how: "merge_audio keeps the reference audio." },
      seed: { supported: true, how: "Seed is passed through." },
      nativeFps: { supported: true, how: "frames_per_second generates at the chosen rate (5–60)." },
      motionStrength: NO_MOTION_STRENGTH,
      qualityControls: { supported: true, how: "Maximum accuracy uses refert_num=5 and disables go_fast." },
      promptGuidance: { supported: false, how: "This model takes no text prompt; detail and scene options have no effect." },
    },
  },
];

export function getModel(id: string): ModelSpec | undefined {
  return MODELS.find((m) => m.id === id);
}
