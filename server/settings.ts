import type { GenerationSettings } from "./providers/types.js";

export const DEFAULT_SETTINGS: GenerationSettings = {
  modelId: "wan-2.2-animate-selfhosted",
  resolution: "1080p",
  fps: 30,
  motionStrength: 100,
  identityLock: true,
  motionAccuracy: "maximum",
  handDetail: true,
  faceDetail: true,
  backgroundPreservation: true,
  cameraMotionTransfer: false,
  motionSmoothing: true,
  keepAudio: true,
  trimToModelLimit: false,
  autoRefine: true,
  scenePrompt: "",
  seed: null,
};

const BOOL_KEYS = [
  "identityLock",
  "handDetail",
  "faceDetail",
  "backgroundPreservation",
  "cameraMotionTransfer",
  "motionSmoothing",
  "keepAudio",
  "trimToModelLimit",
  "autoRefine",
] as const;

export function parseSettings(raw: unknown): GenerationSettings {
  const r = (typeof raw === "string" ? JSON.parse(raw) : raw ?? {}) as Partial<GenerationSettings>;
  const s = { ...DEFAULT_SETTINGS, ...r };
  if (!["720p", "1080p", "2k", "4k"].includes(s.resolution)) throw new Error("Invalid resolution");
  if (![24, 30, 50, 60].includes(Number(s.fps))) throw new Error("Invalid frame rate");
  s.fps = Number(s.fps) as GenerationSettings["fps"];
  s.motionStrength = Math.max(0, Math.min(100, Number(s.motionStrength) || 0));
  s.motionAccuracy = s.motionAccuracy === "balanced" ? "balanced" : "maximum";
  for (const k of BOOL_KEYS) s[k] = Boolean(s[k]);
  s.scenePrompt = String(s.scenePrompt ?? "").slice(0, 600);
  s.seed = s.seed === null || s.seed === undefined || (s.seed as unknown) === "" ? null : Math.floor(Number(s.seed)) || 0;
  return s;
}
