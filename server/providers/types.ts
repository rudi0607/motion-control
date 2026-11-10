export type ProviderId = "fal" | "replicate" | "selfhosted";

export type Resolution = "720p" | "1080p" | "2k" | "4k";
export type Fps = 24 | 30 | 50 | 60;

export interface GenerationSettings {
  modelId: string;
  resolution: Resolution;
  fps: Fps;
  /** 0–100. No connected model API exposes this; stored for transparency only. */
  motionStrength: number;
  identityLock: boolean;
  motionAccuracy: "maximum" | "balanced";
  handDetail: boolean;
  faceDetail: boolean;
  backgroundPreservation: boolean;
  cameraMotionTransfer: boolean;
  motionSmoothing: boolean;
  keepAudio: boolean;
  trimToModelLimit: boolean;
  autoRefine: boolean;
  scenePrompt: string;
  seed: number | null;
}

export interface Capability {
  supported: boolean;
  how: string;
}

export interface ModelSpec {
  id: string;
  provider: ProviderId;
  label: string;
  endpoint: string;
  description: string;
  pricing: string;
  minDurationSec: number;
  /** Maximum reference duration given the chosen settings. */
  maxDurationSec: (s: GenerationSettings) => number;
  nativeResolution: string;
  capabilities: {
    identityElement: Capability;
    cameraMotion: Capability;
    audio: Capability;
    seed: Capability;
    nativeFps: Capability;
    motionStrength: Capability;
    qualityControls: Capability;
    promptGuidance: Capability;
  };
}

export interface GenerateRequest {
  imageUrl: string;
  videoUrl: string;
  /** Extra angles of the character for identity binding. */
  identityImageUrls: string[];
  prompt: string;
  settings: GenerationSettings;
  referenceFps: number;
}

export interface SubmitResult {
  requestId: string;
  /** Exact parameters sent to the provider (for transparency in the UI). */
  sentInput: Record<string, unknown>;
}

export type PollResult =
  | { state: "queued"; queuePosition?: number; logs: string[] }
  | { state: "running"; logs: string[] }
  | { state: "succeeded"; outputUrl: string; outputHeaders?: Record<string, string>; logs: string[]; meta?: Record<string, unknown> }
  | { state: "failed"; error: string; logs: string[] };

export interface ProviderAdapter {
  id: ProviderId;
  configured(): boolean;
  upload(data: Buffer, filename: string, contentType: string): Promise<string>;
  submit(model: ModelSpec, req: GenerateRequest): Promise<SubmitResult>;
  poll(model: ModelSpec, requestId: string): Promise<PollResult>;
  cancel(model: ModelSpec, requestId: string): Promise<void>;
}
