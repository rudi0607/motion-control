import { createFalClient, ApiError, ValidationError } from "@fal-ai/client";
import { config } from "../config.js";
import { buildPrompt } from "../prompt.js";
import type { GenerateRequest, ModelSpec, PollResult, ProviderAdapter, SubmitResult } from "./types.js";

let client: ReturnType<typeof createFalClient> | null = null;
function fal() {
  if (!config.falKey) throw new Error("FAL_KEY is not configured on the server.");
  client ??= createFalClient({ credentials: config.falKey });
  return client;
}

export function describeFalError(err: unknown): string {
  if (err instanceof ValidationError) {
    const fields = err.fieldErrors?.map((f) => `${f.loc?.slice(1).join(".") || "input"}: ${f.msg}`).join("; ");
    return `fal.ai rejected the input${fields ? ` — ${fields}` : `: ${err.message}`}`;
  }
  if (err instanceof ApiError) {
    const body: any = err.body;
    const detail = typeof body?.detail === "string" ? body.detail : body?.detail ? JSON.stringify(body.detail) : err.message;
    if (err.status === 401 || err.status === 403) return `fal.ai authentication failed (${err.status}). Check FAL_KEY. ${detail}`;
    if (err.status === 402) return `fal.ai account has insufficient balance (402). ${detail}`;
    return `fal.ai error ${err.status}: ${detail}`;
  }
  return err instanceof Error ? err.message : String(err);
}

function isKling(model: ModelSpec) {
  return model.endpoint.includes("kling-video");
}

export function buildFalInput(model: ModelSpec, req: GenerateRequest): Record<string, unknown> {
  const s = req.settings;
  if (isKling(model)) {
    const orientation = s.cameraMotionTransfer ? "image" : "video";
    // Kling only allows element binding with orientation "video".
    const useElement = s.identityLock && orientation === "video";
    const input: Record<string, unknown> = {
      image_url: req.imageUrl,
      video_url: req.videoUrl,
      character_orientation: orientation,
      keep_original_sound: s.keepAudio,
      prompt: buildPrompt(s, { elementRef: useElement ? "@Element1" : undefined }),
    };
    if (useElement) {
      const [frontal, ...rest] = req.identityImageUrls.length ? req.identityImageUrls : [req.imageUrl];
      const refs = [...rest, req.imageUrl].filter((u, i, a) => a.indexOf(u) === i).slice(0, 3);
      input.elements = [{ frontal_image_url: frontal, reference_image_urls: refs }];
    }
    return input;
  }
  // Wan 2.2 Animate Move
  const max = s.motionAccuracy === "maximum";
  const input: Record<string, unknown> = {
    image_url: req.imageUrl,
    video_url: req.videoUrl,
    resolution: "720p",
    num_inference_steps: max ? 30 : 20,
    video_quality: max ? "maximum" : "high",
    video_write_mode: "balanced",
  };
  if (s.seed !== null) input.seed = s.seed;
  return input;
}

export const falAdapter: ProviderAdapter = {
  id: "fal",
  configured: () => Boolean(config.falKey),

  async upload(data, filename, contentType) {
    try {
      const blob = new Blob([new Uint8Array(data)], { type: contentType });
      return await fal().storage.upload(new File([blob], filename, { type: contentType }));
    } catch (e) {
      throw new Error(`Upload to fal.ai storage failed: ${describeFalError(e)}`);
    }
  },

  async submit(model, req): Promise<SubmitResult> {
    const input = buildFalInput(model, req);
    try {
      const r = await fal().queue.submit(model.endpoint as any, { input: input as any });
      return { requestId: r.request_id, sentInput: input };
    } catch (e) {
      throw new Error(describeFalError(e));
    }
  },

  async poll(model, requestId): Promise<PollResult> {
    let status;
    try {
      status = await fal().queue.status(model.endpoint, { requestId, logs: true });
    } catch (e) {
      // The status endpoint reports errored requests as HTTP errors.
      return { state: "failed", error: describeFalError(e), logs: [] };
    }
    const logs = ((status as any).logs ?? []).map((l: any) => l.message).filter(Boolean);
    if (status.status === "IN_QUEUE") return { state: "queued", queuePosition: status.queue_position, logs };
    if (status.status === "IN_PROGRESS") return { state: "running", logs };
    try {
      const result = await fal().queue.result(model.endpoint as any, { requestId });
      const data: any = result.data;
      const url = data?.video?.url;
      if (!url) return { state: "failed", error: `Model finished without a video: ${JSON.stringify(data).slice(0, 400)}`, logs };
      return { state: "succeeded", outputUrl: url, logs, meta: { seed: data.seed, modelPrompt: data.prompt } };
    } catch (e) {
      return { state: "failed", error: describeFalError(e), logs };
    }
  },

  async cancel(model, requestId) {
    await fal().queue.cancel(model.endpoint, { requestId }).catch(() => undefined);
  },
};
