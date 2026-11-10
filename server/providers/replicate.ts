import { config } from "../config.js";
import type { GenerateRequest, ModelSpec, PollResult, ProviderAdapter } from "./types.js";

const API = "https://api.replicate.com/v1";

async function rep(path: string, init: RequestInit = {}) {
  if (!config.replicateToken) throw new Error("REPLICATE_API_TOKEN is not configured on the server.");
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: { Authorization: `Bearer ${config.replicateToken}`, ...(init.headers ?? {}) },
  });
  const text = await res.text();
  let body: any = text;
  try {
    body = JSON.parse(text);
  } catch {
    /* non-JSON error body */
  }
  if (!res.ok) {
    const detail = body?.detail ?? body?.title ?? text.slice(0, 300);
    if (res.status === 401) throw new Error(`Replicate authentication failed (401). Check REPLICATE_API_TOKEN. ${detail}`);
    if (res.status === 402) throw new Error(`Replicate account requires billing/credit (402). ${detail}`);
    if (res.status === 422) throw new Error(`Replicate rejected the input (422): ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
    throw new Error(`Replicate error ${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return body;
}

export function buildReplicateInput(req: GenerateRequest): Record<string, unknown> {
  const s = req.settings;
  const max = s.motionAccuracy === "maximum";
  const input: Record<string, unknown> = {
    video: req.videoUrl,
    character_image: req.imageUrl,
    resolution: "720",
    // Generated natively at the requested rate (model allows 5–60).
    frames_per_second: Math.min(60, Math.max(5, s.fps)),
    // 5 temporal guidance frames gives better long-range consistency than 1.
    refert_num: max ? 5 : 1,
    go_fast: !max,
    merge_audio: s.keepAudio,
  };
  if (s.seed !== null) input.seed = s.seed;
  return input;
}

export const replicateAdapter: ProviderAdapter = {
  id: "replicate",
  configured: () => Boolean(config.replicateToken),

  async upload(data, filename, contentType) {
    const form = new FormData();
    form.append("content", new Blob([new Uint8Array(data)], { type: contentType }), filename);
    const body = await rep("/files", { method: "POST", body: form });
    const url = body?.urls?.get;
    if (!url) throw new Error("Replicate file upload returned no URL");
    return url;
  },

  async submit(model, req) {
    const input = buildReplicateInput(req);
    const body = await rep(`/models/${model.endpoint}/predictions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input }),
    });
    return { requestId: body.id, sentInput: input };
  },

  async poll(_model, requestId): Promise<PollResult> {
    const p = await rep(`/predictions/${requestId}`);
    const logs = typeof p.logs === "string" ? p.logs.split("\n").filter(Boolean).slice(-200) : [];
    switch (p.status) {
      case "starting":
        return { state: "queued", logs };
      case "processing":
        return { state: "running", logs };
      case "succeeded": {
        const url = Array.isArray(p.output) ? p.output[0] : p.output;
        if (typeof url !== "string") return { state: "failed", error: "Model finished without a video output", logs };
        return { state: "succeeded", outputUrl: url, logs, meta: { predictTime: p.metrics?.predict_time } };
      }
      case "canceled":
        return { state: "failed", error: "Prediction was canceled", logs };
      default:
        return { state: "failed", error: p.error ? String(p.error) : `Prediction ${p.status}`, logs };
    }
  },

  async cancel(_model, requestId) {
    await rep(`/predictions/${requestId}/cancel`, { method: "POST" }).catch(() => undefined);
  },
};
