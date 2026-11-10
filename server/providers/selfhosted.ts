import { config } from "../config.js";
import type { GenerateRequest, ModelSpec, PollResult, ProviderAdapter } from "./types.js";

const base = () => {
  if (!config.selfhostedUrl) throw new Error("SELFHOSTED_WORKER_URL is not configured on the server.");
  return config.selfhostedUrl.replace(/\/+$/, "");
};
const authHeaders = (): Record<string, string> => ({ Authorization: `Bearer ${config.selfhostedToken ?? ""}` });

async function call(path: string, init: RequestInit = {}) {
  let res: Response;
  try {
    res = await fetch(`${base()}${path}`, { ...init, headers: { ...authHeaders(), ...(init.headers ?? {}) } });
  } catch (e) {
    throw new Error(`Self-hosted GPU worker unreachable at ${config.selfhostedUrl}: ${e instanceof Error ? e.message : e}`);
  }
  const text = await res.text();
  let body: any = text;
  try {
    body = JSON.parse(text);
  } catch {
    /* non-JSON */
  }
  if (!res.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? text).slice(0, 300);
    if (res.status === 401) throw new Error(`Self-hosted worker rejected the token (401). Check SELFHOSTED_WORKER_TOKEN.`);
    throw new Error(`Self-hosted worker error ${res.status}: ${detail}`);
  }
  return body;
}

/** Note: the URLs here are worker file ids returned by upload(). */
export function buildSelfhostedInput(model: ModelSpec, req: GenerateRequest): Record<string, unknown> {
  const s = req.settings;
  const max = s.motionAccuracy === "maximum";
  const vace = model.workerBackend === "vace-1.3b";
  const input: Record<string, unknown> = {
    image_file: req.imageUrl,
    video_file: req.videoUrl,
    backend: model.workerBackend,
    resolution: vace ? "480p" : "720p",
    refert_num: max ? 5 : 1,
    sample_steps: vace ? (max ? 40 : 25) : max ? 30 : 20,
    face_lock: s.identityLock,
    // Facial Detail Enhancement: partial GFPGAN blend keeps natural skin texture.
    face_restore: s.faceDetail ? 0.6 : 0,
  };
  if (s.seed !== null) input.seed = s.seed;
  return input;
}

export interface WorkerHealth {
  ok: boolean;
  problems: string[];
  gpus: string[];
  backend?: string;
  max_duration_sec?: number;
}

/** Last successful health response, used for the worker's real duration limit. */
export let lastHealth: WorkerHealth | null = null;

export async function selfhostedHealth(): Promise<WorkerHealth> {
  try {
    const h: WorkerHealth = await call("/v1/health", { signal: AbortSignal.timeout(5000) });
    lastHealth = h;
    return h;
  } catch (e) {
    return { ok: false, problems: [e instanceof Error ? e.message : String(e)], gpus: [] };
  }
}

export const selfhostedAdapter: ProviderAdapter = {
  id: "selfhosted",
  configured: () => Boolean(config.selfhostedUrl && config.selfhostedToken),

  async upload(data, filename, contentType) {
    const form = new FormData();
    form.append("file", new Blob([new Uint8Array(data)], { type: contentType }), filename);
    const body = await call("/v1/files", { method: "POST", body: form });
    return body.file_id as string;
  },

  async submit(model: ModelSpec, req) {
    const input = buildSelfhostedInput(model, req);
    const body = await call("/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
    return { requestId: body.id, sentInput: input };
  },

  async poll(model, requestId): Promise<PollResult> {
    const s = await call(`/v1/jobs/${requestId}`);
    const logs: string[] = s.logs ?? [];
    switch (s.status) {
      case "queued":
        return { state: "queued", queuePosition: s.queue_position ?? undefined, logs };
      case "preprocessing":
        return { state: "running", logs: [...logs, "Stage: pose extraction"] };
      case "generating":
        return { state: "running", logs: [...logs, `Stage: ${model.workerBackend === "vace-1.3b" ? "Wan2.1-VACE" : "Wan2.2-Animate"} diffusion`] };
      case "face_lock":
        return { state: "running", logs: [...logs, "Stage: face identity lock + restoration"] };
      case "succeeded":
        return {
          state: "succeeded",
          outputUrl: `${base()}/v1/jobs/${requestId}/output`,
          outputHeaders: authHeaders(),
          logs,
          meta: s.face_report ? { faceReport: s.face_report } : undefined,
        };
      default:
        return { state: "failed", error: s.error ?? `Worker job ${s.status}`, logs };
    }
  },

  async cancel(_model, requestId) {
    await call(`/v1/jobs/${requestId}/cancel`, { method: "POST" }).catch(() => undefined);
  },
};
