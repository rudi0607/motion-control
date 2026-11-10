import type { Job, Preflight } from "../server/jobs";
import type { Capability, GenerationSettings, ProviderId } from "../server/providers/types";
import type { QualityCheck } from "../server/quality";

export type { Job, GenerationSettings, QualityCheck, Preflight };

export interface ModelInfo {
  id: string;
  provider: ProviderId;
  label: string;
  endpoint: string;
  description: string;
  pricing: string;
  minDurationSec: number;
  maxDuration: { cameraOn: number; cameraOff: number };
  nativeResolution: string;
  available: boolean;
  capabilities: Record<string, Capability>;
}

export interface AppConfig {
  providers: Record<ProviderId, boolean>;
  defaults: GenerationSettings;
  maxUploadMB: number;
  models: ModelInfo[];
}

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error ?? `Request failed (${res.status})`);
  return body as T;
}

export const getConfig = () => fetch("/api/config").then((r) => json<AppConfig>(r));
export const listJobs = () => fetch("/api/jobs").then((r) => json<Job[]>(r));
export const cancelJob = (id: string) => fetch(`/api/jobs/${id}/cancel`, { method: "POST" }).then((r) => json<Job>(r));
export const regenerateJob = (id: string, settings?: Partial<GenerationSettings>) =>
  fetch(`/api/jobs/${id}/regenerate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings }),
  }).then((r) => json<Job>(r));
export const postClientChecks = (id: string, checks: QualityCheck[]) =>
  fetch(`/api/jobs/${id}/client-checks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checks }),
  }).then((r) => json<{ ok: boolean }>(r));

export const fileUrl = (job: Job, name: string, download = false) =>
  `/api/jobs/${job.id}/files/${name}${download ? "?download=1" : `?v=${job.updatedAt}`}`;

/** Multipart upload with progress (fetch has no upload progress events). */
export function createJob(form: FormData, onProgress: (f: number) => void): Promise<Job> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body: any = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body?.error ?? `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Network error while uploading"));
    xhr.send(form);
  });
}

export function subscribeJob(id: string, onJob: (j: Job) => void): () => void {
  const es = new EventSource(`/api/jobs/${id}/events`);
  es.onmessage = (e) => onJob(JSON.parse(e.data));
  return () => es.close();
}
