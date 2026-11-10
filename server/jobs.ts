import { EventEmitter } from "node:events";
import fs from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { config } from "./config.js";
import {
  type MediaInfo,
  RESOLUTION_SHORT_EDGE,
  extractAudio,
  frameStats,
  freezeIntervals,
  prepareImage,
  prepareReferenceVideo,
  probe,
  renderFinal,
  run,
} from "./media.js";
import { PROVIDER_NAME, REQUIRED_ENV, adapters, getModel } from "./providers/index.js";
import { selfhostedHealth } from "./providers/selfhosted.js";
import type { GenerationSettings, ModelSpec } from "./providers/types.js";
import { type QualityCheck, runServerChecks } from "./quality.js";

export type StageStatus = "pending" | "running" | "done" | "failed" | "skipped";
export interface Stage {
  id: string;
  label: string;
  status: StageStatus;
  detail?: string;
  startedAt?: number;
  endedAt?: number;
}

/** Summary of the in-browser MediaPipe analysis that runs before submission. */
export interface Preflight {
  image?: {
    faceDetected: boolean;
    /** Normalised [0..1] face bounding box in the character image. */
    faceBox?: { x: number; y: number; w: number; h: number };
    poseDetected: boolean;
    personAreaFraction?: number;
  };
  video?: {
    framesAnalyzed: number;
    poseDetectionRate: number;
    faceDetectionRate: number;
    handDetectionRate: number;
    fullBodyRate: number;
    blinkCount: number;
  };
}

export interface Job {
  id: string;
  parentJobId?: string;
  createdAt: number;
  updatedAt: number;
  status: "running" | "succeeded" | "failed" | "canceled";
  modelId: string;
  modelLabel: string;
  settings: GenerationSettings;
  preflight: Preflight;
  inputs: {
    imageFile: string;
    imageInfo: MediaInfo;
    videoFile: string;
    videoInfo: MediaInfo;
    identityFiles: string[];
    trimmedToSec?: number;
  };
  provider: {
    id: string;
    endpoint: string;
    requestId?: string;
    queuePosition?: number;
    state?: string;
    logs: string[];
    sentInput?: Record<string, unknown>;
    meta?: Record<string, unknown>;
    submittedAt?: number;
  };
  stages: Stage[];
  warnings: string[];
  notes: string[];
  quality?: {
    checks: QualityCheck[];
    initialChecks?: QualityCheck[];
    refinement?: string;
    clientChecks?: QualityCheck[];
  };
  output?: { file: string; info: MediaInfo; rawFile: string; rawInfo: MediaInfo };
  error?: string;
}

const STAGES: [string, string][] = [
  ["analyze_reference", "Analyze reference video"],
  ["extract_motion", "Extract pose, face & hand landmarks"],
  ["identity_reference", "Lock character identity reference"],
  ["upload", "Upload inputs to model provider"],
  ["generate", "Generate motion transfer"],
  ["temporal", "Temporal consistency & smoothing"],
  ["quality", "Automated quality checks"],
  ["refine", "Refine problematic frames"],
  ["render", "Render final MP4"],
];

const jobs = new Map<string, Job>();
export const events = new EventEmitter();
events.setMaxListeners(1000);
const running = new Map<string, { cancel: boolean }>();

export const jobDir = (id: string) => path.join(config.dataDir, "jobs", id);
export const getJob = (id: string) => jobs.get(id);
export const listJobs = () => [...jobs.values()].sort((a, b) => b.createdAt - a.createdAt);

async function save(job: Job) {
  job.updatedAt = Date.now();
  events.emit(job.id, job);
  await fs.writeFile(path.join(jobDir(job.id), "job.json"), JSON.stringify(job, null, 2));
}

function stage(job: Job, id: string) {
  const s = job.stages.find((x) => x.id === id);
  if (!s) throw new Error(`unknown stage ${id}`);
  return s;
}

async function setStage(job: Job, id: string, status: StageStatus, detail?: string) {
  const s = stage(job, id);
  s.status = status;
  if (detail !== undefined) s.detail = detail;
  if (status === "running") s.startedAt = Date.now();
  if (status === "done" || status === "failed" || status === "skipped") s.endedAt = Date.now();
  await save(job);
}

export interface ValidationResult {
  errors: string[];
  warnings: string[];
  maxDurationSec: number;
}

export function validateInputs(model: ModelSpec, s: GenerationSettings, image: MediaInfo, video: MediaInfo): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const maxDurationSec = model.maxDurationSec(s);
  if (Math.min(image.width, image.height) < 300) errors.push(`Character image is too small (${image.width}×${image.height}); use at least 300 px on the short edge.`);
  const ar = image.width / image.height;
  if (ar > 2.5 || ar < 0.4) errors.push(`Character image aspect ratio ${ar.toFixed(2)} is outside the supported 2:5 – 5:2 range.`);
  if (video.durationSec <= 0) errors.push("Could not determine the reference video duration.");
  if (video.durationSec > maxDurationSec + 0.05 && !s.trimToModelLimit)
    errors.push(
      `Reference video is ${video.durationSec.toFixed(1)} s but ${model.label} accepts at most ${maxDurationSec} s with these settings. Enable "Trim to model limit" or shorten the video.`,
    );
  if (video.durationSec > maxDurationSec + 0.05 && s.trimToModelLimit)
    warnings.push(`Reference will be trimmed to the first ${maxDurationSec} s (model limit); the output cannot match the full ${video.durationSec.toFixed(1)} s.`);
  if (video.durationSec < model.minDurationSec) warnings.push(`Reference is shorter than ${model.minDurationSec} s; ${model.label} may reject it.`);
  if (Math.min(video.width, video.height) < 360) warnings.push(`Reference video is low resolution (${video.width}×${video.height}); pose extraction may be less accurate.`);
  if (s.motionStrength !== 100) warnings.push("Motion Strength is not exposed by any connected model API; the full reference motion will be used.");
  if (!model.capabilities.cameraMotion.supported && s.cameraMotionTransfer) warnings.push(`${model.label} does not transfer camera motion.`);
  const shortEdge = RESOLUTION_SHORT_EDGE[s.resolution];
  if (shortEdge > 1080) warnings.push(`${s.resolution.toUpperCase()} is produced by Lanczos upscaling of the model's native output, not native generation.`);
  return { errors, warnings, maxDurationSec };
}

export interface CreateJobInput {
  model: ModelSpec;
  settings: GenerationSettings;
  preflight: Preflight;
  imagePath: string;
  videoPath: string;
  identityPaths: string[];
  parentJobId?: string;
}

/** Probe + validate uploaded files. Throws with a user-facing message on invalid input. */
export async function probeInputs(imagePath: string, videoPath: string) {
  const [imageInfo, videoInfo] = await Promise.all([
    probe(imagePath).catch((e) => {
      throw new Error(`Character image: ${e.message}`);
    }),
    probe(videoPath).catch((e) => {
      throw new Error(`Reference video: ${e.message}`);
    }),
  ]);
  return { imageInfo, videoInfo };
}

export async function createJob(input: CreateJobInput): Promise<Job> {
  const adapter = adapters[input.model.provider];
  if (!adapter.configured())
    throw new Error(`${REQUIRED_ENV[input.model.provider]} is not set on the server, so ${input.model.label} cannot run. Add it to the environment and restart.`);
  const health = await selfhostedHealth();
  if (!health.ok) throw new Error(`GPU worker is not ready: ${health.problems.join(" · ")}`);
  if (health.backend && health.backend !== input.model.workerBackend)
    throw new Error(`Your GPU worker runs ${health.backend}; select the matching model (this one needs WAN_BACKEND=${input.model.workerBackend}).`);
  const { imageInfo, videoInfo } = await probeInputs(input.imagePath, input.videoPath);
  const v = validateInputs(input.model, input.settings, imageInfo, videoInfo);
  if (v.errors.length) throw new Error(v.errors.join(" "));

  const id = randomUUID();
  const dir = jobDir(id);
  await fs.mkdir(dir, { recursive: true });
  const move = async (from: string, name: string) => {
    const to = path.join(dir, name);
    await fs.rename(from, to).catch(async () => {
      await fs.copyFile(from, to);
      await fs.rm(from, { force: true });
    });
    return name;
  };
  const imageFile = await move(input.imagePath, `source-image${path.extname(input.imagePath) || ".img"}`);
  const videoFile = await move(input.videoPath, `source-video${path.extname(input.videoPath) || ".vid"}`);
  const identityFiles: string[] = [];
  for (const [i, p] of input.identityPaths.entries()) identityFiles.push(await move(p, `identity-src-${i}${path.extname(p) || ".img"}`));

  const job: Job = {
    id,
    parentJobId: input.parentJobId,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    status: "running",
    modelId: input.model.id,
    modelLabel: input.model.label,
    settings: input.settings,
    preflight: input.preflight,
    inputs: { imageFile, imageInfo, videoFile, videoInfo, identityFiles },
    provider: { id: input.model.provider, endpoint: input.model.endpoint, logs: [] },
    stages: STAGES.map(([sid, label]) => ({ id: sid, label, status: "pending" })),
    warnings: v.warnings,
    notes: [],
  };
  jobs.set(id, job);
  await save(job);
  void runPipeline(job, input.model);
  return job;
}

/** Duplicate a finished job's inputs into a new job (new sample / new seed). */
export async function regenerate(parent: Job, settingsPatch: Partial<GenerationSettings>): Promise<Job> {
  const model = getModel(parent.modelId);
  if (!model) throw new Error("Model no longer available");
  const tmp = path.join(config.dataDir, "uploads");
  await fs.mkdir(tmp, { recursive: true });
  const copy = async (name: string) => {
    const to = path.join(tmp, `${randomUUID()}${path.extname(name)}`);
    await fs.copyFile(path.join(jobDir(parent.id), name), to);
    return to;
  };
  const settings = { ...parent.settings, ...settingsPatch };
  if (model.capabilities.seed.supported && settingsPatch.seed === undefined) settings.seed = Math.floor(Math.random() * 2 ** 31);
  return createJob({
    model,
    settings,
    preflight: parent.preflight,
    imagePath: await copy(parent.inputs.imageFile),
    videoPath: await copy(parent.inputs.videoFile),
    identityPaths: await Promise.all(parent.inputs.identityFiles.filter((f) => f.startsWith("identity-src-")).map(copy)),
    parentJobId: parent.id,
  });
}

export async function cancelJob(job: Job) {
  const ctl = running.get(job.id);
  if (ctl) ctl.cancel = true;
  const model = getModel(job.modelId);
  if (model && job.provider.requestId) await adapters[model.provider].cancel(model, job.provider.requestId);
  if (job.status === "running") {
    job.status = "canceled";
    job.error = "Canceled by user.";
    for (const s of job.stages) if (s.status === "running" || s.status === "pending") s.status = "skipped";
    await save(job);
  }
}

class Canceled extends Error {}

async function download(url: string, to: string, headers?: Record<string, string>) {
  const res = await fetch(url, { headers });
  if (!res.ok || !res.body) throw new Error(`Downloading model output failed: HTTP ${res.status}`);
  await fs.writeFile(to, Buffer.from(await res.arrayBuffer()));
}

async function cropFace(src: string, out: string, img: MediaInfo, box: { x: number; y: number; w: number; h: number }) {
  // Expand the detected face box to include hair/jaw, square it, clamp to the image.
  const cx = (box.x + box.w / 2) * img.width;
  const cy = (box.y + box.h / 2) * img.height;
  const side = Math.min(Math.max(box.w * img.width, box.h * img.height) * 1.9, img.width, img.height);
  const x = Math.round(Math.min(Math.max(cx - side / 2, 0), img.width - side));
  const y = Math.round(Math.min(Math.max(cy - side / 2, 0), img.height - side));
  const s = Math.floor(side / 2) * 2;
  const r = await run("ffmpeg", ["-hide_banner", "-nostdin", "-y", "-i", src, "-vf", `crop=${s}:${s}:${x}:${y},scale='max(512,iw)':-2`, "-frames:v", "1", "-update", "1", out]);
  if (r.code !== 0) throw new Error("Face crop failed");
}

async function runPipeline(job: Job, model: ModelSpec) {
  const ctl = { cancel: false };
  running.set(job.id, ctl);
  const dir = jobDir(job.id);
  const p = (n: string) => path.join(dir, n);
  const check = () => {
    if (ctl.cancel) throw new Canceled();
  };
  const adapter = adapters[model.provider];
  const s = job.settings;
  let current = "analyze_reference";
  try {
    // 1. Reference video analysis + normalisation.
    await setStage(job, current, "running");
    const max = model.maxDurationSec(s);
    const trimTo = job.inputs.videoInfo.durationSec > max + 0.05 ? max : undefined;
    await prepareReferenceVideo(p(job.inputs.videoFile), p("reference.mp4"), { maxDurationSec: trimTo });
    const refInfo = await probe(p("reference.mp4"));
    job.inputs.trimmedToSec = trimTo;
    const vi = job.inputs.videoInfo;
    await setStage(
      job,
      current,
      "done",
      `${vi.width}×${vi.height}, ${vi.fps.toFixed(2)} fps, ${vi.durationSec.toFixed(2)} s, ${vi.hasAudio ? "with" : "no"} audio` +
        (trimTo ? ` — trimmed to ${trimTo} s` : "") +
        ` → normalised H.264 ${refInfo.width}×${refInfo.height}`,
    );
    check();

    // 2. Landmark extraction happened in the browser (MediaPipe) before submission.
    current = "extract_motion";
    const pv = job.preflight.video;
    if (pv) {
      await setStage(
        job,
        current,
        "done",
        `${pv.framesAnalyzed} frames: body ${(pv.poseDetectionRate * 100).toFixed(0)}%, face ${(pv.faceDetectionRate * 100).toFixed(0)}%, hands ${(pv.handDetectionRate * 100).toFixed(0)}%, full body ${(pv.fullBodyRate * 100).toFixed(0)}%, ${pv.blinkCount} blink(s). The model also performs its own internal motion extraction.`,
      );
      if (pv.poseDetectionRate < 0.6) job.warnings.push(`A person was detected in only ${(pv.poseDetectionRate * 100).toFixed(0)}% of reference frames; motion transfer may be unreliable in the remaining frames.`);
    } else {
      await setStage(job, current, "skipped", "In-browser landmark analysis was not provided; the model performs its own motion extraction.");
    }

    // 3. Identity reference.
    current = "identity_reference";
    await setStage(job, current, "running");
    await prepareImage(p(job.inputs.imageFile), p("character.png"));
    const identity: string[] = [];
    const pi = job.preflight.image;
    const useElement = model.capabilities.identityElement.supported && s.identityLock && !s.cameraMotionTransfer;
    if (useElement && pi?.faceBox) {
      await cropFace(p("character.png"), p("identity-face.png"), job.inputs.imageInfo, pi.faceBox);
      identity.push("identity-face.png");
    }
    for (const [i, f] of job.inputs.identityFiles.entries()) {
      await prepareImage(p(f), p(`identity-angle-${i}.png`));
      identity.push(`identity-angle-${i}.png`);
    }
    const idDetail = [
      `Character image ${job.inputs.imageInfo.width}×${job.inputs.imageInfo.height}`,
      pi ? (pi.faceDetected ? "face detected" : "no face detected") : "face not analysed",
      useElement
        ? identity.length
          ? `facial element bound from ${identity.length} image(s)`
          : "facial element bound from the character image"
        : s.identityLock
          ? "identity enforced via character image + prompt (no element binding for this model/setting)"
          : "identity lock off",
    ].join(" · ");
    if (pi && !pi.faceDetected) job.warnings.push("No face was detected in the character image; identity preservation cannot be verified automatically.");
    await setStage(job, current, "done", idDetail);
    check();

    // 4. Upload.
    current = "upload";
    await setStage(job, current, "running");
    const up = async (name: string, type: string) => adapter.upload(await fs.readFile(p(name)), name, type);
    const [imageUrl, videoUrl] = await Promise.all([up("character.png", "image/png"), up("reference.mp4", "video/mp4")]);
    const identityImageUrls = useElement ? await Promise.all(identity.map((f) => up(f, "image/png"))) : [];
    await setStage(job, current, "done", `Uploaded ${2 + identityImageUrls.length} file(s) to ${PROVIDER_NAME[model.provider]}`);
    check();

    // 5. Generate.
    current = "generate";
    await setStage(job, current, "running", "Submitting…");
    const sub = await adapter.submit(model, { imageUrl, videoUrl, identityImageUrls, prompt: "", settings: s, referenceFps: refInfo.fps });
    job.provider.requestId = sub.requestId;
    job.provider.sentInput = sub.sentInput;
    job.provider.submittedAt = Date.now();
    await save(job);
    await pollUntilDone(job, model, ctl);
    await continueAfterGeneration(job, model, ctl);
  } catch (e) {
    await fail(job, current, e);
  } finally {
    running.delete(job.id);
  }
}

async function fail(job: Job, current: string, e: unknown) {
  if (e instanceof Canceled || job.status === "canceled") return;
  const msg = e instanceof Error ? e.message : String(e);
  const st = job.stages.find((x) => x.id === current);
  if (st) {
    st.status = "failed";
    st.detail = msg;
    st.endedAt = Date.now();
  }
  for (const x of job.stages) if (x.status === "pending") x.status = "skipped";
  job.status = "failed";
  job.error = msg;
  await save(job);
}

async function pollUntilDone(job: Job, model: ModelSpec, ctl: { cancel: boolean }) {
  const adapter = adapters[model.provider];
  let delay = 2000;
  let transient = 0;
  for (;;) {
    if (ctl.cancel) throw new Canceled();
    let r;
    try {
      r = await adapter.poll(model, job.provider.requestId!);
      transient = 0;
    } catch (e) {
      // Network blips while polling should not lose a paid generation.
      if (++transient > 10) throw e;
      await new Promise((res) => setTimeout(res, 5000));
      continue;
    }
    job.provider.state = r.state;
    job.provider.logs = r.logs.slice(-200);
    const st = stage(job, "generate");
    if (r.state === "queued") {
      job.provider.queuePosition = r.queuePosition;
      st.detail = r.queuePosition !== undefined ? `Queued at provider (position ${r.queuePosition})` : "Queued at provider";
    } else if (r.state === "running") {
      job.provider.queuePosition = undefined;
      const elapsed = Math.round((Date.now() - (job.provider.submittedAt ?? Date.now())) / 1000);
      st.detail = `Generating on ${PROVIDER_NAME[model.provider]} GPU (${elapsed} s elapsed)${r.logs.length ? ` — ${r.logs[r.logs.length - 1]}` : ""}`;
    } else if (r.state === "failed") {
      throw new Error(`Generation failed: ${r.error}`);
    } else {
      job.provider.meta = r.meta;
      st.detail = "Downloading result…";
      await save(job);
      await download(r.outputUrl, path.join(jobDir(job.id), "raw.mp4"), r.outputHeaders);
      return;
    }
    await save(job);
    await new Promise((res) => setTimeout(res, delay));
    delay = Math.min(delay * 1.25, 8000);
  }
}

async function continueAfterGeneration(job: Job, model: ModelSpec, ctl: { cancel: boolean }) {
  const dir = jobDir(job.id);
  const p = (n: string) => path.join(dir, n);
  const s = job.settings;
  let current = "generate";
  try {
    const rawInfo = await probe(p("raw.mp4"));
    await setStage(job, current, "done", `Model returned ${rawInfo.width}×${rawInfo.height} @ ${rawInfo.fps.toFixed(2)} fps, ${rawInfo.durationSec.toFixed(2)} s${rawInfo.hasAudio ? " with audio" : ""}`);
    if (ctl.cancel) throw new Canceled();

    const refInfo = await probe(p("reference.mp4"));
    const wantAudio = s.keepAudio && refInfo.hasAudio;
    let audioFile: string | undefined;
    // Re-mux the reference audio when the model dropped it so lips and sound stay aligned to the reference timeline.
    if (wantAudio && !rawInfo.hasAudio) {
      await extractAudio(p("reference.mp4"), p("reference-audio.m4a"));
      audioFile = p("reference-audio.m4a");
    }
    const renderOpts = {
      targetShortEdge: RESOLUTION_SHORT_EDGE[s.resolution],
      fps: s.fps,
      motionInterpolation: s.motionSmoothing,
      deflicker: false,
      durationSec: refInfo.durationSec,
      audioFile,
      keepProviderAudio: wantAudio,
    };

    current = "temporal";
    await setStage(job, current, "running", "Rendering with ffmpeg…");
    const notes = await renderFinal(p("raw.mp4"), p("output.mp4"), rawInfo, renderOpts);
    if (audioFile) notes.push("Reference audio muxed onto the generated video.");
    job.notes.push(...notes);
    await setStage(job, current, "done", notes.length ? notes.join(" ") : "Output already matched the requested format; re-encoded to H.264.");

    current = "quality";
    await setStage(job, current, "running");
    const [refStats, refFreeze] = await Promise.all([frameStats(p("reference.mp4")), freezeIntervals(p("reference.mp4"))]);
    const qc = async () => {
      const info = await probe(p("output.mp4"));
      const [st, fr] = await Promise.all([frameStats(p("output.mp4")), freezeIntervals(p("output.mp4"))]);
      return runServerChecks({ output: info, outputStats: st, outputFreezes: fr, reference: refInfo, referenceStats: refStats, referenceFreezes: refFreeze, expectAudio: wantAudio });
    };
    let checks = await qc();
    job.quality = { checks };
    const issues = checks.filter((c) => c.status === "warn" || c.status === "fail");
    await setStage(job, current, "done", issues.length ? `${issues.length} issue(s) flagged: ${issues.map((c) => c.label).join(", ")}` : "No issues flagged by the automated video checks.");

    current = "refine";
    const flicker = checks.find((c) => c.id === "flicker");
    if (s.autoRefine && flicker && (flicker.status === "warn" || flicker.status === "fail")) {
      await setStage(job, current, "running", "Re-rendering with temporal deflicker…");
      await fs.rename(p("output.mp4"), p("output-pre-refine.mp4"));
      const n2 = await renderFinal(p("raw.mp4"), p("output.mp4"), rawInfo, { ...renderOpts, deflicker: true });
      const before = checks;
      checks = await qc();
      const after = checks.find((c) => c.id === "flicker")!;
      const improved = after.status === "pass" || (after.status === "warn" && flicker.status === "fail");
      job.quality = { checks, initialChecks: before, refinement: `Deflicker pass: ${flicker.value} → ${after.value}.` };
      job.notes.push(...n2.filter((n) => n.includes("deflicker")));
      await setStage(job, current, "done", `Deflicker pass ${improved ? "reduced" : "did not resolve"} flicker (${flicker.value} → ${after.value}).`);
    } else {
      const structural = checks.filter((c) => (c.id === "continuity" || c.id === "frozen") && c.status !== "pass");
      await setStage(
        job,
        current,
        "skipped",
        structural.length
          ? "Jumps/frozen segments cannot be repaired locally because the model only regenerates whole clips. Use Regenerate for a new sample."
          : s.autoRefine
            ? "No correctable issues detected."
            : "Auto-refine disabled.",
      );
    }

    current = "render";
    await setStage(job, current, "running");
    const outInfo = await probe(p("output.mp4"));
    job.output = { file: "output.mp4", info: outInfo, rawFile: "raw.mp4", rawInfo };
    await setStage(job, current, "done", `${outInfo.width}×${outInfo.height} @ ${outInfo.fps.toFixed(2)} fps, ${outInfo.durationSec.toFixed(2)} s, ${(outInfo.sizeBytes / 1e6).toFixed(1)} MB H.264 MP4`);
    job.status = "succeeded";
    await save(job);
  } catch (e) {
    await fail(job, current, e);
  }
}

export async function saveClientChecks(job: Job, checks: QualityCheck[]) {
  job.quality ??= { checks: [] };
  job.quality.clientChecks = checks.map((c) => ({ ...c, source: "vision" as const }));
  await save(job);
}

/** Load persisted jobs and resume polling for generations that were in flight when the server stopped. */
export async function restoreJobs() {
  const root = path.join(config.dataDir, "jobs");
  await fs.mkdir(root, { recursive: true });
  for (const id of await fs.readdir(root)) {
    try {
      const job: Job = JSON.parse(await fs.readFile(path.join(root, id, "job.json"), "utf8"));
      jobs.set(id, job);
      if (job.status !== "running") continue;
      const model = getModel(job.modelId);
      const gen = job.stages.find((s) => s.id === "generate");
      if (model && job.provider.requestId && gen?.status === "running" && adapters[model.provider].configured()) {
        const ctl = { cancel: false };
        running.set(id, ctl);
        void (async () => {
          try {
            await pollUntilDone(job, model, ctl);
            await continueAfterGeneration(job, model, ctl);
          } catch (e) {
            await fail(job, "generate", e);
          } finally {
            running.delete(id);
          }
        })();
      } else {
        await fail(job, job.stages.find((s) => s.status === "running")?.id ?? "render", new Error("Server restarted while this job was processing; please start it again."));
      }
    } catch {
      /* ignore unreadable job dirs */
    }
  }
}
