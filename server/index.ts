import express, { type NextFunction, type Request, type Response } from "express";
import multer from "multer";
import fs from "node:fs/promises";
import path from "node:path";
import { config } from "./config.js";
import { cancelJob, createJob, events, getJob, jobDir, listJobs, regenerate, restoreJobs, saveClientChecks, type Job } from "./jobs.js";
import { MODELS, adapters, getModel } from "./providers/index.js";
import type { QualityCheck } from "./quality.js";
import { DEFAULT_SETTINGS, parseSettings } from "./settings.js";

const app = express();
app.disable("x-powered-by");
app.use(express.json({ limit: "2mb" }));

const uploadDir = path.join(config.dataDir, "uploads");
await fs.mkdir(uploadDir, { recursive: true });
const upload = multer({
  storage: multer.diskStorage({
    destination: uploadDir,
    filename: (_req, file, cb) => cb(null, `${Date.now()}-${Math.random().toString(36).slice(2)}${path.extname(file.originalname).toLowerCase().slice(0, 8)}`),
  }),
  limits: { fileSize: config.maxUploadBytes, files: 5 },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    // Content is verified with ffprobe later; some browsers send octet-stream for .mov/.mkv.
    const ok =
      file.fieldname === "video"
        ? file.mimetype.startsWith("video/") || [".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"].includes(ext)
        : file.mimetype.startsWith("image/") || [".png", ".jpg", ".jpeg", ".webp", ".bmp"].includes(ext);
    if (ok) cb(null, true);
    else cb(new Error(`${file.fieldname}: unsupported file type ${file.mimetype}`));
  },
});

const publicJob = (j: Job) => j;

app.get("/api/config", (_req, res) => {
  res.json({
    providers: { fal: adapters.fal.configured(), replicate: adapters.replicate.configured() },
    defaults: DEFAULT_SETTINGS,
    maxUploadMB: config.maxUploadBytes / 1024 / 1024,
    models: MODELS.map((m) => ({
      ...m,
      maxDurationSec: undefined,
      maxDuration: { cameraOn: m.maxDurationSec({ ...DEFAULT_SETTINGS, cameraMotionTransfer: true }), cameraOff: m.maxDurationSec({ ...DEFAULT_SETTINGS, cameraMotionTransfer: false }) },
      available: adapters[m.provider].configured(),
    })),
  });
});

app.get("/api/jobs", (_req, res) => res.json(listJobs().map(publicJob)));

app.post(
  "/api/jobs",
  upload.fields([{ name: "image", maxCount: 1 }, { name: "video", maxCount: 1 }, { name: "identity", maxCount: 3 }]),
  async (req, res) => {
    const files = (req.files ?? {}) as Record<string, Express.Multer.File[]>;
    const all = Object.values(files).flat();
    try {
      if (req.body.consent !== "true") throw new Error("Please confirm you own or have permission to animate this character image.");
      const image = files.image?.[0];
      const video = files.video?.[0];
      if (!image) throw new Error("Character image is required.");
      if (!video) throw new Error("Reference video is required.");
      const settings = parseSettings(req.body.settings);
      const model = getModel(settings.modelId);
      if (!model) throw new Error(`Unknown model ${settings.modelId}`);
      const preflight = req.body.preflight ? JSON.parse(req.body.preflight) : {};
      const job = await createJob({
        model,
        settings,
        preflight,
        imagePath: image.path,
        videoPath: video.path,
        identityPaths: (files.identity ?? []).map((f) => f.path),
      });
      res.status(201).json(publicJob(job));
    } catch (e) {
      await Promise.all(all.map((f) => fs.rm(f.path, { force: true })));
      res.status(400).json({ error: e instanceof Error ? e.message : String(e) });
    }
  },
);

function withJob(handler: (job: Job, req: Request, res: Response) => unknown) {
  return async (req: Request, res: Response, next: NextFunction) => {
    const job = getJob(String(req.params.id));
    if (!job) return res.status(404).json({ error: "Job not found" });
    try {
      await handler(job, req, res);
    } catch (e) {
      next(e);
    }
  };
}

app.get("/api/jobs/:id", withJob((job, _req, res) => res.json(publicJob(job))));

app.get(
  "/api/jobs/:id/events",
  withJob((job, req, res) => {
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive", "X-Accel-Buffering": "no" });
    const send = (j: Job) => res.write(`data: ${JSON.stringify(publicJob(j))}\n\n`);
    send(job);
    const onUpdate = (j: Job) => send(j);
    events.on(job.id, onUpdate);
    const ping = setInterval(() => res.write(": ping\n\n"), 15000);
    req.on("close", () => {
      clearInterval(ping);
      events.off(job.id, onUpdate);
    });
  }),
);

app.post("/api/jobs/:id/cancel", withJob(async (job, _req, res) => {
  await cancelJob(job);
  res.json(publicJob(job));
}));

app.post("/api/jobs/:id/regenerate", withJob(async (job, req, res) => {
  try {
    const patch = req.body?.settings ? parseSettings({ ...job.settings, ...req.body.settings }) : {};
    const next = await regenerate(job, patch);
    res.status(201).json(publicJob(next));
  } catch (e) {
    res.status(400).json({ error: e instanceof Error ? e.message : String(e) });
  }
}));

app.post("/api/jobs/:id/client-checks", withJob(async (job, req, res) => {
  const checks = (req.body?.checks ?? []) as QualityCheck[];
  if (!Array.isArray(checks) || checks.length > 50) return res.status(400).json({ error: "Invalid checks" });
  await saveClientChecks(job, checks);
  res.json({ ok: true });
}));

const SERVABLE = /^(source-image\.\w+|source-video\.\w+|character\.png|reference\.mp4|raw\.mp4|output\.mp4|output-pre-refine\.mp4|identity-face\.png|identity-angle-\d\.png)$/;
app.get("/api/jobs/:id/files/:name", withJob((job, req, res) => {
  const name = String(req.params.name);
  if (!SERVABLE.test(name)) return res.status(404).end();
  const opts = { root: jobDir(job.id), dotfiles: "deny" as const };
  if (req.query.download) {
    const base = `motion-${job.id.slice(0, 8)}-${job.settings.resolution}-${job.settings.fps}fps.mp4`;
    return res.download(name, base, opts);
  }
  res.sendFile(name, opts);
}));

app.use("/api", (_req, res) => res.status(404).json({ error: "Not found" }));

app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
  const isUpload = err instanceof multer.MulterError || /unsupported file type/.test(err?.message ?? "");
  const status = err?.code === "LIMIT_FILE_SIZE" ? 413 : isUpload ? 400 : 500;
  const msg = err?.code === "LIMIT_FILE_SIZE" ? `File too large (max ${config.maxUploadBytes / 1024 / 1024} MB)` : err?.message ?? "Server error";
  if (status === 500) console.error(err);
  res.status(status).json({ error: msg });
});

await restoreJobs();

// Self-host the MediaPipe WASM runtime so in-browser landmark tracking does not depend on a JS CDN.
app.use("/mediapipe/wasm", express.static(path.resolve("node_modules/@mediapipe/tasks-vision/wasm"), { maxAge: "7d" }));

if (config.isProd) {
  const dist = path.resolve("dist");
  app.use(express.static(dist, { index: false }));
  app.get("/{*splat}", (_req, res) => res.sendFile(path.join(dist, "index.html")));
} else {
  const { createServer } = await import("vite");
  const vite = await createServer({ server: { middlewareMode: true }, appType: "spa" });
  app.use(vite.middlewares);
}

app.listen(config.port, "0.0.0.0", () => {
  const p = [adapters.fal.configured() && "fal.ai", adapters.replicate.configured() && "Replicate"].filter(Boolean);
  console.log(`Motion Control Studio on http://localhost:${config.port} — providers: ${p.length ? p.join(", ") : "NONE (set FAL_KEY or REPLICATE_API_TOKEN)"}`);
});
