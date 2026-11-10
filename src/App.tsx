import { useEffect, useMemo, useRef, useState } from "react";
import { cancelJob, createJob, getConfig, listJobs, regenerateJob, subscribeJob, type AppConfig, type GenerationSettings, type Job } from "./api";
import { DropZone, ImageIcon, VideoIcon } from "./components/DropZone";
import { IdentityBadge, type IdentityState } from "./components/IdentityBadge";
import { LandmarkMedia, LayerToggles } from "./components/LandmarkMedia";
import { PipelinePanel } from "./components/PipelinePanel";
import { ResultPanel } from "./components/ResultPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import type { Layers } from "./vision/draw";
import { analyzeImage, analyzeVideo, type AnalysisResult, type FrameLandmarks, type ImageAnalysis } from "./vision/landmarks";
import { preflightIssues, summarizePreflight } from "./vision/metrics";

function useObjectUrl(file: File | null) {
  const url = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => void (url && URL.revokeObjectURL(url)), [url]);
  return url;
}

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [settings, setSettings] = useState<GenerationSettings | null>(null);

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [identityFiles, setIdentityFiles] = useState<File[]>([]);
  const imageUrl = useObjectUrl(imageFile);
  const videoUrl = useObjectUrl(videoFile);

  const [imageAnalysis, setImageAnalysis] = useState<ImageAnalysis | null>(null);
  const [imageState, setImageState] = useState<"idle" | "running" | "error">("idle");
  const [videoAnalysis, setVideoAnalysis] = useState<AnalysisResult | null>(null);
  const [videoProgress, setVideoProgress] = useState<number | null>(null);
  const [liveFrame, setLiveFrame] = useState<FrameLandmarks | null>(null);
  const [visionError, setVisionError] = useState<string | null>(null);
  const [refDuration, setRefDuration] = useState<number | null>(null);
  const [layers, setLayers] = useState<Layers>({ body: true, hands: true, face: true });

  const [consent, setConsent] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const refVideo = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    getConfig()
      .then((c) => {
        setConfig(c);
        const firstAvailable = c.models.find((m) => m.available)?.id ?? c.defaults.modelId;
        setSettings({ ...c.defaults, modelId: firstAvailable });
      })
      .catch((e) => setConfigError(e.message));
    listJobs().then(setJobs).catch(() => undefined);
  }, []);

  // Character image analysis.
  useEffect(() => {
    setImageAnalysis(null);
    if (!imageUrl) return setImageState("idle");
    let alive = true;
    setImageState("running");
    analyzeImage(imageUrl)
      .then((a) => alive && (setImageAnalysis(a), setImageState("idle")))
      .catch((e) => alive && (setImageState("error"), setVisionError(e.message)));
    return () => {
      alive = false;
    };
  }, [imageUrl]);

  // Reference video frame-by-frame landmark extraction.
  useEffect(() => {
    setVideoAnalysis(null);
    setLiveFrame(null);
    setRefDuration(null);
    if (!videoUrl) return setVideoProgress(null);
    const ctl = new AbortController();
    const probe = document.createElement("video");
    probe.preload = "metadata";
    probe.src = videoUrl;
    probe.onloadedmetadata = () => setRefDuration(probe.duration);
    setVideoProgress(0);
    analyzeVideo(videoUrl, {
      sampleFps: 12,
      signal: ctl.signal,
      onProgress: (f, frame) => {
        setVideoProgress(f);
        setLiveFrame(frame);
      },
    })
      .then((a) => {
        setVideoAnalysis(a);
        setLiveFrame(null);
      })
      .catch((e) => e.name !== "AbortError" && setVisionError(e.message))
      .finally(() => !ctl.signal.aborted && setVideoProgress(null));
    return () => ctl.abort();
  }, [videoUrl]);

  useEffect(() => {
    if (!activeJob || activeJob.status !== "running") return;
    return subscribeJob(activeJob.id, (j) => {
      setActiveJob(j);
      setJobs((js) => [j, ...js.filter((x) => x.id !== j.id)]);
    });
  }, [activeJob?.id, activeJob?.status]);

  const preflight = useMemo(() => summarizePreflight(imageAnalysis, videoAnalysis), [imageAnalysis, videoAnalysis]);
  const issues = useMemo(() => preflightIssues(imageAnalysis, preflight), [imageAnalysis, preflight]);
  const model = config?.models.find((m) => m.id === settings?.modelId);

  if (configError) return <div className="p-10 text-red-300">Could not reach the server: {configError}</div>;
  if (!config || !settings) return <div className="grid h-full place-items-center text-ink-400">Loading studio…</div>;

  const elementMode = model?.capabilities.identityElement.supported && settings.identityLock && !settings.cameraMotionTransfer;
  const identityState: IdentityState = (() => {
    const vc = activeJob?.quality?.clientChecks?.find((c) => c.id === "identity");
    if (vc && activeJob?.status === "succeeded") return { kind: "verified", check: vc };
    if (!imageFile) return { kind: "none" };
    if (imageState === "running") return { kind: "analyzing" };
    if (imageAnalysis && !imageAnalysis.frame.face) return { kind: "no-face" };
    return { kind: "locked", mode: !settings.identityLock ? "off" : elementMode ? "element" : "prompt" };
  })();

  const maxDur = model ? (settings.cameraMotionTransfer ? model.maxDuration.cameraOn : model.maxDuration.cameraOff) : 0;
  const durationBlock = refDuration !== null && refDuration > maxDur + 0.05 && !settings.trimToModelLimit;
  const analyzing = imageState === "running" || videoProgress !== null;
  const busy = uploadProgress !== null || activeJob?.status === "running";
  const blockers = [
    !imageFile && "Upload a character image",
    !videoFile && "Upload a reference video",
    !model?.available && `${model?.label} needs ${model?.requiredEnv} on the server`,
    durationBlock && `Reference is ${refDuration?.toFixed(1)} s; this model/setting allows ${maxDur} s (enable trim)`,
    analyzing && "Wait for landmark analysis to finish",
    !consent && "Confirm you have rights to the character image",
  ].filter(Boolean) as string[];

  const submit = async () => {
    if (!imageFile || !videoFile) return;
    setSubmitError(null);
    setUploadProgress(0);
    const form = new FormData();
    form.append("image", imageFile);
    form.append("video", videoFile);
    identityFiles.forEach((f) => form.append("identity", f));
    form.append("settings", JSON.stringify(settings));
    form.append("preflight", JSON.stringify(preflight));
    form.append("consent", String(consent));
    try {
      const job = await createJob(form, setUploadProgress);
      setActiveJob(job);
      setJobs((js) => [job, ...js]);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploadProgress(null);
    }
  };

  const noProvider = !config.providers.fal && !config.providers.replicate && !config.providers.selfhosted;
  const sh = config.selfhostedHealth;
  const chip = (on: boolean, label: string, state: string) => (
    <span className={`chip ${on ? "bg-emerald-500/15 text-emerald-300" : "bg-ink-800 text-ink-400"}`}>
      {label} {state}
    </span>
  );

  return (
    <div className="mx-auto flex min-h-full max-w-[1680px] flex-col gap-4 p-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-accent to-accent-2 text-white shadow-lg shadow-accent/30">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
              <circle cx="12" cy="5" r="2" />
              <path d="M12 8v6M7 11l5-2 5 2M9 21l3-7 3 7" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-white">Motion Control Studio</h1>
            <p className="text-xs text-ink-400">Animate your existing character with motion from any reference video</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          {chip(!!sh?.ok, "Self-hosted GPU", !config.providers.selfhosted ? "not configured" : sh?.ok ? `ready · ${sh.gpus.join(", ")}` : "unhealthy")}
          {chip(config.providers.fal, "fal.ai", config.providers.fal ? "connected" : "not configured")}
          {chip(config.providers.replicate, "Replicate", config.providers.replicate ? "connected" : "not configured")}
        </div>
      </header>

      {config.providers.selfhosted && sh && !sh.ok && (
        <div className="rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-100">
          Self-hosted GPU worker is not ready: {sh.problems.join(" · ")}
        </div>
      )}

      {noProvider && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          No generation backend is configured. Deploy the GPU worker in <code className="text-amber-300">worker/</code> and set{" "}
          <code className="text-amber-300">SELFHOSTED_WORKER_URL</code> + <code className="text-amber-300">SELFHOSTED_WORKER_TOKEN</code>, or set{" "}
          <code className="text-amber-300">FAL_KEY</code> / <code className="text-amber-300">REPLICATE_API_TOKEN</code>, then restart. Uploading and landmark analysis work
          without a backend; generation does not.
        </div>
      )}

      <div className="grid flex-1 gap-4 xl:grid-cols-[320px_minmax(0,1fr)_360px]">
        <aside className="panel max-h-[calc(100vh-110px)] overflow-y-auto p-4 xl:sticky xl:top-4">
          <SettingsPanel config={config} settings={settings} onChange={setSettings} referenceDuration={refDuration} />
        </aside>

        <main className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <section className="panel p-4">
              <div className="mb-3 flex items-center justify-between">
                <span className="panel-title">1 · AI character image</span>
                {imageAnalysis && (
                  <span className="text-[11px] text-ink-400">
                    {imageAnalysis.width}×{imageAnalysis.height}
                  </span>
                )}
              </div>
              <DropZone accept="image/png,image/jpeg,image/webp" title="Upload your character" hint="Permanent identity source · PNG, JPG or WebP · face clearly visible" icon={<ImageIcon />} file={imageFile} onFile={setImageFile}>
                {imageUrl && <LandmarkMedia kind="image" src={imageUrl} analysis={imageAnalysis ? { frames: [imageAnalysis.frame], durationSec: 0, width: imageAnalysis.width, height: imageAnalysis.height, sampleFps: 1 } : null} layers={layers} className="h-[300px]" />}
              </DropZone>
            </section>
            <section className="panel p-4">
              <div className="mb-3 flex items-center justify-between">
                <span className="panel-title">2 · Motion reference video</span>
                {refDuration !== null && <span className="text-[11px] text-ink-400">{refDuration.toFixed(1)} s</span>}
              </div>
              <DropZone accept="video/mp4,video/quicktime,video/webm" title="Upload reference video" hint={`Motion source only · MP4/MOV/WebM · up to ${maxDur} s for this model`} icon={<VideoIcon />} file={videoFile} onFile={setVideoFile}>
                {videoUrl &&
                  (videoProgress !== null && liveFrame ? (
                    <LandmarkMedia kind="video" src={videoUrl} frame={liveFrame} layers={layers} muted className="h-[300px]" label="Extracting landmarks…" videoRef={refVideo} />
                  ) : (
                    <LandmarkMedia kind="video" src={videoUrl} analysis={videoAnalysis} layers={layers} controls className="h-[300px]" videoRef={refVideo} />
                  ))}
              </DropZone>
            </section>
          </div>

          <section className="panel p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <span className="panel-title">Motion analysis · pose tracking</span>
              <LayerToggles layers={layers} onChange={setLayers} />
            </div>
            {videoProgress !== null && (
              <div className="mb-3">
                <div className="mb-1 flex justify-between text-xs text-ink-300">
                  <span>Analyzing reference frame by frame (MediaPipe pose · hands · face)</span>
                  <span className="tabular-nums">{Math.round(videoProgress * 100)}%</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-ink-700">
                  <div className="h-full bg-gradient-to-r from-accent to-accent-2 transition-all" style={{ width: `${videoProgress * 100}%` }} />
                </div>
              </div>
            )}
            {preflight.video ? (
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                {[
                  ["Frames analysed", String(preflight.video.framesAnalyzed)],
                  ["Body tracked", `${Math.round(preflight.video.poseDetectionRate * 100)}%`],
                  ["Full body", `${Math.round(preflight.video.fullBodyRate * 100)}%`],
                  ["Face tracked", `${Math.round(preflight.video.faceDetectionRate * 100)}%`],
                  ["Hands tracked", `${Math.round(preflight.video.handDetectionRate * 100)}%`],
                ].map(([k, v]) => (
                  <div key={k} className="rounded-lg bg-ink-850 px-3 py-2">
                    <div className="text-[11px] text-ink-400">{k}</div>
                    <div className="text-lg font-semibold tabular-nums text-white">{v}</div>
                  </div>
                ))}
              </div>
            ) : (
              videoProgress === null && <p className="text-sm text-ink-400">Upload a reference video to extract body, hand and face landmarks in your browser.</p>
            )}
            {issues.length > 0 && (
              <ul className="mt-3 space-y-1 text-xs">
                {issues.map((i, k) => (
                  <li key={k} className={i.level === "ok" ? "text-emerald-300" : i.level === "warn" ? "text-amber-300" : "text-red-300"}>
                    {i.level === "ok" ? "✓" : "!"} {i.text}
                  </li>
                ))}
              </ul>
            )}
            {visionError && <p className="mt-2 text-xs text-red-300">{visionError} You can still generate; the model performs its own motion extraction.</p>}
          </section>

          {activeJob?.status === "succeeded" && activeJob.output && (
            <section className="panel p-4">
              <div className="panel-title mb-3">Result · {activeJob.modelLabel}</div>
              <ResultPanel
                key={activeJob.id}
                job={activeJob}
                onVisionChecks={(checks) => setActiveJob((j) => (j ? { ...j, quality: { ...(j.quality ?? { checks: [] }), clientChecks: checks } } : j))}
                onRegenerate={async () => {
                  try {
                    const j = await regenerateJob(activeJob.id);
                    setActiveJob(j);
                    setJobs((js) => [j, ...js]);
                  } catch (e) {
                    setSubmitError(e instanceof Error ? e.message : String(e));
                  }
                }}
              />
            </section>
          )}
        </main>

        <aside className="space-y-4">
          <section className="panel space-y-3 p-4">
            <div className="panel-title">Face identity protection</div>
            <IdentityBadge state={identityState} />
            {elementMode && (
              <div>
                <div className="mb-1 text-xs text-ink-300">Extra character angles (optional, up to 3)</div>
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  multiple
                  className="block w-full text-xs text-ink-300 file:mr-2 file:rounded-lg file:border-0 file:bg-ink-700 file:px-3 file:py-1.5 file:text-slate-200"
                  onChange={(e) => setIdentityFiles(Array.from(e.target.files ?? []).slice(0, 3))}
                />
                <p className="mt-1 text-[11px] text-ink-400">Side/three-quarter views of the same character strengthen the facial element.</p>
              </div>
            )}
          </section>

          <section className="panel space-y-3 p-4">
            <label className="flex items-start gap-2 text-xs text-ink-300">
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} className="mt-0.5 accent-[var(--color-accent)]" />
              I own this character image or have permission to animate it, and I won't use the output to impersonate a real person.
            </label>
            <button type="button" className="btn-primary w-full py-3 text-base" disabled={blockers.length > 0 || busy} onClick={submit}>
              {uploadProgress !== null ? `Uploading ${Math.round(uploadProgress * 100)}%` : activeJob?.status === "running" ? "Generating…" : "Generate motion video"}
            </button>
            {blockers.length > 0 && !busy && (
              <ul className="space-y-0.5 text-[11px] text-ink-400">
                {blockers.map((b) => (
                  <li key={b}>• {b}</li>
                ))}
              </ul>
            )}
            {submitError && <p className="rounded-lg bg-red-500/10 p-2 text-xs text-red-300">{submitError}</p>}
          </section>

          {activeJob && (
            <section className="panel space-y-3 p-4">
              <div className="flex items-center justify-between">
                <span className="panel-title">Generation pipeline</span>
                <span className="text-[11px] text-ink-400">{activeJob.modelLabel}</span>
              </div>
              {activeJob.warnings.length > 0 && (
                <ul className="space-y-1 rounded-lg bg-amber-500/10 p-2 text-[11px] text-amber-200">
                  {activeJob.warnings.map((w, i) => (
                    <li key={i}>! {w}</li>
                  ))}
                </ul>
              )}
              <PipelinePanel job={activeJob} onCancel={() => cancelJob(activeJob.id).then(setActiveJob)} />
              {activeJob.error && <p className="rounded-lg bg-red-500/10 p-2 text-xs text-red-300">{activeJob.error}</p>}
            </section>
          )}

          {jobs.length > 0 && (
            <section className="panel p-4">
              <div className="panel-title mb-2">History</div>
              <ul className="max-h-72 space-y-1 overflow-y-auto">
                {jobs.map((j) => (
                  <li key={j.id}>
                    <button
                      type="button"
                      onClick={() => setActiveJob(j)}
                      className={`flex w-full items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left text-xs hover:bg-ink-800 ${activeJob?.id === j.id ? "bg-ink-800" : ""}`}
                    >
                      <span className="truncate text-slate-300">
                        {new Date(j.createdAt).toLocaleString()} · {j.modelLabel}
                      </span>
                      <span
                        className={`chip ${
                          j.status === "succeeded" ? "bg-emerald-500/15 text-emerald-300" : j.status === "running" ? "bg-accent/20 text-violet-200" : "bg-red-500/15 text-red-300"
                        }`}
                      >
                        {j.status}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </aside>
      </div>
    </div>
  );
}
