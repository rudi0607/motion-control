import { useCallback, useEffect, useRef, useState } from "react";
import { fileUrl, postClientChecks, type Job, type QualityCheck } from "../api";
import { analyzeImage, analyzeVideo, type AnalysisResult, type ImageAnalysis } from "../vision/landmarks";
import { visionChecks } from "../vision/metrics";
import type { Layers } from "../vision/draw";
import { LandmarkMedia, LayerToggles } from "./LandmarkMedia";
import { QualityReport } from "./PipelinePanel";

type Mode = "side" | "wipe" | "raw";

/** Keeps follower videos locked to the leader's play state and time. */
function useSyncedVideos(leader: React.RefObject<HTMLVideoElement | null>, followers: React.RefObject<HTMLVideoElement | null>[], deps: unknown[]) {
  useEffect(() => {
    const L = leader.current;
    if (!L) return;
    const each = (fn: (v: HTMLVideoElement) => void) => followers.forEach((r) => r.current && fn(r.current));
    const onPlay = () => each((v) => void v.play().catch(() => undefined));
    const onPause = () => each((v) => v.pause());
    const onSeek = () => each((v) => (v.currentTime = Math.min(L.currentTime, (v.duration || Infinity) - 0.01)));
    const onRate = () => each((v) => (v.playbackRate = L.playbackRate));
    const timer = setInterval(() => each((v) => !L.paused && Math.abs(v.currentTime - L.currentTime) > 0.08 && (v.currentTime = L.currentTime)), 500);
    L.addEventListener("play", onPlay);
    L.addEventListener("pause", onPause);
    L.addEventListener("seeked", onSeek);
    L.addEventListener("ratechange", onRate);
    return () => {
      clearInterval(timer);
      L.removeEventListener("play", onPlay);
      L.removeEventListener("pause", onPause);
      L.removeEventListener("seeked", onSeek);
      L.removeEventListener("ratechange", onRate);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

const visionCache = new Map<string, { image: ImageAnalysis | null; ref: AnalysisResult; out: AnalysisResult }>();

export function ResultPanel(props: { job: Job; onRegenerate: () => void; onVisionChecks: (c: QualityCheck[]) => void }) {
  const { job } = props;
  const [mode, setMode] = useState<Mode>("side");
  const [wipe, setWipe] = useState(50);
  const [layers, setLayers] = useState<Layers>({ body: false, hands: false, face: false });
  const [vision, setVision] = useState<{ image: ImageAnalysis | null; ref: AnalysisResult; out: AnalysisResult } | null>(visionCache.get(job.id) ?? null);
  const [visionProgress, setVisionProgress] = useState<number | null>(null);
  const [visionError, setVisionError] = useState<string | null>(null);
  const outRef = useRef<HTMLVideoElement | null>(null);
  const refRef = useRef<HTMLVideoElement | null>(null);
  const rawRef = useRef<HTMLVideoElement | null>(null);

  useSyncedVideos(outRef, [refRef, rawRef], [mode, job.id]);

  const runVision = useCallback(async () => {
    setVisionError(null);
    setVisionProgress(0);
    try {
      const image = await analyzeImage(fileUrl(job, "character.png")).catch(() => null);
      const ref = await analyzeVideo(fileUrl(job, "reference.mp4"), { sampleFps: 10, onProgress: (f) => setVisionProgress(f * 0.5) });
      const out = await analyzeVideo(fileUrl(job, "output.mp4"), { sampleFps: 10, onProgress: (f) => setVisionProgress(0.5 + f * 0.5) });
      const v = { image, ref, out };
      visionCache.set(job.id, v);
      setVision(v);
      const checks = visionChecks(image, ref, out);
      props.onVisionChecks(checks);
      await postClientChecks(job.id, checks).catch(() => undefined);
    } catch (e) {
      setVisionError(e instanceof Error ? e.message : String(e));
    } finally {
      setVisionProgress(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id]);

  useEffect(() => {
    setVision(visionCache.get(job.id) ?? null);
    if (job.status === "succeeded" && !job.quality?.clientChecks && !visionCache.has(job.id)) void runVision();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id, job.status]);

  const seek = (t: number) => {
    if (outRef.current) {
      outRef.current.currentTime = t;
      outRef.current.pause();
    }
  };

  const out = job.output!;
  const vChecks = job.quality?.clientChecks;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex gap-1 rounded-xl bg-ink-850 p-1">
          {(
            [
              ["side", "Reference ↔ Generated"],
              ["wipe", "Before / After"],
              ["raw", "Raw model ↔ Final"],
            ] as [Mode, string][]
          ).map(([m, l]) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium ${mode === m ? "bg-ink-600 text-white" : "text-ink-300 hover:text-white"}`}
            >
              {l}
            </button>
          ))}
        </div>
        {vision && <LayerToggles layers={layers} onChange={setLayers} />}
      </div>

      {mode === "side" && (
        <div className="grid gap-3 md:grid-cols-2">
          <LandmarkMedia kind="video" src={fileUrl(job, "reference.mp4")} videoRef={refRef} analysis={vision?.ref} layers={layers} muted label="Reference (motion source)" className="aspect-video" />
          <LandmarkMedia kind="video" src={fileUrl(job, "output.mp4")} videoRef={outRef} analysis={vision?.out} layers={layers} controls label="Generated" className="aspect-video" />
        </div>
      )}
      {mode === "wipe" && (
        <div>
          <div className="relative aspect-video overflow-hidden rounded-xl bg-black">
            <video ref={outRef} src={fileUrl(job, "output.mp4")} className="absolute inset-0 h-full w-full object-contain" controls playsInline loop />
            <div className="pointer-events-none absolute inset-0" style={{ clipPath: `inset(0 ${100 - wipe}% 0 0)` }}>
              <img src={fileUrl(job, "character.png")} className="h-full w-full bg-black object-contain" alt="Character image" />
            </div>
            <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/80" style={{ left: `${wipe}%` }} />
            <span className="absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 text-[11px]">Before · character image</span>
            <span className="absolute right-2 top-2 rounded bg-black/60 px-2 py-0.5 text-[11px]">After · generated</span>
          </div>
          <input type="range" min={0} max={100} value={wipe} onChange={(e) => setWipe(Number(e.target.value))} className="mt-2 w-full" aria-label="Before/after position" />
        </div>
      )}
      {mode === "raw" && (
        <div className="grid gap-3 md:grid-cols-2">
          <LandmarkMedia kind="video" src={fileUrl(job, "raw.mp4")} videoRef={rawRef} layers={layers} muted label={`Raw model · ${job.output!.rawInfo.width}×${job.output!.rawInfo.height} @ ${job.output!.rawInfo.fps.toFixed(0)}fps`} className="aspect-video" />
          <LandmarkMedia kind="video" src={fileUrl(job, "output.mp4")} videoRef={outRef} analysis={vision?.out} layers={layers} controls label={`Final · ${out.info.width}×${out.info.height} @ ${out.info.fps.toFixed(0)}fps`} className="aspect-video" />
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <a className="btn-primary" href={fileUrl(job, "output.mp4", true)}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 4v12m0 0l-5-5m5 5l5-5M5 20h14" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Download MP4
        </a>
        <button type="button" className="btn-ghost" onClick={props.onRegenerate}>
          Regenerate (new sample)
        </button>
        <span className="text-xs text-ink-400">
          {out.info.width}×{out.info.height} · {out.info.fps.toFixed(2)} fps · {out.info.durationSec.toFixed(2)} s · {(out.info.sizeBytes / 1e6).toFixed(1)} MB
          {out.info.hasAudio ? " · audio" : ""}
        </span>
      </div>

      {job.notes.length > 0 && (
        <ul className="space-y-0.5 text-[11px] text-ink-300">
          {job.notes.map((n, i) => (
            <li key={i}>• {n}</li>
          ))}
        </ul>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        {job.quality?.checks && <QualityReport title="Video checks (ffmpeg)" checks={job.quality.checks} onSeek={seek} />}
        <div>
          {vChecks ? (
            <QualityReport title="Landmark checks (MediaPipe)" checks={vChecks} onSeek={seek} />
          ) : (
            <div className="panel-title mb-2">Landmark checks (MediaPipe)</div>
          )}
          {visionProgress !== null && (
            <div className="mt-2 text-xs text-ink-300">
              Tracking reference and output frames… {Math.round(visionProgress * 100)}%
              <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ink-700">
                <div className="h-full bg-accent-2" style={{ width: `${visionProgress * 100}%` }} />
              </div>
            </div>
          )}
          {visionError && <p className="mt-2 text-xs text-red-300">{visionError}</p>}
          {visionProgress === null && (
            <button type="button" className="btn-ghost mt-2 text-xs" onClick={runVision}>
              {vChecks ? "Re-run landmark checks" : "Run landmark checks"}
            </button>
          )}
        </div>
      </div>
      {job.quality?.refinement && <p className="text-xs text-ink-300">Refinement: {job.quality.refinement}</p>}
      <p className="text-[11px] leading-snug text-ink-400">
        Automated checks are heuristics on sampled frames. They can miss defects (for example texture changes or an extra finger hidden by the 21-point hand model) and
        cannot guarantee every frame is correct — review the video, especially flagged time ranges.
      </p>
    </div>
  );
}
