import type { FrameStat, MediaInfo } from "./media.js";

export type CheckStatus = "pass" | "warn" | "fail" | "skipped";

export interface QualityCheck {
  id: string;
  label: string;
  status: CheckStatus;
  value?: string;
  detail: string;
  /** Problem time ranges in seconds. */
  ranges?: { start: number; end: number }[];
  source: "ffmpeg" | "vision";
}

function median(xs: number[]) {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function groupRanges(times: number[], frameDur: number): { start: number; end: number }[] {
  const out: { start: number; end: number }[] = [];
  for (const t of [...times].sort((a, b) => a - b)) {
    const last = out[out.length - 1];
    if (last && t - last.end <= frameDur * 2.5) last.end = t + frameDur;
    else out.push({ start: t, end: t + frameDur });
  }
  return out;
}

/** Pearson correlation of two series resampled to the same length. */
export function correlation(a: number[], b: number[]): number | null {
  const n = Math.min(a.length, b.length, 400);
  if (n < 8) return null;
  const rs = (xs: number[]) => Array.from({ length: n }, (_, i) => xs[Math.floor((i * xs.length) / n)]);
  const x = rs(a);
  const y = rs(b);
  const mx = x.reduce((s, v) => s + v, 0) / n;
  const my = y.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
    syy += (y[i] - my) ** 2;
  }
  if (sxx === 0 || syy === 0) return null;
  return sxy / Math.sqrt(sxx * syy);
}

/** Frames whose brightness oscillates against both neighbours (flicker), excluding scene changes. */
export function flickerFrames(stats: FrameStat[], threshold = 2.5): number[] {
  const out: number[] = [];
  for (let i = 1; i < stats.length - 1; i++) {
    const a = stats[i - 1].yavg, b = stats[i].yavg, c = stats[i + 1].yavg;
    const d1 = b - a, d2 = c - b;
    // Sign reversal with magnitude on both sides = up-down (or down-up) brightness pulse.
    if (Math.sign(d1) !== Math.sign(d2) && Math.min(Math.abs(d1), Math.abs(d2)) > threshold) out.push(stats[i].t);
  }
  return out;
}

/** Frames with an abrupt jump in content relative to the clip's typical frame-to-frame change. */
export function jumpFrames(stats: FrameStat[]): number[] {
  const difs = stats.slice(1).map((s) => s.ydif);
  const med = median(difs);
  const mad = median(difs.map((d) => Math.abs(d - med))) || 0.5;
  const thr = Math.max(med + 8 * mad * 1.4826, med * 3.5, 10);
  return stats.slice(1).filter((s) => s.ydif > thr).map((s) => s.t);
}

export interface ServerQcInput {
  output: MediaInfo;
  outputStats: FrameStat[];
  outputFreezes: { start: number; end: number }[];
  reference: MediaInfo;
  referenceStats: FrameStat[];
  referenceFreezes: { start: number; end: number }[];
  expectAudio: boolean;
}

export function runServerChecks(i: ServerQcInput): QualityCheck[] {
  const checks: QualityCheck[] = [];
  const fd = 1 / (i.output.fps || 30);

  const dd = Math.abs(i.output.durationSec - i.reference.durationSec);
  checks.push({
    id: "duration",
    label: "Duration matches reference",
    status: dd <= Math.max(0.1, 2 * fd) ? "pass" : dd < 0.5 ? "warn" : "fail",
    value: `${i.output.durationSec.toFixed(2)} s vs ${i.reference.durationSec.toFixed(2)} s`,
    detail: `Difference ${dd.toFixed(3)} s.`,
    source: "ffmpeg",
  });

  if (i.expectAudio) {
    if (!i.output.hasAudio) {
      checks.push({ id: "av_sync", label: "Video/audio synchronization", status: "fail", detail: "Audio was requested but the output has no audio track.", source: "ffmpeg" });
    } else {
      const startOff = Math.abs((i.output.audioStartSec ?? 0) - i.output.videoStartSec);
      const durOff = Math.abs((i.output.audioDurationSec ?? 0) - i.output.durationSec);
      checks.push({
        id: "av_sync",
        label: "Video/audio synchronization",
        status: startOff < 0.045 && durOff < 0.15 ? "pass" : startOff < 0.1 ? "warn" : "fail",
        value: `start offset ${(startOff * 1000).toFixed(0)} ms`,
        detail: `Audio/video start offset ${(startOff * 1000).toFixed(0)} ms, length difference ${(durOff * 1000).toFixed(0)} ms. Container-level check; it does not measure lip/sound alignment.`,
        source: "ffmpeg",
      });
    }
  } else {
    checks.push({ id: "av_sync", label: "Video/audio synchronization", status: "skipped", detail: "No audio requested or reference has no audio.", source: "ffmpeg" });
  }

  const flick = flickerFrames(i.outputStats);
  const refFlick = flickerFrames(i.referenceStats);
  const flickRate = flick.length / Math.max(1, i.outputStats.length);
  const refRate = refFlick.length / Math.max(1, i.referenceStats.length);
  const excess = flickRate - refRate;
  checks.push({
    id: "flicker",
    label: "Frame flicker (brightness)",
    status: excess < 0.01 ? "pass" : excess < 0.04 ? "warn" : "fail",
    value: `${(flickRate * 100).toFixed(1)}% frames (reference ${(refRate * 100).toFixed(1)}%)`,
    detail: "Frames whose brightness pulses up and down against both neighbours, compared with the reference's own rate.",
    ranges: groupRanges(flick, fd),
    source: "ffmpeg",
  });

  const jumps = jumpFrames(i.outputStats);
  const refJumps = jumpFrames(i.referenceStats);
  checks.push({
    id: "continuity",
    label: "Motion continuity (sudden jumps)",
    status: jumps.length <= refJumps.length ? "pass" : jumps.length - refJumps.length <= 2 ? "warn" : "fail",
    value: `${jumps.length} jump(s) (reference ${refJumps.length})`,
    detail: "Abrupt frame-to-frame changes far above the clip's typical motion, e.g. pops, cuts or regenerated segments.",
    ranges: groupRanges(jumps, fd),
    source: "ffmpeg",
  });

  const overlapsRef = (r: { start: number; end: number }) =>
    i.referenceFreezes.some((f) => f.start < r.end && r.start < f.end);
  const newFreezes = i.outputFreezes.filter((r) => !overlapsRef(r)).map((r) => ({ start: r.start, end: Math.min(r.end, i.output.durationSec) }));
  checks.push({
    id: "frozen",
    label: "Frozen motion",
    status: newFreezes.length === 0 ? "pass" : "warn",
    value: newFreezes.length ? `${newFreezes.length} interval(s)` : "none",
    detail: "Intervals where the output stops moving while the reference keeps moving.",
    ranges: newFreezes,
    source: "ffmpeg",
  });

  const corr = correlation(i.referenceStats.slice(1).map((s) => s.ydif), i.outputStats.slice(1).map((s) => s.ydif));
  checks.push({
    id: "timing",
    label: "Motion timing vs reference",
    status: corr === null ? "skipped" : corr > 0.5 ? "pass" : corr > 0.2 ? "warn" : "fail",
    value: corr === null ? undefined : `r = ${corr.toFixed(2)}`,
    detail: "Correlation of global motion energy over time between reference and output. A coarse timing proxy; joint-level comparison runs in the browser.",
    source: "ffmpeg",
  });

  return checks;
}

export interface FaceReport {
  frames: number;
  frames_with_face: number;
  similarity_before: { mean: number; min: number; p10: number } | null;
  similarity_after: { mean: number; min: number; p10: number } | null;
  per_frame_after?: (number | null)[];
  restore?: boolean;
  restore_blend?: number;
}

/** ArcFace (w600k_r50) cosine similarity at or above this is treated as the same identity. */
export const SAME_IDENTITY = 0.5;

/** Identity check from the worker's face-lock pass: ArcFace similarity of every frame to the character. */
export function faceIdentityCheck(r: FaceReport | undefined, outFps: number, rawFps: number): QualityCheck | null {
  if (!r) return null;
  const after = r.similarity_after;
  if (!after) {
    return {
      id: "identity_arcface",
      label: "Face identity (ArcFace)",
      status: "warn",
      detail: "No face was found in the generated frames, so identity could not be locked or measured.",
      source: "ffmpeg",
    };
  }
  const coverage = r.frames ? r.frames_with_face / r.frames : 0;
  // Per-frame values are at the model's native rate, before frame-rate conversion.
  const bad = (r.per_frame_after ?? []).flatMap((v, i) => (v !== null && v < SAME_IDENTITY ? [i / rawFps] : []));
  const status = after.p10 >= SAME_IDENTITY && coverage >= 0.9 ? "pass" : after.mean >= SAME_IDENTITY ? "warn" : "fail";
  const before = r.similarity_before ? ` (before face lock ${r.similarity_before.mean.toFixed(2)})` : "";
  return {
    id: "identity_arcface",
    label: "Face identity (ArcFace)",
    status,
    value: `similarity ${after.mean.toFixed(2)} avg · ${after.min.toFixed(2)} min${before}`,
    detail: `Face-recognition similarity between your character image and ${r.frames_with_face}/${r.frames} generated frames (≥ ${SAME_IDENTITY} = same identity).${coverage < 0.9 ? " Some frames had no detectable face." : ""}${r.restore ? ` Face restoration blend ${r.restore_blend ?? 1}.` : ""}`,
    ranges: groupRanges(bad, 1 / (rawFps || outFps)),
    source: "ffmpeg",
  };
}
