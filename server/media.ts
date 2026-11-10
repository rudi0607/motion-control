import { spawn } from "node:child_process";

export interface ProcResult {
  code: number;
  stdout: string;
  stderr: string;
}

export function run(cmd: string, args: string[], opts: { timeoutMs?: number } = {}): Promise<ProcResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => {
      stderr += d;
      if (stderr.length > 4_000_000) stderr = stderr.slice(-2_000_000);
    });
    const timer = opts.timeoutMs ? setTimeout(() => child.kill("SIGKILL"), opts.timeoutMs) : undefined;
    child.on("error", reject);
    child.on("close", (code) => {
      if (timer) clearTimeout(timer);
      resolve({ code: code ?? -1, stdout, stderr });
    });
  });
}

async function ffmpeg(args: string[], timeoutMs = 60 * 60_000): Promise<ProcResult> {
  const r = await run("ffmpeg", ["-hide_banner", "-nostdin", "-y", ...args], { timeoutMs });
  if (r.code !== 0) {
    const tail = r.stderr.trim().split("\n").slice(-6).join("\n");
    throw new Error(`ffmpeg failed (exit ${r.code}): ${tail}`);
  }
  return r;
}

export interface MediaInfo {
  width: number;
  height: number;
  durationSec: number;
  fps: number;
  frameCount: number | null;
  videoCodec: string | null;
  hasAudio: boolean;
  audioCodec: string | null;
  audioDurationSec: number | null;
  videoStartSec: number;
  audioStartSec: number | null;
  formatName: string;
  sizeBytes: number;
  rotation: number;
}

function parseRate(rate: string | undefined): number {
  if (!rate) return 0;
  const [n, d] = rate.split("/").map(Number);
  if (!d) return n || 0;
  return n / d;
}

export async function probe(file: string): Promise<MediaInfo> {
  const r = await run("ffprobe", ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", file]);
  if (r.code !== 0) throw new Error(`Unreadable media file: ${r.stderr.trim().split("\n").pop() ?? "ffprobe failed"}`);
  const j = JSON.parse(r.stdout);
  const v = (j.streams ?? []).find((s: any) => s.codec_type === "video");
  const a = (j.streams ?? []).find((s: any) => s.codec_type === "audio");
  if (!v) throw new Error("No video or image stream found in file");
  const rotation = Number(
    v.tags?.rotate ?? v.side_data_list?.find((s: any) => s.rotation !== undefined)?.rotation ?? 0,
  );
  let width = Number(v.width);
  let height = Number(v.height);
  if (Math.abs(rotation) % 180 === 90) [width, height] = [height, width];
  const fps = parseRate(v.avg_frame_rate) || parseRate(v.r_frame_rate);
  const durationSec = Number(v.duration ?? j.format?.duration ?? 0) || 0;
  return {
    width,
    height,
    durationSec,
    fps,
    frameCount: v.nb_frames ? Number(v.nb_frames) : null,
    videoCodec: v.codec_name ?? null,
    hasAudio: Boolean(a),
    audioCodec: a?.codec_name ?? null,
    audioDurationSec: a ? Number(a.duration ?? j.format?.duration ?? 0) : null,
    videoStartSec: Number(v.start_time ?? 0) || 0,
    audioStartSec: a ? Number(a.start_time ?? 0) || 0 : null,
    formatName: j.format?.format_name ?? "",
    sizeBytes: Number(j.format?.size ?? 0),
    rotation,
  };
}

/**
 * Normalise the reference video into an H.264/AAC MP4 that every provider accepts,
 * optionally trimming it to the model's maximum duration.
 */
export async function prepareReferenceVideo(input: string, output: string, opts: { maxDurationSec?: number }) {
  const args = ["-i", input];
  if (opts.maxDurationSec) args.push("-t", String(opts.maxDurationSec));
  args.push(
    "-map", "0:v:0", "-map", "0:a:0?",
    // Cap the long edge at 1920 (no upscaling) and force even dimensions for yuv420p.
    "-vf", "scale='if(gte(iw,ih),min(1920,iw),-2)':'if(gte(iw,ih),-2,min(1920,ih))',scale=trunc(iw/2)*2:trunc(ih/2)*2",
    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k",
    "-movflags", "+faststart",
    output,
  );
  await ffmpeg(args);
}

/** Convert any still image (webp, png, jpeg, bmp, ...) to PNG. */
export async function prepareImage(input: string, output: string) {
  await ffmpeg(["-i", input, "-frames:v", "1", "-update", "1", output]);
}

export async function extractAudio(input: string, output: string) {
  await ffmpeg(["-i", input, "-vn", "-c:a", "aac", "-b:a", "192k", output]);
}

export const RESOLUTION_SHORT_EDGE: Record<string, number> = { "720p": 720, "1080p": 1080, "2k": 1440, "4k": 2160 };

export interface RenderOptions {
  /** Target short-edge size (720/1080/1440/2160). */
  targetShortEdge: number;
  fps: number;
  /** Motion-compensated frame interpolation when raising the frame rate. */
  motionInterpolation: boolean;
  deflicker: boolean;
  /** Reference duration to match (trims, or holds the last frame). */
  durationSec: number;
  /** Optional audio track (reference audio) to mux in, replacing provider audio. */
  audioFile?: string;
  keepProviderAudio: boolean;
}

export function buildRenderFilter(src: MediaInfo, o: RenderOptions): { filter: string; notes: string[] } {
  const notes: string[] = [];
  const filters: string[] = [];
  if (o.deflicker) {
    filters.push("deflicker=size=5:mode=pm");
    notes.push("Applied temporal luminance deflicker (ffmpeg deflicker, 5-frame window).");
  }
  if (o.fps > src.fps + 0.5 && o.motionInterpolation) {
    filters.push(`minterpolate=fps=${o.fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1`);
    notes.push(`Raised frame rate ${src.fps.toFixed(2)} → ${o.fps} fps with motion-compensated interpolation.`);
  } else if (Math.abs(o.fps - src.fps) > 0.01) {
    filters.push(`fps=${o.fps}`);
    notes.push(
      o.fps > src.fps
        ? `Raised frame rate ${src.fps.toFixed(2)} → ${o.fps} fps by frame duplication (enable Motion Smoothing for interpolated frames).`
        : `Converted frame rate ${src.fps.toFixed(2)} → ${o.fps} fps.`,
    );
  }
  const srcShort = Math.min(src.width, src.height);
  const landscape = src.width >= src.height;
  if (o.targetShortEdge !== srcShort) {
    filters.push(landscape ? `scale=-2:${o.targetShortEdge}:flags=lanczos` : `scale=${o.targetShortEdge}:-2:flags=lanczos`);
    notes.push(
      o.targetShortEdge > srcShort
        ? `Upscaled from native ${src.width}×${src.height} with Lanczos resampling. This adds pixels, not generated detail.`
        : `Downscaled from native ${src.width}×${src.height}.`,
    );
  }
  if (o.durationSec > src.durationSec + 0.05) {
    const pad = o.durationSec - src.durationSec;
    filters.push(`tpad=stop_mode=clone:stop_duration=${pad.toFixed(3)}`);
    notes.push(`Model output was ${pad.toFixed(2)} s shorter than the reference; held the last frame to match duration.`);
  }
  filters.push("format=yuv420p");
  return { filter: filters.join(","), notes };
}

export async function renderFinal(input: string, output: string, src: MediaInfo, o: RenderOptions) {
  const { filter, notes } = buildRenderFilter(src, o);
  const args = ["-i", input];
  if (o.audioFile) args.push("-i", o.audioFile);
  args.push("-map", "0:v:0");
  if (o.audioFile) args.push("-map", "1:a:0");
  else if (o.keepProviderAudio && src.hasAudio) args.push("-map", "0:a:0");
  args.push(
    "-vf", filter,
    "-t", o.durationSec.toFixed(3),
    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high",
    "-r", String(o.fps),
    "-c:a", "aac", "-b:a", "192k",
    "-movflags", "+faststart",
    output,
  );
  await ffmpeg(args);
  return notes;
}

export interface FrameStat {
  t: number;
  /** Mean luma of the frame. */
  yavg: number;
  /** Mean absolute luma difference to the previous frame. */
  ydif: number;
}

/** Per-frame luma statistics via ffmpeg signalstats. */
export async function frameStats(file: string): Promise<FrameStat[]> {
  const r = await run("ffmpeg", [
    "-hide_banner", "-nostdin", "-i", file,
    "-map", "0:v:0", "-vf", "scale=-2:360,signalstats,metadata=mode=print:file=-",
    "-f", "null", "-",
  ]);
  if (r.code !== 0) throw new Error(`signalstats failed: ${r.stderr.trim().split("\n").pop()}`);
  const out: FrameStat[] = [];
  let cur: FrameStat | null = null;
  for (const line of r.stdout.split("\n")) {
    const fm = line.match(/pts_time:([-\d.]+)/);
    if (fm) {
      cur = { t: Number(fm[1]), yavg: 0, ydif: 0 };
      out.push(cur);
      continue;
    }
    if (!cur) continue;
    const kv = line.match(/lavfi\.signalstats\.(YAVG|YDIF)=([-\d.]+)/);
    if (kv) cur[kv[1] === "YAVG" ? "yavg" : "ydif"] = Number(kv[2]);
  }
  return out;
}

export async function freezeIntervals(file: string, minDurationSec = 0.6): Promise<{ start: number; end: number }[]> {
  const r = await run("ffmpeg", [
    "-hide_banner", "-nostdin", "-i", file, "-map", "0:v:0",
    "-vf", `freezedetect=n=0.002:d=${minDurationSec}`, "-f", "null", "-",
  ]);
  const out: { start: number; end: number }[] = [];
  let start: number | null = null;
  for (const line of r.stderr.split("\n")) {
    const s = line.match(/freeze_start: ([\d.]+)/);
    const e = line.match(/freeze_end: ([\d.]+)/);
    if (s) start = Number(s[1]);
    if (e && start !== null) {
      out.push({ start, end: Number(e[1]) });
      start = null;
    }
  }
  if (start !== null) out.push({ start, end: Number.POSITIVE_INFINITY });
  return out;
}
