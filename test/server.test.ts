import { afterAll, beforeAll, describe, expect, it } from "vitest";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { frameStats, freezeIntervals, prepareReferenceVideo, probe, renderFinal, run } from "../server/media";
import { runServerChecks } from "../server/quality";
import { buildSelfhostedInput } from "../server/providers/selfhosted";
import { getModel } from "../server/providers/index";
import { parseSettings } from "../server/settings";
import { validateInputs } from "../server/jobs";
import type { GenerateRequest } from "../server/providers/types";

let dir: string;
const f = (n: string) => path.join(dir, n);
const ff = (...args: string[]) => run("ffmpeg", ["-hide_banner", "-y", "-loglevel", "error", ...args]);

beforeAll(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "mc-test-"));
  // Moving test pattern with audio: stands in for a reference / model output.
  await ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=3", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", f("ref.mov"));
  // Same clip with strong alternating brightness on every other frame (flicker).
  await ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=3", "-vf", "eq=brightness='if(mod(n,2),0.15,0)':eval=frame", "-c:v", "libx264", "-pix_fmt", "yuv420p", f("flicker.mp4"));
  await ff("-f", "lavfi", "-i", "color=c=gray:size=400x600", "-frames:v", "1", f("char.png"));
});
afterAll(async () => {
  await fs.rm(dir, { recursive: true, force: true });
});

const req = (over: Partial<ReturnType<typeof parseSettings>> = {}, identity: string[] = []): GenerateRequest => ({
  imageUrl: "https://x/img.png",
  videoUrl: "https://x/ref.mp4",
  identityImageUrls: identity,
  prompt: "",
  settings: parseSettings({ ...over }),
  referenceFps: 24,
});

describe("settings", () => {
  it("applies defaults and rejects invalid values", () => {
    const s = parseSettings(JSON.stringify({ fps: "60", motionStrength: 400 }));
    expect(s.fps).toBe(60);
    expect(s.motionStrength).toBe(100);
    expect(s.identityLock).toBe(true);
    expect(() => parseSettings({ resolution: "8k" })).toThrow();
    expect(() => parseSettings({ fps: 25 })).toThrow();
  });
});

describe("worker payloads", () => {
  it("VACE 1.3B (8 GB GPUs) sends backend, 480p and step counts", () => {
    const m = getModel("wan-2.1-vace-selfhosted")!;
    expect(buildSelfhostedInput(m, req({ seed: 9, motionAccuracy: "maximum" }))).toEqual({
      image_file: "https://x/img.png",
      video_file: "https://x/ref.mp4",
      backend: "vace-1.3b",
      resolution: "480p",
      refert_num: 5,
      sample_steps: 40,
      seed: 9,
    });
    expect(buildSelfhostedInput(m, req({ motionAccuracy: "balanced" }))).toMatchObject({ sample_steps: 25 });
  });

  it("Animate 14B maps accuracy to refert_num and steps", () => {
    const m = getModel("wan-2.2-animate-selfhosted")!;
    expect(buildSelfhostedInput(m, req({ motionAccuracy: "maximum" }))).toMatchObject({ backend: "animate-14b", resolution: "720p", refert_num: 5, sample_steps: 30 });
    expect(buildSelfhostedInput(m, req({ motionAccuracy: "balanced" }))).toMatchObject({ refert_num: 1, sample_steps: 20 });
  });
});

describe("validation", () => {
  const img = { width: 800, height: 1200 } as any;
  it("enforces the VACE clip limit and allows trimming", () => {
    const m = getModel("wan-2.1-vace-selfhosted")!;
    const long = { durationSec: 12, width: 1080, height: 1920 } as any;
    expect(validateInputs(m, parseSettings({}), img, { ...long, durationSec: 4 }).errors).toHaveLength(0);
    expect(validateInputs(m, parseSettings({}), img, long).errors[0]).toMatch(/at most 5 s/);
    const trimmed = validateInputs(m, parseSettings({ trimToModelLimit: true }), img, long);
    expect(trimmed.errors).toHaveLength(0);
    expect(trimmed.warnings.join(" ")).toMatch(/trimmed/);
  });
  it("rejects tiny or extreme-aspect character images", () => {
    const m = getModel("wan-2.2-animate-selfhosted")!;
    const v = { durationSec: 5, width: 1280, height: 720 } as any;
    expect(validateInputs(m, parseSettings({}), { width: 200, height: 200 } as any, v).errors.length).toBeGreaterThan(0);
    expect(validateInputs(m, parseSettings({}), { width: 3000, height: 400 } as any, v).errors.length).toBeGreaterThan(0);
  });
});

describe("media pipeline", () => {
  it("normalises, trims, renders to target fps/resolution with reference audio, and passes QC", async () => {
    await prepareReferenceVideo(f("ref.mov"), f("ref.mp4"), { maxDurationSec: 2.5 });
    const ref = await probe(f("ref.mp4"));
    expect(ref.videoCodec).toBe("h264");
    expect(ref.durationSec).toBeCloseTo(2.5, 1);

    // Simulate a model output that is silent, shorter and lower-res than requested.
    await ff("-i", f("ref.mp4"), "-t", "2.2", "-an", "-vf", "scale=480:270", "-c:v", "libx264", "-pix_fmt", "yuv420p", f("raw.mp4"));
    await ff("-i", f("ref.mp4"), "-vn", "-c:a", "aac", f("audio.m4a"));
    const raw = await probe(f("raw.mp4"));
    const notes = await renderFinal(f("raw.mp4"), f("out.mp4"), raw, {
      targetShortEdge: 720,
      fps: 30,
      motionInterpolation: true,
      deflicker: false,
      durationSec: ref.durationSec,
      audioFile: f("audio.m4a"),
      keepProviderAudio: false,
    });
    const out = await probe(f("out.mp4"));
    expect(out.height).toBe(720);
    expect(out.fps).toBeCloseTo(30, 0);
    expect(out.hasAudio).toBe(true);
    expect(Math.abs(out.durationSec - ref.durationSec)).toBeLessThan(0.1);
    expect(notes.join(" ")).toMatch(/motion-compensated/);
    expect(notes.join(" ")).toMatch(/held the last frame/);

    const [os_, rs] = await Promise.all([frameStats(f("out.mp4")), frameStats(f("ref.mp4"))]);
    expect(os_.length).toBeGreaterThan(60);
    const checks = runServerChecks({
      output: out,
      outputStats: os_,
      outputFreezes: await freezeIntervals(f("out.mp4")),
      reference: ref,
      referenceStats: rs,
      referenceFreezes: await freezeIntervals(f("ref.mp4")),
      expectAudio: true,
    });
    const by = Object.fromEntries(checks.map((c) => [c.id, c.status]));
    expect(by.duration).toBe("pass");
    expect(by.av_sync).toBe("pass");
    expect(by.flicker).toBe("pass");
  });

  it("keeps the full reference duration when interpolating 16 fps up to 30 fps", async () => {
    await ff("-f", "lavfi", "-i", "testsrc2=size=480x832:rate=16:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", f("vace16.mp4"));
    const raw = await probe(f("vace16.mp4"));
    await renderFinal(f("vace16.mp4"), f("vace30.mp4"), raw, { targetShortEdge: 720, fps: 30, motionInterpolation: true, deflicker: false, durationSec: 4, keepProviderAudio: false });
    const out = await probe(f("vace30.mp4"));
    expect(out.fps).toBeCloseTo(30, 0);
    expect(Math.abs(out.durationSec - 4)).toBeLessThan(0.05);
  });

  it("flags flicker and the deflicker re-render reduces it", async () => {
    const ref = await probe(f("ref.mov"));
    const raw = await probe(f("flicker.mp4"));
    const rs = await frameStats(f("ref.mov"));
    const qc = async (file: string) =>
      runServerChecks({ output: await probe(file), outputStats: await frameStats(file), outputFreezes: [], reference: ref, referenceStats: rs, referenceFreezes: [], expectAudio: false }).find((c) => c.id === "flicker")!;
    const before = await qc(f("flicker.mp4"));
    expect(before.status).toBe("fail");
    await renderFinal(f("flicker.mp4"), f("deflickered.mp4"), raw, { targetShortEdge: 360, fps: 24, motionInterpolation: false, deflicker: true, durationSec: raw.durationSec, keepProviderAudio: false });
    const after = await qc(f("deflickered.mp4"));
    const rate = (c: typeof before) => Number(c.value!.split("%")[0]);
    expect(rate(after)).toBeLessThan(rate(before));
  });
});
