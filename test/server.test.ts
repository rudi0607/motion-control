import { afterAll, beforeAll, describe, expect, it } from "vitest";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { frameStats, freezeIntervals, prepareReferenceVideo, probe, renderFinal, run } from "../server/media";
import { runServerChecks } from "../server/quality";
import { buildFalInput } from "../server/providers/fal";
import { buildReplicateInput } from "../server/providers/replicate";
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

describe("provider payloads", () => {
  it("Kling binds a facial element with orientation=video when identity lock is on", () => {
    const input = buildFalInput(getModel("kling-v3-pro")!, req({}, ["https://x/face.png"])) as any;
    expect(input.character_orientation).toBe("video");
    expect(input.elements).toEqual([{ frontal_image_url: "https://x/face.png", reference_image_urls: ["https://x/img.png"] }]);
    expect(input.prompt).toContain("@Element1");
    expect(input.prompt).toMatch(/Do not use the reference video person's face/);
    expect(input.keep_original_sound).toBe(true);
  });

  it("Kling camera motion uses orientation=image and drops element binding", () => {
    const input = buildFalInput(getModel("kling-v3-standard")!, req({ cameraMotionTransfer: true })) as any;
    expect(input.character_orientation).toBe("image");
    expect(input.elements).toBeUndefined();
    expect(input.prompt).not.toContain("@Element1");
  });

  it("Wan (fal) maps accuracy and seed", () => {
    const input = buildFalInput(getModel("wan-2.2-animate-fal")!, req({ seed: 42, motionAccuracy: "maximum" })) as any;
    expect(input).toMatchObject({ resolution: "720p", num_inference_steps: 30, video_quality: "maximum", seed: 42 });
    expect(input.prompt).toBeUndefined();
  });

  it("Self-hosted worker maps accuracy, seed and file ids", () => {
    const input = buildSelfhostedInput(req({ seed: 9, motionAccuracy: "maximum" })) as any;
    expect(input).toEqual({ image_file: "https://x/img.png", video_file: "https://x/ref.mp4", resolution: "720p", refert_num: 5, sample_steps: 30, seed: 9 });
    expect(buildSelfhostedInput(req({ motionAccuracy: "balanced" }))).toMatchObject({ refert_num: 1, sample_steps: 20 });
  });

  it("Wan (Replicate) generates natively at the requested fps", () => {
    const input = buildReplicateInput(req({ fps: 50, keepAudio: false, motionAccuracy: "balanced" })) as any;
    expect(input).toMatchObject({ frames_per_second: 50, refert_num: 1, go_fast: true, merge_audio: false, resolution: "720" });
  });
});

describe("validation", () => {
  const img = { width: 800, height: 1200 } as any;
  it("enforces Kling duration limits per orientation", () => {
    const m = getModel("kling-v3-pro")!;
    expect(validateInputs(m, parseSettings({}), img, { durationSec: 25, width: 1080, height: 1920 } as any).errors).toHaveLength(0);
    expect(validateInputs(m, parseSettings({ cameraMotionTransfer: true }), img, { durationSec: 25, width: 1080, height: 1920 } as any).errors[0]).toMatch(/at most 10 s/);
    const trimmed = validateInputs(m, parseSettings({ cameraMotionTransfer: true, trimToModelLimit: true }), img, { durationSec: 25, width: 1080, height: 1920 } as any);
    expect(trimmed.errors).toHaveLength(0);
    expect(trimmed.warnings.join(" ")).toMatch(/trimmed/);
  });
  it("rejects tiny or extreme-aspect character images", () => {
    const m = getModel("kling-v3-pro")!;
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
