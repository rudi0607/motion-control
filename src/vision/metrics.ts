import type { QualityCheck } from "../../server/quality";
import type { Preflight } from "../../server/jobs";
import type { AnalysisResult, FrameLandmarks, ImageAnalysis, Pt } from "./landmarks";

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const std = (xs: number[]) => {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
};
const median = (xs: number[]) => {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const d2 = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
const d3 = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

function corr(a: number[], b: number[]): number | null {
  const n = Math.min(a.length, b.length);
  if (n < 6) return null;
  const x = a.slice(0, n), y = b.slice(0, n);
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
    syy += (y[i] - my) ** 2;
  }
  if (sxx < 1e-9 || syy < 1e-9) return null;
  return sxy / Math.sqrt(sxx * syy);
}

function ranges(ts: number[], step: number) {
  const out: { start: number; end: number }[] = [];
  for (const t of [...ts].sort((a, b) => a - b)) {
    const l = out[out.length - 1];
    if (l && t - l.end <= step * 2.5) l.end = t + step;
    else out.push({ start: t, end: t + step });
  }
  return out;
}

// --- Pose helpers -----------------------------------------------------------
const P = { lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16, lHip: 23, rHip: 24, lKn: 25, rKn: 26, lAn: 27, rAn: 28, nose: 0 };
const BONES: [string, number, number][] = [
  ["L upper arm", P.lSh, P.lEl],
  ["R upper arm", P.rSh, P.rEl],
  ["L forearm", P.lEl, P.lWr],
  ["R forearm", P.rEl, P.rWr],
  ["L thigh", P.lHip, P.lKn],
  ["R thigh", P.rHip, P.rKn],
  ["L shin", P.lKn, P.lAn],
  ["R shin", P.rKn, P.rAn],
];
const JOINTS: [string, number, number, number][] = [
  ["L elbow", P.lSh, P.lEl, P.lWr],
  ["R elbow", P.rSh, P.rEl, P.rWr],
  ["L shoulder", P.lHip, P.lSh, P.lEl],
  ["R shoulder", P.rHip, P.rSh, P.rEl],
  ["L knee", P.lHip, P.lKn, P.lAn],
  ["R knee", P.rHip, P.rKn, P.rAn],
  ["L hip", P.lSh, P.lHip, P.lKn],
  ["R hip", P.rSh, P.rHip, P.rKn],
];
const visible = (f: FrameLandmarks, ...idx: number[]) => !!f.pose && idx.every((i) => (f.pose![i].v ?? 0) > 0.6);

function angle(a: Pt, b: Pt, c: Pt) {
  const v1 = [a.x - b.x, a.y - b.y, a.z - b.z];
  const v2 = [c.x - b.x, c.y - b.y, c.z - b.z];
  const dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  const n = Math.hypot(...v1) * Math.hypot(...v2);
  return n ? Math.acos(Math.max(-1, Math.min(1, dot / n))) : 0;
}

// --- Face helpers -----------------------------------------------------------
const F = { top: 10, chin: 152, lCheek: 234, rCheek: 454, lOut: 33, lIn: 133, rOut: 263, rIn: 362, noseTip: 1, bridge: 168, mL: 61, mR: 291, lipTop: 13, lipBot: 14 };

/** Scale-invariant facial proportions; only meaningful for near-frontal, neutral-mouth frames. */
export function faceSignature(f: FrameLandmarks["face"]): number[] | null {
  if (!f) return null;
  const W = d2(f[F.lCheek], f[F.rCheek]);
  const H = d2(f[F.top], f[F.chin]);
  if (W < 1e-4 || H < 1e-4) return null;
  return [
    d2(f[F.lOut], f[F.rOut]) / W,
    d2(f[F.lIn], f[F.rIn]) / W,
    d2(f[F.lOut], f[F.lIn]) / W,
    d2(f[F.rOut], f[F.rIn]) / W,
    d2(f[F.mL], f[F.mR]) / W,
    d2(f[F.bridge], f[F.noseTip]) / H,
    d2(f[F.noseTip], f[F.lipTop]) / H,
    d2(f[F.lipBot], f[F.chin]) / H,
    W / H,
  ];
}

/** Rough yaw proxy: where the nose tip sits between the cheeks (0.5 = frontal). */
function frontalness(f: FrameLandmarks["face"]) {
  if (!f) return 1;
  const l = f[F.lCheek].x, r = f[F.rCheek].x;
  return Math.abs((f[F.noseTip].x - l) / (r - l || 1) - 0.5);
}

const usableForIdentity = (fr: FrameLandmarks) => !!fr.face && frontalness(fr.face) < 0.08 && (fr.blend?.jawOpen ?? 0) < 0.25;

function faceNormalized(f: NonNullable<FrameLandmarks["face"]>): number[] {
  const keys = [F.lOut, F.lIn, F.rOut, F.rIn, F.noseTip, F.bridge, F.mL, F.mR, F.chin, F.top];
  const cx = mean(keys.map((k) => f[k].x)), cy = mean(keys.map((k) => f[k].y));
  const s = d2(f[F.lCheek], f[F.rCheek]) || 1;
  return keys.flatMap((k) => [(f[k].x - cx) / s, (f[k].y - cy) / s]);
}

/** High-frequency landmark jitter: mean magnitude of the second temporal difference. */
function faceJitter(frames: FrameLandmarks[]) {
  const vals: { t: number; j: number }[] = [];
  for (let i = 1; i < frames.length - 1; i++) {
    const [a, b, c] = [frames[i - 1].face, frames[i].face, frames[i + 1].face];
    if (!a || !b || !c) continue;
    const [na, nb, nc] = [faceNormalized(a), faceNormalized(b), faceNormalized(c)];
    vals.push({ t: frames[i].t, j: mean(na.map((_, k) => Math.abs(na[k] - 2 * nb[k] + nc[k]))) });
  }
  return vals;
}

function blinkCount(frames: FrameLandmarks[]) {
  let n = 0;
  let closed = false;
  for (const f of frames) {
    const b = f.blend ? (f.blend.blinkL + f.blend.blinkR) / 2 : 0;
    if (!closed && b > 0.5) {
      n++;
      closed = true;
    } else if (closed && b < 0.3) closed = false;
  }
  return n;
}

const fullBody = (f: FrameLandmarks) => visible(f, P.nose, P.lSh, P.rSh, P.lHip, P.rHip, P.lKn, P.rKn);

export function summarizePreflight(image: ImageAnalysis | null, video: AnalysisResult | null): Preflight {
  const pf: Preflight = {};
  if (image)
    pf.image = {
      faceDetected: !!image.frame.face,
      faceBox: image.faceBox,
      poseDetected: !!image.frame.pose,
      personAreaFraction: image.personAreaFraction,
    };
  if (video) {
    const fr = video.frames;
    const n = Math.max(1, fr.length);
    pf.video = {
      framesAnalyzed: fr.length,
      poseDetectionRate: fr.filter((f) => f.pose).length / n,
      faceDetectionRate: fr.filter((f) => f.face).length / n,
      handDetectionRate: fr.filter((f) => f.hands.length).length / n,
      fullBodyRate: fr.filter(fullBody).length / n,
      blinkCount: blinkCount(fr),
    };
  }
  return pf;
}

export interface PreflightIssue {
  level: "error" | "warn" | "ok";
  text: string;
}

export function preflightIssues(image: ImageAnalysis | null, pf: Preflight): PreflightIssue[] {
  const out: PreflightIssue[] = [];
  if (pf.image) {
    if (!pf.image.faceDetected) out.push({ level: "warn", text: "No face found in the character image — identity cannot be locked or verified." });
    else out.push({ level: "ok", text: "Character face detected and used as identity reference." });
    if (!pf.image.poseDetected) out.push({ level: "warn", text: "No body pose found in the character image; models need a clearly visible person." });
    else if ((pf.image.personAreaFraction ?? 1) < 0.05) out.push({ level: "warn", text: "Character occupies less than 5% of the image; Kling recommends a larger subject." });
    if (image && pf.image.faceBox && pf.image.faceBox.w * image.width < 80) out.push({ level: "warn", text: "Character face is small in the image; fine facial identity may be harder to preserve." });
  }
  if (pf.video) {
    const v = pf.video;
    if (v.poseDetectionRate < 0.6) out.push({ level: "warn", text: `A person is visible in only ${(v.poseDetectionRate * 100).toFixed(0)}% of reference frames.` });
    else out.push({ level: "ok", text: `Body pose tracked in ${(v.poseDetectionRate * 100).toFixed(0)}% of reference frames.` });
    if (v.faceDetectionRate < 0.5) out.push({ level: "warn", text: "Face rarely visible in the reference; facial expression and gaze transfer will be limited." });
  }
  return out;
}

/** Compares the generated video against the reference and the character image. */
export function visionChecks(image: ImageAnalysis | null, ref: AnalysisResult, out: AnalysisResult): QualityCheck[] {
  const checks: QualityCheck[] = [];
  const step = 1 / out.sampleFps;
  const n = Math.min(ref.frames.length, out.frames.length);
  const of = out.frames.slice(0, n);
  const rf = ref.frames.slice(0, n);
  const src = "vision" as const;

  // Identity: facial proportion consistency against the character image.
  const srcSig = image?.frame.face && frontalness(image.frame.face) < 0.12 ? faceSignature(image.frame.face) : null;
  const usable = of.filter(usableForIdentity);
  if (!srcSig) {
    checks.push({ id: "identity", label: "Face identity consistency", status: "skipped", detail: "Character image has no clear frontal face to compare against.", source: src });
  } else if (usable.length < 3) {
    checks.push({ id: "identity", label: "Face identity consistency", status: "skipped", detail: `Only ${usable.length} near-frontal output frame(s) with a neutral mouth; not enough to compare.`, source: src });
  } else {
    const devs = usable.map((f) => {
      const s = faceSignature(f.face)!;
      return { t: f.t, d: mean(s.map((v, i) => Math.abs(v - srcSig[i]) / srcSig[i])) };
    });
    const md = median(devs.map((d) => d.d));
    const bad = devs.filter((d) => d.d > 0.14).map((d) => d.t);
    const third = Math.max(1, Math.floor(devs.length / 3));
    const drift = mean(devs.slice(-third).map((d) => d.d)) - mean(devs.slice(0, third).map((d) => d.d));
    checks.push({
      id: "identity",
      label: "Face identity consistency",
      status: md < 0.07 && bad.length / devs.length < 0.1 ? "pass" : md < 0.12 ? "warn" : "fail",
      value: `${((1 - md) * 100).toFixed(1)}% geometric match`,
      detail: `Facial proportions (eye spacing, nose, mouth, jaw ratios) in ${devs.length} near-frontal frames vs the character image; drift first→last third ${(drift * 100).toFixed(1)} pts. Geometric proxy, not face recognition — it cannot detect texture or skin-tone changes.`,
      ranges: ranges(bad, step),
      source: src,
    });
  }

  // Facial landmark stability.
  const jo = faceJitter(of), jr = faceJitter(rf);
  if (jo.length < 5) {
    checks.push({ id: "face_stability", label: "Facial landmark stability", status: "skipped", detail: "Face not tracked in enough consecutive output frames.", source: src });
  } else {
    const mo = median(jo.map((x) => x.j)), mr = median(jr.map((x) => x.j)) || mo;
    const thr = Math.max(mr * 3, mo * 4, 0.02);
    const spikes = jo.filter((x) => x.j > thr).map((x) => x.t);
    const ratio = mo / (mr || 1e-6);
    checks.push({
      id: "face_stability",
      label: "Facial landmark stability",
      status: ratio < 1.6 && spikes.length <= 2 ? "pass" : ratio < 2.5 ? "warn" : "fail",
      value: `jitter ×${ratio.toFixed(2)} vs reference`,
      detail: "High-frequency shaking of facial landmarks (face flicker/morphing indicator), normalised by face size and compared with the reference.",
      ranges: ranges(spikes, step),
      source: src,
    });
  }

  // Hands.
  const handFrames = of.filter((f) => f.hands.length);
  if (!handFrames.length) {
    checks.push({ id: "hands", label: "Hand & finger integrity", status: rf.some((f) => f.hands.length) ? "warn" : "skipped", detail: rf.some((f) => f.hands.length) ? "Hands are visible in the reference but were not detected in the output." : "No hands visible.", source: src });
  } else {
    const badT: number[] = [];
    for (const f of handFrames) {
      for (const h of f.hands) {
        const p = h.world ?? h.pts;
        const palm = d3(p[0], p[9]) || 1e-6;
        const fingers = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
        const implausible = fingers.some((fg) => {
          const seg = [d3(p[fg[0]], p[fg[1]]), d3(p[fg[1]], p[fg[2]]), d3(p[fg[2]], p[fg[3]])];
          const total = seg[0] + seg[1] + seg[2];
          return total / palm > 2.6 || seg.some((s) => s / palm > 1.3);
        });
        if (implausible) badT.push(f.t);
      }
    }
    const rate = badT.length / handFrames.length;
    checks.push({
      id: "hands",
      label: "Hand & finger integrity",
      status: rate < 0.05 ? "pass" : rate < 0.15 ? "warn" : "fail",
      value: `${(rate * 100).toFixed(1)}% of hand frames implausible`,
      detail: `Finger segment lengths vs palm size in ${handFrames.length} frames. The landmarker always fits 21 points per hand, so extra or missing fingers cannot be counted directly — review flagged ranges visually.`,
      ranges: ranges(badT, step),
      source: src,
    });
  }

  // Body anatomy: 3D bone length stability.
  const boneDev: number[] = [];
  const boneBadT: number[] = [];
  for (const [, a, b] of BONES) {
    const lens = of.filter((f) => f.poseWorld && visible(f, a, b)).map((f) => ({ t: f.t, l: d3(f.poseWorld![a], f.poseWorld![b]) }));
    if (lens.length < 5) continue;
    const m = median(lens.map((x) => x.l));
    for (const x of lens) {
      const dev = Math.abs(x.l - m) / (m || 1);
      boneDev.push(dev);
      if (dev > 0.3) boneBadT.push(x.t);
    }
  }
  if (!boneDev.length) {
    checks.push({ id: "anatomy", label: "Body proportions & joints", status: "skipped", detail: "Limbs not tracked in enough output frames.", source: src });
  } else {
    const rate = new Set(boneBadT).size / Math.max(1, of.filter((f) => f.pose).length);
    checks.push({
      id: "anatomy",
      label: "Body proportions & joints",
      status: rate < 0.05 ? "pass" : rate < 0.15 ? "warn" : "fail",
      value: `median bone-length deviation ${(median(boneDev) * 100).toFixed(1)}%`,
      detail: "3D limb lengths (upper arm, forearm, thigh, shin) should stay constant; frames where a bone stretches/shrinks >30% are flagged as possible deformation.",
      ranges: ranges([...new Set(boneBadT)], step),
      source: src,
    });
  }

  // Motion fidelity: joint-angle correlation.
  const cors: { name: string; r: number }[] = [];
  for (const [name, a, b, c] of JOINTS) {
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < n; i++) {
      const fo = of[i], fr = rf[i];
      if (fo.poseWorld && fr.poseWorld && visible(fo, a, b, c) && visible(fr, a, b, c)) {
        xs.push(angle(fr.poseWorld[a], fr.poseWorld[b], fr.poseWorld[c]));
        ys.push(angle(fo.poseWorld[a], fo.poseWorld[b], fo.poseWorld[c]));
      }
    }
    if (xs.length >= 8 && std(xs) > 0.12) {
      const r = corr(xs, ys);
      if (r !== null) cors.push({ name, r });
    }
  }
  if (!cors.length) {
    checks.push({ id: "motion_fidelity", label: "Motion accuracy vs reference", status: "skipped", detail: "Not enough joint movement visible in both videos to compare.", source: src });
  } else {
    const m = mean(cors.map((c) => c.r));
    const worst = [...cors].sort((a, b) => a.r - b.r)[0];
    checks.push({
      id: "motion_fidelity",
      label: "Motion accuracy vs reference",
      status: m > 0.7 ? "pass" : m > 0.4 ? "warn" : "fail",
      value: `r = ${m.toFixed(2)} across ${cors.length} joint(s)`,
      detail: `Correlation of joint angles over time between reference and output. Weakest: ${worst.name} (r = ${worst.r.toFixed(2)}).`,
      source: src,
    });
  }

  // Blinking.
  const rb = blinkCount(rf), ob = blinkCount(of);
  const faceOut = of.filter((f) => f.face).length;
  checks.push({
    id: "blink",
    label: "Natural blinking",
    status: faceOut < 5 ? "skipped" : rb === 0 ? "pass" : ob === 0 ? "warn" : Math.abs(ob - rb) <= Math.max(1, rb * 0.4) ? "pass" : "warn",
    value: `${ob} blink(s) (reference ${rb})`,
    detail: "Eye-blink events detected from face blendshapes in sampled frames.",
    source: src,
  });

  // Mouth / lip motion vs reference.
  const jr2: number[] = [], jo2: number[] = [];
  for (let i = 0; i < n; i++) if (rf[i].blend && of[i].blend) {
    jr2.push(rf[i].blend!.jawOpen);
    jo2.push(of[i].blend!.jawOpen);
  }
  const rj = std(jr2) > 0.03 ? corr(jr2, jo2) : null;
  checks.push({
    id: "lip",
    label: "Mouth & lip motion vs reference",
    status: rj === null ? "skipped" : rj > 0.6 ? "pass" : rj > 0.3 ? "warn" : "fail",
    value: rj === null ? undefined : `r = ${rj.toFixed(2)}`,
    detail: "Correlation of mouth opening over time with the reference performer. Lip-sync to audio is inherited from reference motion; it is not measured against the audio waveform.",
    source: src,
  });

  // Frozen expression.
  const so = std(jo2) + std(of.filter((f) => f.blend).map((f) => f.blend!.smile));
  const sr = std(jr2) + std(rf.filter((f) => f.blend).map((f) => f.blend!.smile));
  checks.push({
    id: "expression",
    label: "Facial expression liveliness",
    status: sr < 0.04 || jo2.length < 5 ? "skipped" : so > sr * 0.4 ? "pass" : "warn",
    value: jo2.length >= 5 ? `${((so / (sr || 1)) * 100).toFixed(0)}% of reference variation` : undefined,
    detail: "Detects frozen faces: how much the output's expressions vary compared with the reference.",
    source: src,
  });

  return checks;
}
