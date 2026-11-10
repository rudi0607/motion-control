import { FaceLandmarker, FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

const MODEL_BASE = "https://storage.googleapis.com/mediapipe-models";
const MODELS = {
  pose: `${MODEL_BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`,
  hand: `${MODEL_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
  face: `${MODEL_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`,
};

export interface Pt {
  x: number;
  y: number;
  z: number;
  v?: number;
}

/** Face landmark indices kept per frame (contours for drawing + geometry points). */
export const FACE_KEEP = [
  // Geometry anchors
  10, 152, 234, 454, 33, 133, 263, 362, 1, 168, 61, 291, 13, 14, 159, 145, 386, 374,
  // Contours for drawing
  127, 162, 21, 54, 103, 67, 109, 338, 297, 332, 284, 251, 389, 356, 323, 361, 288, 397, 365, 379, 378, 400, 377, 148, 176, 149, 150, 136, 172, 58, 132, 93,
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 160, 158, 153, 144, 387, 385, 380, 373, 0, 17, 78, 308, 81, 311, 178, 402, 4, 98, 327,
];

export interface FrameLandmarks {
  t: number;
  pose?: Pt[];
  /** 3D world landmarks in metres (hip-centred), robust to foreshortening. */
  poseWorld?: Pt[];
  hands: { side: string; pts: Pt[]; world?: Pt[] }[];
  /** Face landmarks keyed by original index (subset in FACE_KEEP). */
  face?: Record<number, Pt>;
  blend?: { blinkL: number; blinkR: number; jawOpen: number; smile: number };
}

export interface AnalysisResult {
  frames: FrameLandmarks[];
  durationSec: number;
  width: number;
  height: number;
  sampleFps: number;
}

type Mode = "IMAGE" | "VIDEO";
interface LandmarkerSet {
  pose: PoseLandmarker;
  hand: HandLandmarker;
  face: FaceLandmarker;
}
const sets: Partial<Record<Mode, Promise<LandmarkerSet>>> = {};

async function create(mode: Mode): Promise<LandmarkerSet> {
  const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
  const mk = async (delegate: "GPU" | "CPU"): Promise<LandmarkerSet> => {
    const [pose, hand, face] = await Promise.all([
      PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODELS.pose, delegate },
        runningMode: mode,
        numPoses: 1,
      }),
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODELS.hand, delegate },
        runningMode: mode,
        numHands: 2,
      }),
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODELS.face, delegate },
        runningMode: mode,
        numFaces: 1,
        outputFaceBlendshapes: true,
      }),
    ]);
    return { pose, hand, face };
  };
  try {
    return await mk("GPU");
  } catch {
    return await mk("CPU");
  }
}

function getSet(mode: Mode) {
  sets[mode] ??= create(mode).catch((e) => {
    delete sets[mode];
    throw new Error(`Could not load MediaPipe landmark models (${e?.message ?? e}). Landmark tracking needs access to storage.googleapis.com.`);
  });
  return sets[mode]!;
}

let videoTs = 0;

function detect(set: LandmarkerSet, src: HTMLCanvasElement, mode: Mode, t: number): FrameLandmarks {
  const ts = (videoTs += 50);
  const pose = mode === "VIDEO" ? set.pose.detectForVideo(src, ts) : set.pose.detect(src);
  const hand = mode === "VIDEO" ? set.hand.detectForVideo(src, ts) : set.hand.detect(src);
  const face = mode === "VIDEO" ? set.face.detectForVideo(src, ts) : set.face.detect(src);
  const out: FrameLandmarks = { t, hands: [] };
  if (pose.landmarks[0]) {
    out.pose = pose.landmarks[0].map((p) => ({ x: p.x, y: p.y, z: p.z, v: p.visibility }));
    out.poseWorld = pose.worldLandmarks[0]?.map((p) => ({ x: p.x, y: p.y, z: p.z, v: p.visibility }));
  }
  hand.landmarks.forEach((pts, i) => {
    out.hands.push({
      side: hand.handedness[i]?.[0]?.categoryName ?? "?",
      pts: pts.map((p) => ({ x: p.x, y: p.y, z: p.z })),
      world: hand.worldLandmarks[i]?.map((p) => ({ x: p.x, y: p.y, z: p.z })),
    });
  });
  const fl = face.faceLandmarks[0];
  if (fl) {
    out.face = {};
    for (const i of FACE_KEEP) out.face[i] = { x: fl[i].x, y: fl[i].y, z: fl[i].z };
    const bs = face.faceBlendshapes[0]?.categories ?? [];
    const g = (n: string) => bs.find((c) => c.categoryName === n)?.score ?? 0;
    out.blend = { blinkL: g("eyeBlinkLeft"), blinkR: g("eyeBlinkRight"), jawOpen: g("jawOpen"), smile: (g("mouthSmileLeft") + g("mouthSmileRight")) / 2 };
  }
  return out;
}

function toCanvas(src: CanvasImageSource, w: number, h: number, maxSide = 720) {
  const scale = Math.min(1, maxSide / Math.max(w, h));
  const c = document.createElement("canvas");
  c.width = Math.round(w * scale);
  c.height = Math.round(h * scale);
  c.getContext("2d")!.drawImage(src, 0, 0, c.width, c.height);
  return c;
}

export interface ImageAnalysis {
  frame: FrameLandmarks;
  width: number;
  height: number;
  faceBox?: { x: number; y: number; w: number; h: number };
  personAreaFraction?: number;
}

export async function analyzeImage(url: string): Promise<ImageAnalysis> {
  const img = new Image();
  img.src = url;
  await img.decode();
  const set = await getSet("IMAGE");
  const c = toCanvas(img, img.naturalWidth, img.naturalHeight, 1280);
  const frame = detect(set, c, "IMAGE", 0);
  const res: ImageAnalysis = { frame, width: img.naturalWidth, height: img.naturalHeight };
  if (frame.face) {
    const pts = Object.values(frame.face);
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const x = Math.min(...xs), y = Math.min(...ys);
    res.faceBox = { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
  }
  if (frame.pose) {
    const vis = frame.pose.filter((p) => (p.v ?? 1) > 0.5);
    if (vis.length) {
      const xs = vis.map((p) => p.x), ys = vis.map((p) => p.y);
      res.personAreaFraction = Math.min(1, (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys)) * 1.3);
    }
  }
  return res;
}

function seek(v: HTMLVideoElement, t: number) {
  return new Promise<void>((resolve, reject) => {
    const done = () => {
      v.removeEventListener("seeked", done);
      v.removeEventListener("error", fail);
      resolve();
    };
    const fail = () => reject(new Error("Video decode error while seeking"));
    v.addEventListener("seeked", done);
    v.addEventListener("error", fail);
    v.currentTime = t;
  });
}

export async function analyzeVideo(
  url: string,
  opts: { sampleFps?: number; maxDurationSec?: number; onProgress?: (fraction: number, frame: FrameLandmarks) => void; signal?: AbortSignal } = {},
): Promise<AnalysisResult> {
  const v = document.createElement("video");
  v.muted = true;
  v.playsInline = true;
  v.preload = "auto";
  v.crossOrigin = "anonymous";
  v.src = url;
  await new Promise<void>((res, rej) => {
    v.onloadeddata = () => res();
    v.onerror = () => rej(new Error("This browser cannot decode the video. Try an H.264 MP4."));
  });
  const set = await getSet("VIDEO");
  const sampleFps = opts.sampleFps ?? 12;
  const dur = Math.min(v.duration, opts.maxDurationSec ?? Infinity);
  const n = Math.max(1, Math.floor(dur * sampleFps));
  const frames: FrameLandmarks[] = [];
  for (let i = 0; i < n; i++) {
    if (opts.signal?.aborted) throw new DOMException("aborted", "AbortError");
    const t = Math.min(i / sampleFps, Math.max(0, v.duration - 0.001));
    await seek(v, t);
    const c = toCanvas(v, v.videoWidth, v.videoHeight);
    const f = detect(set, c, "VIDEO", t);
    frames.push(f);
    opts.onProgress?.((i + 1) / n, f);
    // Yield so the UI stays responsive.
    if (i % 4 === 0) await new Promise((r) => setTimeout(r, 0));
  }
  const res = { frames, durationSec: v.duration, width: v.videoWidth, height: v.videoHeight, sampleFps };
  v.removeAttribute("src");
  v.load();
  return res;
}

export function frameAt(a: AnalysisResult | null, t: number): FrameLandmarks | undefined {
  if (!a || !a.frames.length) return undefined;
  const i = Math.min(a.frames.length - 1, Math.max(0, Math.round(t * a.sampleFps)));
  return a.frames[i];
}
