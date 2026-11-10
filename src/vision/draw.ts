import { FACE_KEEP, type FrameLandmarks } from "./landmarks";

const POSE_EDGES: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32],
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22], [0, 2], [0, 5], [2, 7], [5, 8], [9, 10],
];
const HAND_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10];
const FACE_LINES = [
  FACE_OUTLINE,
  [33, 160, 158, 133, 153, 144, 33],
  [263, 387, 385, 362, 380, 373, 263],
  [70, 63, 105, 66, 107],
  [336, 296, 334, 293, 300],
  [61, 81, 13, 311, 291, 402, 14, 178, 61],
  [168, 4, 1],
];

export interface Layers {
  body: boolean;
  hands: boolean;
  face: boolean;
}

/** Draw landmarks for a frame onto a canvas covering a media element rendered with object-fit: contain. */
export function drawLandmarks(ctx: CanvasRenderingContext2D, f: FrameLandmarks | undefined, box: { x: number; y: number; w: number; h: number }, layers: Layers) {
  if (!f) return;
  const X = (x: number) => box.x + x * box.w;
  const Y = (y: number) => box.y + y * box.h;
  ctx.lineCap = "round";
  if (layers.body && f.pose) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(124,92,255,0.9)";
    for (const [a, b] of POSE_EDGES) {
      const p = f.pose[a], q = f.pose[b];
      if ((p.v ?? 1) < 0.5 || (q.v ?? 1) < 0.5) continue;
      ctx.beginPath();
      ctx.moveTo(X(p.x), Y(p.y));
      ctx.lineTo(X(q.x), Y(q.y));
      ctx.stroke();
    }
    ctx.fillStyle = "#fff";
    f.pose.forEach((p, i) => {
      if (i > 10 && (p.v ?? 1) > 0.5) {
        ctx.beginPath();
        ctx.arc(X(p.x), Y(p.y), 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }
  if (layers.hands) {
    for (const h of f.hands) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = h.side === "Left" ? "rgba(34,211,238,0.95)" : "rgba(250,204,21,0.95)";
      for (const [a, b] of HAND_EDGES) {
        ctx.beginPath();
        ctx.moveTo(X(h.pts[a].x), Y(h.pts[a].y));
        ctx.lineTo(X(h.pts[b].x), Y(h.pts[b].y));
        ctx.stroke();
      }
      ctx.fillStyle = ctx.strokeStyle;
      for (const p of h.pts) {
        ctx.beginPath();
        ctx.arc(X(p.x), Y(p.y), 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  if (layers.face && f.face) {
    ctx.lineWidth = 1.3;
    ctx.strokeStyle = "rgba(52,211,153,0.9)";
    for (const line of FACE_LINES) {
      ctx.beginPath();
      line.forEach((i, k) => {
        const p = f.face![i];
        if (!p) return;
        if (k === 0) ctx.moveTo(X(p.x), Y(p.y));
        else ctx.lineTo(X(p.x), Y(p.y));
      });
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(52,211,153,0.7)";
    for (const i of FACE_KEEP) {
      const p = f.face[i];
      if (!p) continue;
      ctx.fillRect(X(p.x) - 0.75, Y(p.y) - 0.75, 1.5, 1.5);
    }
  }
}

/** Content rectangle of a media element displayed with object-fit: contain. */
export function containBox(elW: number, elH: number, mediaW: number, mediaH: number) {
  if (!mediaW || !mediaH) return { x: 0, y: 0, w: elW, h: elH };
  const s = Math.min(elW / mediaW, elH / mediaH);
  const w = mediaW * s, h = mediaH * s;
  return { x: (elW - w) / 2, y: (elH - h) / 2, w, h };
}
