"""Face identity lock + face restoration for generated videos.

For every frame:
  1. detect the face (SCRFD) and its 5 keypoints, smoothed over time to avoid jitter;
  2. replace the identity with the character image's ArcFace identity (inswapper_128);
  3. optionally restore detail at 512 px (GFPGAN 1.4);
  4. feather-blend the face back into the frame.

It also measures ArcFace cosine similarity to the character before and after, so the
app can report real identity numbers instead of guessing.

Runs on onnxruntime (CUDA when available, otherwise CPU) and needs ~2 GB of VRAM.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
FFHQ_512 = np.array(
    [[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904], [0.39308822, 0.72541100], [0.61150205, 0.72490465]],
    dtype=np.float32,
) * 512

MODEL_FILES = {
    "detector": "scrfd_2.5g.onnx",
    "arcface": "arcface_w600k_r50.onnx",
    "swapper": "inswapper_128.onnx",
    "restorer": "gfpgan_1.4.onnx",
}


def providers() -> list[str]:
    avail = ort.get_available_providers()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]


def session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=providers())


def estimate_affine(kps: np.ndarray, dst: np.ndarray) -> np.ndarray:
    m, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
    return m


@dataclass
class Face:
    box: np.ndarray  # x1, y1, x2, y2
    kps: np.ndarray  # 5 x 2
    score: float


class Detector:
    """SCRFD face detector (insightface), decoded without the insightface package."""

    def __init__(self, path: Path, size: int = 640, thresh: float = 0.5):
        self.s = session(path)
        self.size = size
        self.thresh = thresh
        self.strides = [8, 16, 32]
        self.anchors = 2
        self._centers: dict = {}

    def detect(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.s.run(None, {self.s.get_inputs()[0].name: blob})
        n = len(self.strides)
        boxes, kpss, scores = [], [], []
        for i, stride in enumerate(self.strides):
            sc = outs[i].reshape(-1)
            bb = outs[i + n] * stride
            kp = outs[i + 2 * n] * stride
            fh = fw = self.size // stride
            key = (fh, fw, stride)
            if key not in self._centers:
                c = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
                c = (c * stride).reshape(-1, 2)
                self._centers[key] = np.repeat(c, self.anchors, axis=0)
            centers = self._centers[key]
            keep = np.where(sc >= self.thresh)[0]
            if not len(keep):
                continue
            c = centers[keep]
            b = bb[keep]
            boxes.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], -1))
            k = kp[keep].reshape(-1, 5, 2) + c[:, None, :]
            kpss.append(k)
            scores.append(sc[keep])
        if not boxes:
            return []
        boxes = np.concatenate(boxes) / scale
        kpss = np.concatenate(kpss) / scale
        scores = np.concatenate(scores)
        xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1).tolist()
        idx = cv2.dnn.NMSBoxes(xywh, scores.tolist(), self.thresh, 0.4)
        idx = np.array(idx).reshape(-1)
        faces = [Face(boxes[i], kpss[i], float(scores[i])) for i in idx]
        return sorted(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]), reverse=True)


class ArcFace:
    def __init__(self, path: Path):
        self.s = session(path)

    def embed(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        m = estimate_affine(kps, ARCFACE_DST)
        crop = cv2.warpAffine(img, m, (112, 112), borderValue=0)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        e = self.s.run(None, {self.s.get_inputs()[0].name: blob})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)


class Swapper:
    def __init__(self, path: Path):
        self.s = session(path)
        # The identity projection matrix is stored as the graph's last initializer.
        self.emap = numpy_helper.to_array(onnx.load(str(path)).graph.initializer[-1])
        # inswapper aligns to the ArcFace template shifted for a 128 px crop.
        self.dst = ARCFACE_DST.copy()
        self.dst[:, 0] += 8.0

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        lat = embedding.reshape(1, -1) @ self.emap
        return (lat / np.linalg.norm(lat)).astype(np.float32)

    def swap(self, img: np.ndarray, kps: np.ndarray, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = estimate_affine(kps, self.dst)
        crop = cv2.warpAffine(img, m, (128, 128), borderValue=0)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 255, (128, 128), (0, 0, 0), swapRB=True)
        out = self.s.run(None, {"target": blob, "source": latent})[0][0]
        face = (np.clip(out.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
        return face, m


class Restorer:
    def __init__(self, path: Path):
        self.s = session(path)

    def restore(self, img: np.ndarray, kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = estimate_affine(kps, FFHQ_512)
        crop = cv2.warpAffine(img, m, (512, 512), borderMode=cv2.BORDER_REFLECT)
        x = crop[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        y = self.s.run(None, {self.s.get_inputs()[0].name: x})[0][0]
        y = ((np.clip(y, -1, 1) + 1) / 2 * 255).transpose(1, 2, 0)[:, :, ::-1].astype(np.uint8)
        return y, m


def paste(frame: np.ndarray, face: np.ndarray, m: np.ndarray, erode: float = 0.08, feather: float = 0.12, strength: float = 1.0) -> np.ndarray:
    size = face.shape[0]
    inv = cv2.invertAffineTransform(m)
    h, w = frame.shape[:2]
    warped = cv2.warpAffine(face, inv, (w, h), borderValue=0)
    mask = np.zeros((size, size), dtype=np.float32)
    pad = int(size * erode)
    mask[pad : size - pad, pad : size - pad] = 1.0
    k = max(1, int(size * feather)) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = cv2.warpAffine(mask, inv, (w, h), borderValue=0)[:, :, None] * strength
    return (warped * mask + frame * (1 - mask)).astype(np.uint8)


class KpsSmoother:
    """Exponential smoothing of keypoints; resets on large jumps (cuts or a new face)."""

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.prev: Optional[np.ndarray] = None

    def __call__(self, kps: np.ndarray, face_size: float) -> np.ndarray:
        if self.prev is None or np.abs(kps - self.prev).max() > face_size * 0.25:
            self.prev = kps.copy()
        else:
            self.prev = self.alpha * kps + (1 - self.alpha) * self.prev
        return self.prev


def probe_video(path: Path) -> tuple[int, int, str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    return int(out[0]), int(out[1]), out[2]


def lock_video(video: Path, character: Path, out: Path, models_dir: Path, restore: bool = True, strength: float = 1.0, restore_blend: float = 0.6,
               report_path: Optional[Path] = None, progress_every: int = 25) -> dict:
    det = Detector(models_dir / MODEL_FILES["detector"])
    arc = ArcFace(models_dir / MODEL_FILES["arcface"])
    swp = Swapper(models_dir / MODEL_FILES["swapper"])
    rst = Restorer(models_dir / MODEL_FILES["restorer"]) if restore else None

    ref = cv2.imread(str(character))
    if ref is None:
        raise SystemExit(f"Cannot read character image {character}")
    ref_faces = det.detect(ref)
    if not ref_faces:
        raise SystemExit("No face found in the character image; face lock needs a clearly visible face.")
    ref_emb = arc.embed(ref, ref_faces[0].kps)
    latent = swp.latent(ref_emb)

    w, h, rate = probe_video(video)
    cap = cv2.VideoCapture(str(video))
    tmp = out.with_suffix(".noaudio.mp4")
    enc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", rate,
         "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", str(tmp)],
        stdin=subprocess.PIPE,
    )
    smooth = KpsSmoother()
    before, after, missing = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = det.detect(frame)
        if faces:
            f = faces[0]
            size = float(max(f.box[2] - f.box[0], f.box[3] - f.box[1]))
            kps = smooth(f.kps, size)
            before.append(float(arc.embed(frame, f.kps) @ ref_emb))
            swapped, m = swp.swap(frame, kps, latent)
            frame = paste(frame, swapped, m, strength=strength)
            if rst:
                restored, m2 = rst.restore(frame, kps)
                # Partial blend keeps natural skin texture; full GFPGAN looks airbrushed.
                frame = paste(frame, restored, m2, erode=0.1, feather=0.15, strength=restore_blend)
            after_faces = det.detect(frame)
            after.append(float(arc.embed(frame, after_faces[0].kps) @ ref_emb) if after_faces else float("nan"))
        else:
            smooth.prev = None
            missing.append(i)
        enc.stdin.write(frame.tobytes())
        i += 1
        if progress_every and i % progress_every == 0:
            print(f"face-lock frame {i}", flush=True)
    cap.release()
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg encode failed")
    # Copy the original audio track (if any) back on.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp), "-i", str(video), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:v", "copy", "-c:a", "copy", "-shortest", str(out)], check=True)
    tmp.unlink(missing_ok=True)

    def stats(xs):
        v = np.array([x for x in xs if not np.isnan(x)])
        return None if not len(v) else {"mean": round(float(v.mean()), 4), "min": round(float(v.min()), 4), "p10": round(float(np.percentile(v, 10)), 4)}

    report = {
        "frames": i,
        "frames_with_face": len(before),
        "frames_without_face": missing[:500],
        "similarity_before": stats(before),
        "similarity_after": stats(after),
        "per_frame_after": [round(x, 4) if not np.isnan(x) else None for x in after],
        "restore": restore,
        "restore_blend": restore_blend if restore else 0,
        "providers": providers(),
    }
    if report_path:
        report_path.write_text(json.dumps(report))
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--character", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--report", type=Path)
    p.add_argument("--no-restore", action="store_true")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--restore-blend", type=float, default=0.6, help="0 = no restoration, 1 = full GFPGAN")
    a = p.parse_args(argv)
    r = lock_video(a.video, a.character, a.out, a.models, restore=not a.no_restore, strength=a.strength,
                   restore_blend=a.restore_blend, report_path=a.report)
    print(json.dumps({k: v for k, v in r.items() if k != "per_frame_after"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
