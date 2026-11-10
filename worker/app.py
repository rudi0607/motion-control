"""Self-hosted motion-transfer GPU worker for Wan2.2-Animate-14B (animation mode).

Runs the official Wan2.2 preprocessing (pose/face extraction + retargeting) and
generate.py on a local GPU. Jobs run one at a time because a single 14B model
saturates one GPU.
"""

from __future__ import annotations

import hmac
import json
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    token: str = os.environ.get("WORKER_TOKEN", "")
    data_dir: Path = Path(os.environ.get("WORKER_DATA_DIR", "/data/jobs"))
    # "animate-14b": Wan2.2-Animate-14B (48–80 GB VRAM).
    # "vace-1.3b":  Wan2.1-VACE-1.3B pose-to-video (consumer GPUs, ~8–12 GB VRAM, 480p).
    backend: str = os.environ.get("WAN_BACKEND", "animate-14b")
    wan_repo: Path = Path(os.environ.get("WAN_REPO", "/opt/Wan2.2"))
    ckpt_dir: Path = Path(os.environ.get("WAN_CKPT_DIR", "/models/Wan2.2-Animate-14B"))
    wan21_repo: Path = Path(os.environ.get("WAN21_REPO", "/opt/Wan2.1"))
    vace_repo: Path = Path(os.environ.get("VACE_REPO", "/opt/VACE"))
    vace_ckpt_dir: Path = Path(os.environ.get("VACE_CKPT_DIR", "/models/Wan2.1-VACE-1.3B"))
    # Must be 4n+1. 81 frames = ~5 s per generated part at 16 fps; lower it if an 8 GB GPU runs out of memory.
    vace_frames: int = int(os.environ.get("VACE_FRAME_NUM", "81"))
    # Frames shared (and crossfaded) between consecutive parts of a long video.
    vace_overlap: int = int(os.environ.get("VACE_OVERLAP", "8"))
    facelock_models: Path = Path(os.environ.get("FACELOCK_MODELS_DIR", "/models/facelock"))
    facelock_script: Path = Path(os.environ.get("FACELOCK_SCRIPT", str(Path(__file__).parent / "facelock.py")))
    python: str = os.environ.get("WAN_PYTHON", "python")
    # Retargeting with FLUX Kontext adapts the reference skeleton to the character's body shape.
    use_flux: bool = env_bool("WAN_USE_FLUX", True)
    retarget: bool = env_bool("WAN_RETARGET", True)
    # Offload + dtype conversion + T5 on CPU for GPUs with < 80 GB. Always on for vace-1.3b.
    low_vram: bool = env_bool("WAN_LOW_VRAM", False)
    num_gpus: int = int(os.environ.get("WAN_NUM_GPUS", "1"))
    # Longer references are generated in parts and joined (vace) or natively (animate-14b).
    max_duration_sec: float = float(os.environ.get("WORKER_MAX_DURATION_SEC", "60"))
    timeout_sec: int = int(os.environ.get("WORKER_JOB_TIMEOUT_SEC", str(3 * 3600)))

    def effective_max_duration(self) -> float:
        return self.max_duration_sec


VACE_FPS = 16
FACELOCK_FILES = ("scrfd_2.5g.onnx", "arcface_w600k_r50.onnx", "inswapper_128.onnx", "gfpgan_1.4.onnx")
VACE_PROMPT = (
    "The person from the reference image performs the movements of the pose video. "
    "Keep the exact same face, identity, skin tone, hairstyle, clothing and body proportions as the reference image. "
    "Photorealistic, natural motion, stable lighting, real camera footage."
)
S = Settings()


class JobParams(BaseModel):
    image_file: str
    video_file: str
    backend: Optional[str] = None
    resolution: str = Field("720p", pattern="^(480p|720p)$")
    refert_num: int = Field(5, description="Temporal guidance frames: 1 or 5")
    sample_steps: int = Field(20, ge=4, le=60)
    seed: Optional[int] = None
    retarget: Optional[bool] = None
    # Re-apply the character's face identity to every generated frame.
    face_lock: bool = True
    # 0 = no face restoration, 1 = full GFPGAN (smooths skin). 0.6 keeps natural texture.
    face_restore: float = Field(0.6, ge=0, le=1)


@dataclass
class Job:
    id: str
    params: dict
    status: str = "queued"  # queued | preprocessing | generating | succeeded | failed | canceled
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None
    logs: list = field(default_factory=list)
    commands: list = field(default_factory=list)
    face_report: Optional[dict] = None


jobs: dict[str, Job] = {}
queue: deque[str] = deque()
lock = threading.Lock()
wake = threading.Event()
procs: dict[str, subprocess.Popen] = {}


def job_dir(job_id: str) -> Path:
    return S.data_dir / job_id


def files_dir() -> Path:
    return S.data_dir / "_files"


def persist(job: Job) -> None:
    d = job_dir(job.id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps(asdict(job)))


def build_commands(job: Job) -> list[list[str]]:
    p = JobParams(**job.params)
    d = job_dir(job.id)
    area = ["1280", "720"] if p.resolution == "720p" else ["832", "480"]
    pre = [
        S.python,
        str(S.wan_repo / "wan/modules/animate/preprocess/preprocess_data.py"),
        "--ckpt_path", str(S.ckpt_dir / "process_checkpoint"),
        "--video_path", str(d / "reference.mp4"),
        "--refer_path", str(d / "character.png"),
        "--save_path", str(d / "process_results"),
        "--resolution_area", *area,
        "--fps", "30",
    ]
    retarget = S.retarget if p.retarget is None else p.retarget
    if retarget:
        pre.append("--retarget_flag")
        if S.use_flux:
            pre.append("--use_flux")

    gen_args = [
        str(S.wan_repo / "generate.py"),
        "--task", "animate-14B",
        "--ckpt_dir", str(S.ckpt_dir),
        "--src_root_path", str(d / "process_results"),
        "--refert_num", str(p.refert_num),
        "--sample_steps", str(p.sample_steps),
        "--save_file", str(d / "output.mp4"),
    ]
    if p.seed is not None:
        gen_args += ["--base_seed", str(p.seed)]
    if S.low_vram:
        gen_args += ["--offload_model", "True", "--convert_model_dtype", "--t5_cpu"]
    if S.num_gpus > 1:
        gen = [S.python, "-m", "torch.distributed.run", "--nnodes", "1", "--nproc_per_node", str(S.num_gpus), *gen_args,
               "--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(S.num_gpus)]
    else:
        gen = [S.python, *gen_args]
    return [pre, gen]


def video_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    w, h = (int(x) for x in out.stdout.strip().split(",")[:2])
    return w, h


def build_vace_prepare(job: Job) -> list[str]:
    """Resample the reference to the model's 16 fps, capped at the worker's total duration limit."""
    d = job_dir(job.id)
    max_frames = int(S.max_duration_sec * VACE_FPS) + 1
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(d / "reference.mp4"),
            "-vf", f"fps={VACE_FPS}", "-frames:v", str(max_frames), "-an",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(d / "reference16.mp4")]


def build_vace_pose(job: Job) -> list[str]:
    d = job_dir(job.id)
    return [S.python, str(S.vace_repo / "vace/vace_preproccess.py"), "--task", "pose",
            "--video", str(d / "reference16.mp4"), "--pre_save_dir", str(d / "process_results")]


def build_vace_generate(job: Job, width: int, height: int, frames: int, pose: Path, out: Path) -> list[str]:
    p = JobParams(**job.params)
    d = job_dir(job.id)
    size = "832*480" if width >= height else "480*832"
    cmd = [
        S.python, str(S.wan21_repo / "generate.py"),
        "--task", "vace-1.3B",
        "--size", size,
        "--ckpt_dir", str(S.vace_ckpt_dir),
        "--src_video", str(pose),
        "--src_ref_images", str(d / "character.png"),
        "--frame_num", str(frames),
        "--sample_steps", str(p.sample_steps),
        "--prompt", VACE_PROMPT,
        "--save_file", str(out),
        # 8 GB cards need the DiT offloaded between steps and T5 on the CPU.
        "--offload_model", "True", "--t5_cpu",
    ]
    # Same seed for every chunk keeps appearance and lighting consistent across joins.
    cmd += ["--base_seed", str(p.seed if p.seed is not None else job_seed(job))]
    return cmd


def job_seed(job: Job) -> int:
    return int(job.id[:8], 16) % (2**31)


def count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=300,
    )
    return int(out.stdout.strip() or 0)


def plan_chunks(total: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """Frame ranges [start, end) covering `total` frames. Each length is 4n+1; consecutive chunks overlap."""
    if total < 5:
        raise RuntimeError(f"Reference video is too short after resampling to {VACE_FPS} fps ({total} frames)")
    if total <= chunk:
        n = ((total - 1) // 4) * 4 + 1
        return [(0, n)]
    ranges, start = [], 0
    step = chunk - overlap
    while True:
        end = start + chunk
        if end >= total:
            # Align the last chunk to the end; its overlap with the previous one just grows.
            ranges.append((total - chunk, total))
            break
        ranges.append((start, end))
        start += step
    return ranges


def cut_frames(src: Path, dst: Path, start: int, end: int) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src),
            "-vf", f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS", "-an",
            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(dst)]


def join_chunks(chunks: list[Path], ranges: list[tuple[int, int]], out: Path, fps: int = VACE_FPS) -> int:
    """Concatenate chunk videos, linearly crossfading the overlapping frames."""
    import cv2
    import numpy as np

    frames: dict[int, np.ndarray] = {}
    weights: dict[int, float] = {}
    size = None
    for idx, (path, (s, e)) in enumerate(zip(chunks, ranges)):
        cap = cv2.VideoCapture(str(path))
        prev_end = ranges[idx - 1][1] if idx else s
        ov = max(0, prev_end - s)
        i = 0
        while s + i < e:
            ok, f = cap.read()
            if not ok:
                break
            if size is None:
                size = (f.shape[1], f.shape[0])
            elif (f.shape[1], f.shape[0]) != size:
                f = cv2.resize(f, size)
            g = s + i
            # Ramp the new chunk in across the overlap so joins don't pop.
            w = (i + 1) / (ov + 1) if i < ov else 1.0
            if g in frames:
                frames[g] = (frames[g].astype(np.float32) * (1 - w) + f.astype(np.float32) * w)
                weights[g] = 1.0
            else:
                frames[g] = f.astype(np.float32)
            i += 1
        cap.release()
    if not frames or size is None:
        raise RuntimeError("No frames were generated")
    n = max(frames) + 1
    enc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE,
    )
    last = None
    for g in range(n):
        f = frames.get(g, last)
        last = f
        enc.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("Encoding the joined video failed")
    return n


def build_facelock(job: Job, p: "JobParams") -> list[str]:
    d = job_dir(job.id)
    cmd = [S.python, str(S.facelock_script),
           "--video", str(d / "output.mp4"), "--character", str(d / "character.png"),
           "--out", str(d / "output-locked.mp4"), "--models", str(S.facelock_models),
           "--report", str(d / "facelock.json"), "--restore-blend", str(p.face_restore)]
    if p.face_restore <= 0:
        cmd.append("--no-restore")
    return cmd


def run_step(job: Job, cmd: list[str], cwd: Optional[Path] = None) -> None:
    job.commands.append(shlex.join(cmd))
    cwd = cwd or S.wan_repo
    proc = subprocess.Popen(
        cmd, cwd=cwd if cwd.exists() else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True,
    )
    procs[job.id] = proc
    deadline = time.time() + S.timeout_sec
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            job.logs.append(line[-500:])
            del job.logs[:-300]
        if time.time() > deadline:
            os.killpg(proc.pid, signal.SIGKILL)
            raise RuntimeError(f"Step timed out after {S.timeout_sec}s")
    code = proc.wait()
    procs.pop(job.id, None)
    if job.status == "canceled":
        raise RuntimeError("canceled")
    if code != 0:
        tail = "\n".join(job.logs[-8:])
        raise RuntimeError(f"{Path(cmd[1]).name if len(cmd) > 1 and cmd[0] == S.python else cmd[0]} exited with code {code}:\n{tail}")


def process(job: Job) -> None:
    d = job_dir(job.id)
    job.started_at = time.time()
    try:
        job.status = "preprocessing"
        persist(job)
        if S.backend == "vace-1.3b":
            run_step(job, build_vace_prepare(job))
            total = count_frames(d / "reference16.mp4")
            ranges = plan_chunks(total, S.vace_frames, S.vace_overlap)
            # VACE annotators resolve models/VACE-Annotators relative to the VACE repo.
            run_step(job, build_vace_pose(job), cwd=S.vace_repo)
            pose = d / "process_results" / "src_video-pose.mp4"
            if not pose.exists():
                raise RuntimeError("Pose extraction produced no src_video-pose.mp4 (no person detected?)")
            w, h = video_size(d / "reference16.mp4")
            job.status = "generating"
            persist(job)
            chunks = []
            for i, (s, e) in enumerate(ranges):
                job.logs.append(f"Generating part {i + 1}/{len(ranges)} (frames {s}-{e - 1} of {total})")
                part_pose, part_out = d / f"pose_{i:03d}.mp4", d / f"part_{i:03d}.mp4"
                run_step(job, cut_frames(pose, part_pose, s, e))
                run_step(job, build_vace_generate(job, w, h, e - s, part_pose, part_out), cwd=S.wan21_repo)
                if not part_out.exists():
                    raise RuntimeError(f"generate.py wrote no output for part {i + 1}")
                chunks.append(part_out)
            if len(chunks) == 1:
                shutil.move(chunks[0], d / "output.mp4")
            else:
                job.logs.append(f"Joining {len(chunks)} parts with {S.vace_overlap}-frame crossfades")
                join_chunks(chunks, ranges, d / "output.mp4")
            for f in [*d.glob("pose_*.mp4"), *d.glob("part_*.mp4")]:
                f.unlink(missing_ok=True)
        else:
            pre, gen = build_commands(job)
            run_step(job, pre)
            job.status = "generating"
            persist(job)
            run_step(job, gen)
        if not (d / "output.mp4").exists():
            raise RuntimeError("generate.py finished but output.mp4 was not written")
        p = JobParams(**job.params)
        if p.face_lock:
            job.status = "face_lock"
            persist(job)
            run_step(job, build_facelock(job, p), cwd=Path(__file__).parent)
            locked = d / "output-locked.mp4"
            if not locked.exists():
                raise RuntimeError("Face lock produced no output")
            shutil.move(d / "output.mp4", d / "output-raw.mp4")
            shutil.move(locked, d / "output.mp4")
            report = d / "facelock.json"
            if report.exists():
                job.face_report = json.loads(report.read_text())
        job.status = "succeeded"
    except Exception as e:  # noqa: BLE001 - surfaced to the client
        if job.status != "canceled":
            job.status = "failed"
            job.error = str(e)
    finally:
        job.ended_at = time.time()
        persist(job)
        shutil.rmtree(d / "process_results", ignore_errors=True)


def worker_loop() -> None:
    while True:
        wake.wait()
        with lock:
            job_id = queue.popleft() if queue else None
            if not queue:
                wake.clear()
        if job_id and jobs[job_id].status == "queued":
            process(jobs[job_id])


def restore() -> None:
    S.data_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(S.data_dir.glob("*/job.json"), key=lambda x: x.stat().st_mtime):
        job = Job(**json.loads(p.read_text()))
        if job.status in {"preprocessing", "generating"}:
            job.status, job.error = "failed", "Worker restarted while this job was running"
            persist(job)
        jobs[job.id] = job
        if job.status == "queued":
            queue.append(job.id)
    if queue:
        wake.set()


def auth(authorization: str = Header(default="")) -> None:
    if not S.token:
        raise HTTPException(500, "WORKER_TOKEN is not configured on the worker")
    if not hmac.compare_digest(authorization.encode(), f"Bearer {S.token}".encode()):
        raise HTTPException(401, "Invalid worker token")


app = FastAPI(title="Motion Control GPU Worker")


@app.on_event("startup")
def _startup() -> None:
    restore()
    threading.Thread(target=worker_loop, daemon=True).start()


def gpu_info() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


@app.get("/v1/health")
def health(_: None = Depends(auth)):
    problems = []
    if S.backend == "vace-1.3b":
        if not (S.wan21_repo / "generate.py").exists():
            problems.append(f"Wan2.1 repo not found at {S.wan21_repo}")
        if not (S.vace_repo / "vace/vace_preproccess.py").exists():
            problems.append(f"VACE repo not found at {S.vace_repo}")
        if not (S.vace_repo / "models/VACE-Annotators/pose").exists():
            problems.append(f"VACE pose annotators not found at {S.vace_repo / 'models/VACE-Annotators/pose'}")
        if not (S.vace_ckpt_dir / "config.json").exists():
            problems.append(f"Wan2.1-VACE-1.3B weights not found at {S.vace_ckpt_dir}")
        if S.vace_overlap >= S.vace_frames - 4:
            problems.append(f"VACE_OVERLAP ({S.vace_overlap}) must be smaller than VACE_FRAME_NUM - 4")
        if (S.vace_frames - 1) % 4:
            problems.append(f"VACE_FRAME_NUM must be 4n+1 (got {S.vace_frames})")
    elif S.backend == "animate-14b":
        if not (S.wan_repo / "generate.py").exists():
            problems.append(f"Wan2.2 repo not found at {S.wan_repo}")
        if not (S.ckpt_dir / "process_checkpoint").exists():
            problems.append(f"Wan2.2-Animate-14B weights not found at {S.ckpt_dir}")
    else:
        problems.append(f"Unknown WAN_BACKEND {S.backend!r} (use animate-14b or vace-1.3b)")
    missing = [f for f in FACELOCK_FILES if not (S.facelock_models / f).exists()]
    if missing:
        problems.append(f"Face-lock models missing in {S.facelock_models}: {', '.join(missing)}")
    gpus = gpu_info()
    if not gpus:
        problems.append("No NVIDIA GPU visible (nvidia-smi failed)")
    return {
        "ok": not problems,
        "problems": problems,
        "gpus": gpus,
        "backend": S.backend,
        "queue_length": len(queue),
        "low_vram": S.low_vram or S.backend == "vace-1.3b",
        "max_duration_sec": S.effective_max_duration(),
    }


@app.post("/v1/files")
async def upload_file(file: UploadFile = File(...), _: None = Depends(auth)):
    files_dir().mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "").suffix.lower()[:8]
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm"}:
        raise HTTPException(400, f"Unsupported file type {ext!r}")
    fid = f"{uuid.uuid4().hex}{ext}"
    with (files_dir() / fid).open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"file_id": fid}


def _file(fid: str) -> Path:
    p = files_dir() / Path(fid).name
    if not p.exists():
        raise HTTPException(400, f"Unknown file id {fid}")
    return p


@app.post("/v1/jobs", status_code=201)
def create_job(params: JobParams, _: None = Depends(auth)):
    if params.refert_num not in (1, 5):
        raise HTTPException(422, "refert_num must be 1 or 5")
    if params.backend and params.backend != S.backend:
        raise HTTPException(409, f"This worker runs {S.backend!r}, not {params.backend!r}. Pick the matching model in the app.")
    img, vid = _file(params.image_file), _file(params.video_file)
    job = Job(id=uuid.uuid4().hex, params=params.model_dump())
    d = job_dir(job.id)
    d.mkdir(parents=True)
    shutil.move(img, d / "character.png")
    shutil.move(vid, d / "reference.mp4")
    with lock:
        jobs[job.id] = job
        queue.append(job.id)
        persist(job)
        wake.set()
    return _status(job.id)


def _status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    pos = list(queue).index(job_id) if job_id in queue else None
    return {
        "id": job.id,
        "status": job.status,
        "queue_position": pos,
        "error": job.error,
        "logs": job.logs[-100:],
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "face_report": job.face_report and {k: v for k, v in job.face_report.items() if k != "frames_without_face"},
    }


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, _: None = Depends(auth)):
    return _status(job_id)


@app.get("/v1/jobs/{job_id}/output")
def output(job_id: str, _: None = Depends(auth)):
    job = jobs.get(job_id)
    if not job or job.status != "succeeded":
        raise HTTPException(404, "Output not available")
    return FileResponse(job_dir(job_id) / "output.mp4", media_type="video/mp4")


@app.post("/v1/jobs/{job_id}/cancel")
def cancel(job_id: str, _: None = Depends(auth)):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status in {"queued", "preprocessing", "generating"}:
        job.status, job.error = "canceled", "Canceled by client"
        with lock:
            if job_id in queue:
                queue.remove(job_id)
        proc = procs.get(job_id)
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        persist(job)
    return _status(job.id)
