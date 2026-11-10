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
    wan_repo: Path = Path(os.environ.get("WAN_REPO", "/opt/Wan2.2"))
    ckpt_dir: Path = Path(os.environ.get("WAN_CKPT_DIR", "/models/Wan2.2-Animate-14B"))
    python: str = os.environ.get("WAN_PYTHON", "python")
    # Retargeting with FLUX Kontext adapts the reference skeleton to the character's body shape.
    use_flux: bool = env_bool("WAN_USE_FLUX", True)
    retarget: bool = env_bool("WAN_RETARGET", True)
    # Offload + dtype conversion + T5 on CPU for GPUs with < 80 GB.
    low_vram: bool = env_bool("WAN_LOW_VRAM", False)
    num_gpus: int = int(os.environ.get("WAN_NUM_GPUS", "1"))
    max_duration_sec: float = float(os.environ.get("WORKER_MAX_DURATION_SEC", "30"))
    timeout_sec: int = int(os.environ.get("WORKER_JOB_TIMEOUT_SEC", str(3 * 3600)))


S = Settings()


class JobParams(BaseModel):
    image_file: str
    video_file: str
    resolution: str = Field("720p", pattern="^(480p|720p)$")
    refert_num: int = Field(5, description="Temporal guidance frames: 1 or 5")
    sample_steps: int = Field(20, ge=4, le=60)
    seed: Optional[int] = None
    retarget: Optional[bool] = None


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


def run_step(job: Job, cmd: list[str]) -> None:
    job.commands.append(shlex.join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=S.wan_repo if S.wan_repo.exists() else None,
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
        raise RuntimeError(f"{Path(cmd[1]).name} exited with code {code}:\n{tail}")


def process(job: Job) -> None:
    d = job_dir(job.id)
    job.started_at = time.time()
    try:
        pre, gen = build_commands(job)
        job.status = "preprocessing"
        persist(job)
        run_step(job, pre)
        job.status = "generating"
        persist(job)
        run_step(job, gen)
        if not (d / "output.mp4").exists():
            raise RuntimeError("generate.py finished but output.mp4 was not written")
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
    if not (S.wan_repo / "generate.py").exists():
        problems.append(f"Wan2.2 repo not found at {S.wan_repo}")
    if not (S.ckpt_dir / "process_checkpoint").exists():
        problems.append(f"Wan2.2-Animate-14B weights not found at {S.ckpt_dir}")
    gpus = gpu_info()
    if not gpus:
        problems.append("No NVIDIA GPU visible (nvidia-smi failed)")
    return {
        "ok": not problems,
        "problems": problems,
        "gpus": gpus,
        "queue_length": len(queue),
        "low_vram": S.low_vram,
        "max_duration_sec": S.max_duration_sec,
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
