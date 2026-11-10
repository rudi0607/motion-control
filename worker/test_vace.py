"""VACE-1.3B (low-VRAM) backend tests using stub scripts plus real ffmpeg resampling."""

import importlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

STUBS = Path(__file__).parent / "test_stubs"
H = {"Authorization": "Bearer secret"}


@pytest.fixture()
def vace_client(tmp_path, monkeypatch):
    wan21 = tmp_path / "Wan2.1"
    vace = tmp_path / "VACE"
    (vace / "vace").mkdir(parents=True)
    (vace / "models/VACE-Annotators/pose").mkdir(parents=True)
    wan21.mkdir()
    shutil.copy(STUBS / "vace_preproccess.py", vace / "vace/vace_preproccess.py")
    shutil.copy(STUBS / "wan21_generate.py", wan21 / "generate.py")
    (tmp_path / "vace-ckpt").mkdir()
    (tmp_path / "vace-ckpt/config.json").write_text("{}")
    monkeypatch.setenv("WORKER_TOKEN", "secret")
    monkeypatch.setenv("WAN_BACKEND", "vace-1.3b")
    monkeypatch.setenv("WAN21_REPO", str(wan21))
    monkeypatch.setenv("VACE_REPO", str(vace))
    monkeypatch.setenv("VACE_CKPT_DIR", str(tmp_path / "vace-ckpt"))
    monkeypatch.setenv("VACE_FRAME_NUM", "49")
    monkeypatch.setenv("WORKER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WAN_PYTHON", sys.executable)
    sys.path.insert(0, str(Path(__file__).parent))
    import app as worker

    worker = importlib.reload(worker)
    from fastapi.testclient import TestClient

    with TestClient(worker.app) as c:
        yield c, worker, tmp_path


def upload(c, name, data):
    r = c.post("/v1/files", files={"file": (name, data)}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["file_id"]


def wait(c, jid, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = c.get(f"/v1/jobs/{jid}", headers=H).json()
        if s["status"] in {"succeeded", "failed", "canceled"}:
            return s
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_vace_backend_resamples_extracts_pose_and_generates(vace_client):
    c, _, tmp = vace_client
    vid_path = tmp / "ref.mp4"
    # 5 s portrait clip at 30 fps, resampled to 16 fps and capped at 49 frames.
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=480x640:rate=30:duration=5",
                    "-pix_fmt", "yuv420p", str(vid_path)], check=True)
    img = upload(c, "char.png", b"IMG")
    vid = upload(c, "ref.mp4", vid_path.read_bytes())
    h = c.get("/v1/health", headers=H).json()
    assert h["backend"] == "vace-1.3b" and h["max_duration_sec"] == 3.0 and h["low_vram"] is True

    r = c.post("/v1/jobs", json={"image_file": img, "video_file": vid, "backend": "animate-14b"}, headers=H)
    assert r.status_code == 409

    r = c.post("/v1/jobs", json={"image_file": img, "video_file": vid, "backend": "vace-1.3b", "sample_steps": 25, "seed": 3}, headers=H)
    assert r.status_code == 201, r.text
    s = wait(c, r.json()["id"])
    assert s["status"] == "succeeded", s
    assert "pose ok" in s["logs"]
    assert "vace generated 480*832 49 25 True True 3" in s["logs"], s["logs"]
    out = c.get(f"/v1/jobs/{s['id']}/output", headers=H)
    assert out.status_code == 200 and len(out.content) > 1000


def test_vace_short_clip_rounds_frames_to_4n_plus_1(vace_client):
    _, worker, tmp = vace_client
    p = tmp / "short.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=16:duration=2",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    assert worker.vace_frame_count(p) == 29  # 32 frames -> largest 4n+1 <= 32


def test_vace_rejects_bad_frame_count(vace_client):
    c, worker, _ = vace_client
    worker.S.vace_frames = 50
    assert any("4n+1" in p for p in c.get("/v1/health", headers=H).json()["problems"])
