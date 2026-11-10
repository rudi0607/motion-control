"""API, queue and command-wiring tests using stub Wan scripts (no GPU required)."""

import importlib
import shutil
import sys
import time
from pathlib import Path

import pytest

STUBS = Path(__file__).parent / "test_stubs"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    repo = tmp_path / "Wan2.2"
    (repo / "wan/modules/animate/preprocess").mkdir(parents=True)
    shutil.copy(STUBS / "preprocess_data.py", repo / "wan/modules/animate/preprocess/preprocess_data.py")
    shutil.copy(STUBS / "generate.py", repo / "generate.py")
    (tmp_path / "ckpt/process_checkpoint").mkdir(parents=True)
    monkeypatch.setenv("WORKER_TOKEN", "secret")
    monkeypatch.setenv("WAN_REPO", str(repo))
    monkeypatch.setenv("WAN_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORKER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WAN_PYTHON", sys.executable)
    monkeypatch.setenv("FACELOCK_SCRIPT", str(STUBS / "facelock.py"))
    fl = tmp_path / "facelock-models"
    fl.mkdir()
    for name in ("scrfd_2.5g.onnx", "arcface_w600k_r50.onnx", "inswapper_128.onnx", "gfpgan_1.4.onnx"):
        (fl / name).write_bytes(b"")
    monkeypatch.setenv("FACELOCK_MODELS_DIR", str(fl))
    sys.path.insert(0, str(Path(__file__).parent))
    import app as worker

    worker = importlib.reload(worker)
    from fastapi.testclient import TestClient

    with TestClient(worker.app) as c:
        yield c, worker


H = {"Authorization": "Bearer secret"}


def upload(c, name, data=b"x"):
    r = c.post("/v1/files", files={"file": (name, data)}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["file_id"]


def wait(c, jid, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        s = c.get(f"/v1/jobs/{jid}", headers=H).json()
        if s["status"] in {"succeeded", "failed", "canceled"}:
            return s
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_requires_token(client):
    c, _ = client
    assert c.get("/v1/health").status_code == 401
    assert c.get("/v1/jobs/abc", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/v1/files", files={"file": ("a.png", b"x")}).status_code == 401


def test_rejects_bad_files_and_params(client):
    c, _ = client
    assert c.post("/v1/files", files={"file": ("a.exe", b"x")}, headers=H).status_code == 400
    img = upload(c, "a.png")
    vid = upload(c, "a.mp4")
    r = c.post("/v1/jobs", json={"image_file": img, "video_file": vid, "refert_num": 3}, headers=H)
    assert r.status_code == 422
    r = c.post("/v1/jobs", json={"image_file": "missing.png", "video_file": vid}, headers=H)
    assert r.status_code == 400


def test_full_job_runs_preprocess_then_generate(client):
    c, worker = client
    img = upload(c, "char.png", b"IMG")
    vid = upload(c, "ref.mp4", b"VIDEO-BYTES")
    r = c.post("/v1/jobs", json={"image_file": img, "video_file": vid, "refert_num": 5, "sample_steps": 30, "seed": 7}, headers=H)
    assert r.status_code == 201
    s = wait(c, r.json()["id"])
    assert s["status"] == "succeeded", s
    assert any("preprocess ok ['1280', '720'] True True" in l for l in s["logs"])
    assert any("generated animate-14B 5 30 7" in l for l in s["logs"])
    out = c.get(f"/v1/jobs/{s['id']}/output", headers=H)
    assert out.status_code == 200 and out.content == b"VIDEO-BYTES"


def test_failure_surfaces_model_error(client):
    c, _ = client
    img, vid = upload(c, "c.png"), upload(c, "r.mp4")
    r = c.post("/v1/jobs", json={"image_file": img, "video_file": vid, "refert_num": 1, "sample_steps": 13}, headers=H)
    s = wait(c, r.json()["id"])
    assert s["status"] == "failed"
    assert "exited with code 3" in s["error"] and "CUDA out of memory" in s["error"]
    assert c.get(f"/v1/jobs/{s['id']}/output", headers=H).status_code == 404


def test_low_vram_and_multi_gpu_flags(client, monkeypatch):
    _, worker = client
    worker.S.low_vram = True
    worker.S.num_gpus = 2
    job = worker.Job(id="x", params={"image_file": "a", "video_file": "b", "resolution": "480p"})
    pre, gen = worker.build_commands(job)
    assert pre[pre.index("--resolution_area") + 1 : pre.index("--resolution_area") + 3] == ["832", "480"]
    assert "torch.distributed.run" in gen and "--ulysses_size" in gen
    assert gen[gen.index("--offload_model") + 1] == "True" and "--t5_cpu" in gen


def test_health_reports_missing_gpu(client):
    c, _ = client
    h = c.get("/v1/health", headers=H).json()
    assert h["ok"] is False
    assert any("GPU" in p for p in h["problems"])
