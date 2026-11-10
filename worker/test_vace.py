"""VACE-1.3B (low-VRAM) backend tests using stub scripts plus real ffmpeg resampling."""

import importlib
import os
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


def test_vace_long_video_is_generated_in_parts_joined_and_face_locked(vace_client):
    c, worker, tmp = vace_client
    vid_path = tmp / "ref.mp4"
    # 5 s at 30 fps = 80 frames at 16 fps -> parts of 49 frames with 8-frame overlap: [0,49) [31,80).
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=480x640:rate=30:duration=5",
                    "-pix_fmt", "yuv420p", str(vid_path)], check=True)
    img = upload(c, "char.png", b"IMG")
    vid = upload(c, "ref.mp4", vid_path.read_bytes())
    h = c.get("/v1/health", headers=H).json()
    assert h["backend"] == "vace-1.3b" and h["max_duration_sec"] == 60 and h["low_vram"] is True

    r = c.post("/v1/jobs", json={"image_file": img, "video_file": vid, "backend": "animate-14b"}, headers=H)
    assert r.status_code == 409

    r = c.post("/v1/jobs", json={"image_file": img, "video_file": vid, "backend": "vace-1.3b", "sample_steps": 25, "seed": 3,
                                 "face_restore": 0.5}, headers=H)
    assert r.status_code == 201, r.text
    s = wait(c, r.json()["id"])
    assert s["status"] == "succeeded", s
    assert "pose ok" in s["logs"]
    assert "Generating part 1/2 (frames 0-48 of 80)" in s["logs"]
    assert "Generating part 2/2 (frames 31-79 of 80)" in s["logs"]
    assert s["logs"].count("vace generated 480*832 49 25 True True 3") == 2, s["logs"]
    assert "facelock ok 0.5 False" in s["logs"]
    assert s["face_report"]["similarity_after"]["mean"] == 0.79
    out = tmp / "out.mp4"
    out.write_bytes(c.get(f"/v1/jobs/{s['id']}/output", headers=H).content)
    assert worker.count_frames(out) == 80


def test_plan_chunks_covers_every_frame_with_valid_lengths():
    import app as worker

    for total in (5, 30, 81, 82, 150, 961):
        ranges = worker.plan_chunks(total, 81, 8)
        assert ranges[0][0] == 0 and ranges[-1][1] <= total
        assert total - ranges[-1][1] < 4  # at most 3 trailing frames dropped (4n+1 rounding)
        for (s, e) in ranges:
            assert (e - s - 1) % 4 == 0
        for (a, b) in zip(ranges, ranges[1:]):
            assert b[0] < a[1], "consecutive parts must overlap"
    with pytest.raises(RuntimeError):
        worker.plan_chunks(3, 81, 8)


def test_join_chunks_crossfades_overlap(tmp_path):
    import app as worker

    # Part A is black, part B is white; overlap frames must ramp between them.
    a, b, out = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "j.mp4"
    for path, color in ((a, "black"), (b, "white")):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:size=64x64:rate=16",
                        "-frames:v", "13", "-pix_fmt", "yuv420p", str(path)], check=True)
    n = worker.join_chunks([a, b], [(0, 13), (9, 22)], out)
    assert n == 22
    import cv2

    cap = cv2.VideoCapture(str(out))
    means = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        means.append(float(f.mean()))
    assert len(means) == 22
    assert means[0] < 30 and means[-1] > 225
    overlap = means[9:13]
    assert overlap == sorted(overlap) and 30 < overlap[1] < 225


@pytest.mark.skipif(not os.environ.get("FACELOCK_TEST_MODELS"), reason="set FACELOCK_TEST_MODELS to a dir with the real ONNX models")
def test_real_facelock_increases_identity_similarity(tmp_path):
    """Runs the real models on CPU: a different person's face must be replaced by the character's."""
    import facelock

    media = Path(os.environ.get("FACELOCK_TEST_MEDIA", "/tmp/fl/media"))
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(media / "target-540p.mp4"), "-frames:v", "6", str(clip)], check=True)
    r = facelock.lock_video(clip, media / "source.jpg", tmp_path / "locked.mp4", Path(os.environ["FACELOCK_TEST_MODELS"]), restore_blend=0.6)
    assert r["frames_with_face"] == 6
    assert r["similarity_before"]["mean"] < 0.2
    assert r["similarity_after"]["min"] > 0.5


def test_vace_rejects_bad_frame_count(vace_client):
    c, worker, _ = vace_client
    worker.S.vace_frames = 50
    assert any("4n+1" in p for p in c.get("/v1/health", headers=H).json()["problems"])
