"""Test stub for facelock.py: validates the CLI contract, copies the video, writes a report."""
import argparse
import json
import pathlib
import shutil

p = argparse.ArgumentParser()
for a in ["--video", "--character", "--out", "--models", "--report", "--restore-blend", "--strength"]:
    p.add_argument(a)
p.add_argument("--no-restore", action="store_true")
a = p.parse_args()
assert pathlib.Path(a.character).exists()
shutil.copy(a.video, a.out)
pathlib.Path(a.report).write_text(json.dumps({
    "frames": 10, "frames_with_face": 10, "frames_without_face": [],
    "similarity_before": {"mean": 0.31, "min": 0.2, "p10": 0.25},
    "similarity_after": {"mean": 0.79, "min": 0.74, "p10": 0.76},
    "per_frame_after": [0.79] * 10, "restore": not a.no_restore, "restore_blend": float(a.restore_blend),
}))
print("facelock ok", a.restore_blend, a.no_restore)
