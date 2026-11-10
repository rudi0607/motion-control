"""Test stub for VACE vace_preproccess.py --task pose: validates the CLI contract."""
import argparse
import pathlib
import shutil

p = argparse.ArgumentParser()
for a in ["--task", "--video", "--pre_save_dir"]:
    p.add_argument(a)
a = p.parse_args()
assert a.task == "pose", a.task
assert pathlib.Path("models/VACE-Annotators/pose").exists(), "annotators must resolve relative to the VACE repo"
out = pathlib.Path(a.pre_save_dir)
out.mkdir(parents=True, exist_ok=True)
shutil.copy(a.video, out / "src_video-pose.mp4")
print("pose ok")
