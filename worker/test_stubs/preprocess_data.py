"""Test stub for Wan2.2 preprocess_data.py: validates the CLI contract, copies the video."""
import argparse
import pathlib
import shutil

p = argparse.ArgumentParser()
for a in ["--ckpt_path", "--video_path", "--refer_path", "--save_path", "--fps"]:
    p.add_argument(a)
p.add_argument("--resolution_area", nargs=2)
p.add_argument("--retarget_flag", action="store_true")
p.add_argument("--use_flux", action="store_true")
a = p.parse_args()
out = pathlib.Path(a.save_path)
out.mkdir(parents=True)
shutil.copy(a.video_path, out / "src_pose.mp4")
print("preprocess ok", a.resolution_area, a.retarget_flag, a.use_flux)
