"""Test stub for Wan2.2 generate.py: validates the CLI contract, writes the pose video as output."""
import argparse
import pathlib
import shutil
import sys

p = argparse.ArgumentParser()
for a in ["--task", "--ckpt_dir", "--src_root_path", "--refert_num", "--sample_steps", "--save_file", "--base_seed"]:
    p.add_argument(a)
a = p.parse_args()
if a.refert_num == "1" and a.sample_steps == "13":
    print("CUDA out of memory")
    sys.exit(3)
shutil.copy(pathlib.Path(a.src_root_path) / "src_pose.mp4", a.save_file)
print("generated", a.task, a.refert_num, a.sample_steps, a.base_seed)
