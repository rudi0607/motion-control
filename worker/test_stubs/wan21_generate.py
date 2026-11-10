"""Test stub for Wan2.1 generate.py --task vace-1.3B: validates the CLI contract."""
import argparse
import pathlib
import shutil

p = argparse.ArgumentParser()
for a in ["--task", "--size", "--ckpt_dir", "--src_video", "--src_ref_images", "--frame_num", "--sample_steps",
          "--prompt", "--save_file", "--offload_model", "--base_seed"]:
    p.add_argument(a)
p.add_argument("--t5_cpu", action="store_true")
a = p.parse_args()
assert a.task == "vace-1.3B"
assert pathlib.Path(a.src_ref_images).exists() and pathlib.Path(a.src_video).exists()
shutil.copy(a.src_video, a.save_file)
print("vace generated", a.size, a.frame_num, a.sample_steps, a.offload_model, a.t5_cpu, a.base_seed)
