"""Utility for creating train/val folders from an ImageFolder-style dataset.

Usage:
  python scripts/make_subset_imagenetr.py --src data/imagenet-r/raw --dst data/imagenet-r --val-ratio 0.2

The source folder should be organized as:
  raw/class_name/*.jpg

The output will be:
  dst/train/class_name/*.jpg
  dst/val/class_name/*.jpg
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    rng = random.Random(args.seed)
    classes = [p for p in src.iterdir() if p.is_dir()]
    for cls in classes:
        files = [p for p in cls.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        rng.shuffle(files)
        n_val = int(len(files) * args.val_ratio)
        split = {"val": files[:n_val], "train": files[n_val:]}
        for split_name, split_files in split.items():
            out_dir = dst / split_name / cls.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out_dir / f.name)
    print(f"Created split at {dst}")


if __name__ == "__main__":
    main()
