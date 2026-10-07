#!/usr/bin/env python3
"""Build the stratified AneRBC-I split manifest once, so every run uses
exactly the same train/val/test partition. Also stains-normalizes the
training reference image.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def _collect_images(root: Path, pattern: str = "*.png") -> list[Path]:
    paths = []
    for sub in ("healthy", "anemic"):
        paths.extend(sorted((root / sub).glob(pattern)))
    return paths


def _stratified_split(paths: list[Path], train_frac: float, val_frac: float, seed: int):
    random.seed(seed)
    buckets = {"healthy": [], "anemic": []}
    for p in paths:
        label = "anemic" if "anemic" in str(p).lower() else "healthy"
        buckets[label].append(p.as_posix())
    split = {"train": [], "val": [], "test": []}
    for label, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        split["train"].extend(items[:n_train])
        split["val"].extend(items[n_train:n_train + n_val])
        split["test"].extend(items[n_train + n_val:])
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aner-bc-root", default="data/AneRBC-I")
    parser.add_argument("--output", default="data/splits/aner_bc_split_42.json")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.aner_bc_root)
    paths = _collect_images(root)
    if not paths:
        print(f"No .png files found under {root}. Please place AneRBC-I "
              f"images into {root}/healthy and {root}/anemic first.")
        return
    split = _stratified_split(paths, args.train_frac, args.val_frac, args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(f"Wrote {out}")
    for k, v in split.items():
        print(f"  {k}: {len(v)} images")


if __name__ == "__main__":
    main()
