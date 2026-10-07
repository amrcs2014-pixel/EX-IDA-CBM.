#!/usr/bin/env python3
"""Run Phase-2 AneRBC-I fine-tuning for one seed."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.training.train_phase2 import train_phase2


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 fine-tuning.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase1-checkpoint", default="results/logs/phase1/best_backbone.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = Path(args.phase1_checkpoint) if args.phase1_checkpoint else None
    if ckpt is not None and not ckpt.exists():
        print(f"WARNING: no Phase-1 checkpoint at {ckpt}; starting from ImageNet weights.")
        ckpt = None

    report = train_phase2(cfg, phase1_checkpoint=ckpt, seed=args.seed)
    print(f"[Phase 2 seed {args.seed}] best val AUC = {report['best_val_auc']:.4f}")


if __name__ == "__main__":
    main()
