#!/usr/bin/env python3
"""Reproduce Section 5.9: five-seed sweep producing mean ± std for
accuracy, F1 and AUC on the AneRBC-I test partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.augmentation import build_eval_augmentations
from src.data.datasets import AneRBCIDataset
from src.evaluation.metrics import classification_report
from src.models.backbone import build_backbone
from src.models.cbm import ConceptBottleneckModel
from src.training.train_phase2 import train_phase2


def _eval_test(cfg, ckpt_path: Path):
    backbone, feat_dim = build_backbone(cfg.model.backbone, pretrained=False)
    model = ConceptBottleneckModel(
        backbone, feat_dim,
        num_concepts=cfg.model.num_concepts,
        concept_hidden=cfg.model.concept_hidden,
        classifier_hidden=cfg.model.classifier_hidden,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model = model.to(cfg.train.device).eval()
    test_ds = AneRBCIDataset(cfg.data.aner_bc_root, split="test",
                             split_manifest=cfg.data.split_manifest,
                             transform=build_eval_augmentations(cfg.data.input_size))
    loader = DataLoader(test_ds, batch_size=cfg.train.phase2_batch_size,
                        num_workers=cfg.train.num_workers)
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(cfg.train.device)), dim=-1)[:, 1].cpu().numpy())
            labels.append(y.numpy())
    return classification_report(np.concatenate(probs), np.concatenate(labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 101])
    args = parser.parse_args()

    cfg = load_config(args.config)
    per_seed_rows = []
    for seed in args.seeds:
        log_dir = Path(cfg.output_dir) / f"multiseed_seed{seed}"
        train_phase2(cfg, phase1_checkpoint="results/logs/phase1/best_backbone.pt",
                     log_dir=log_dir, seed=seed)
        report = _eval_test(cfg, log_dir / "best.pt")
        per_seed_rows.append({"seed": seed, **report.to_dict()})
        print(f"[Multi-seed] seed {seed} -> acc={report.accuracy:.4f} "
              f"f1={report.f1_macro:.4f} AUC={report.auc_roc:.4f}")

    df = pd.DataFrame(per_seed_rows)
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "multiseed_per_seed.csv", index=False)

    summary = {col: {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=1))}
               for col in df.columns if col != "seed"}
    with open(out_dir / "multiseed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nMulti-seed summary:")
    for k, v in summary.items():
        print(f"  {k:20s}  mean = {v['mean']:.4f}  ±  {v['std']:.4f}")


if __name__ == "__main__":
    main()
