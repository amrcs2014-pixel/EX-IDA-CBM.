#!/usr/bin/env python3
"""Run Phase-1 Chula-RBC-12 pretraining and produce Table 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.augmentation import build_eval_augmentations
from src.data.datasets import CHULA_CLASSES, ChulaRBC12Dataset
from src.models.backbone import MorphologyPretrainHead, build_backbone
from src.training.train_phase1 import train_phase1


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 pretraining.")
    parser.add_argument("--config", required=True, help="Path to configs/phase1.yaml")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and just produce Table 1 from an existing checkpoint.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_dir = Path(cfg.output_dir) / "phase1"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        report = train_phase1(cfg, log_dir=log_dir)
        print(f"[Phase 1] best val accuracy = {report['best_val_accuracy']:.4f}")

    # Build Table 1 from the best checkpoint
    ckpt_path = log_dir / "best_backbone.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}; cannot build Table 1.")
        return
    state = torch.load(ckpt_path, map_location="cpu")

    backbone, feat_dim = build_backbone(cfg.model.backbone, pretrained=False)
    backbone.load_state_dict(state["backbone"])
    head = MorphologyPretrainHead(feat_dim, num_classes=len(CHULA_CLASSES))
    head.load_state_dict(state["head"])
    model = torch.nn.Sequential(backbone, head).to(cfg.train.device).eval()

    val_ds = ChulaRBC12Dataset(cfg.data.chula_rbc_root, split="val",
                               transform=build_eval_augmentations(cfg.data.input_size))
    val_loader = DataLoader(val_ds, batch_size=cfg.train.phase1_batch_size,
                            num_workers=cfg.train.num_workers)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            all_preds.append(model(x.to(cfg.train.device)).argmax(dim=-1).cpu().numpy())
            all_labels.append(y.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, zero_division=0)
    acc = float((preds == labels).mean())

    rows = []
    for i, cls_name in enumerate(CHULA_CLASSES):
        rows.append({"class": cls_name,
                     "precision": round(float(prec[i]), 3),
                     "recall": round(float(rec[i]), 3),
                     "f1": round(float(f1[i]), 3)})
    rows.append({"class": "Macro avg.",
                 "precision": round(float(prec.mean()), 3),
                 "recall": round(float(rec.mean()), 3),
                 "f1": round(float(f1.mean()), 3)})
    rows.append({"class": "Overall accuracy",
                 "precision": "", "recall": "", "f1": round(acc, 3)})

    out_tbl = Path("results/tables/table1_phase1_morph.csv")
    out_tbl.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_tbl, index=False)
    with open(log_dir / "table1_phase1_morph.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "macro_f1": float(f1.mean())}, f, indent=2)
    print(f"[Phase 1] Table 1 written to {out_tbl}")


if __name__ == "__main__":
    main()
