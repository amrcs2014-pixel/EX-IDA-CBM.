#!/usr/bin/env python3
"""Reproduce Table 9: one-at-a-time hyperparameter sensitivity sweep
on the validation partition only (never the test partition)."""

from __future__ import annotations

import argparse
import copy
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


SWEEP = {
    "Weight decay λ": {
        "key": ("train", "weight_decay"),
        "values": [1e-5, 1e-4, 1e-3],
        "reported": 1e-4,
    },
    "Batch size": {
        "key": ("train", "phase2_batch_size"),
        "values": [32, 64, 128],
        "reported": 64,
    },
    "Focal-loss γ": {
        "key": ("train", "focal_gamma"),
        "values": [1.0, 2.0, 3.0],
        "reported": 2.0,
    },
    "Label smoothing ε": {
        "key": ("train", "label_smoothing"),
        "values": [0.0, 0.1, 0.2],
        "reported": 0.1,
    },
}


def _eval_val_auc(cfg, ckpt_path: Path) -> float:
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
    val_ds = AneRBCIDataset(cfg.data.aner_bc_root, split="val",
                            split_manifest=cfg.data.split_manifest,
                            transform=build_eval_augmentations(cfg.data.input_size))
    loader = DataLoader(val_ds, batch_size=cfg.train.phase2_batch_size,
                        num_workers=cfg.train.num_workers)
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(cfg.train.device)), dim=-1)[:, 1].cpu().numpy())
            labels.append(y.numpy())
    rep = classification_report(np.concatenate(probs), np.concatenate(labels))
    return rep.auc_roc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    base = load_config(args.config)

    rows = []
    for hp_name, spec in SWEEP.items():
        section, field = spec["key"]
        for v in spec["values"]:
            c = copy.deepcopy(base)
            getattr(c, section).__setattr__(field, v)
            c.experiment_name = f"hp_{field}_{v}"
            log_dir = Path(c.output_dir) / f"hp_sens_{field}_{v}"
            train_phase2(c, phase1_checkpoint="results/logs/phase1/best_backbone.pt",
                         log_dir=log_dir, seed=c.train.seed)
            auc = _eval_val_auc(c, log_dir / "best.pt")
            tag = f"{v}" + (" (reported)" if v == spec["reported"] else "")
            rows.append({"hyperparameter": hp_name, "value": tag,
                         "val_auc_roc": round(auc, 3)})
            print(f"[HP sweep] {hp_name} = {v} -> val AUC {auc:.4f}")

    out = Path("results/tables") / "table9_hp_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
