"""Phase-1: 12-class morphology pretraining on Chula-RBC-12.

Produces the backbone weights that Phase-2 is initialized from, and the
numbers that populate Table 1 of the paper.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from ..config import PipelineConfig
from ..data.augmentation import build_eval_augmentations, build_train_augmentations
from ..data.datasets import CHULA_CLASSES, ChulaRBC12Dataset
from ..models.backbone import MorphologyPretrainHead, build_backbone
from ._focal import FocalLoss


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _class_balanced_sampler(dataset) -> WeightedRandomSampler:
    """Weight samples inversely by class frequency."""
    labels = np.array([s["class"] for s in dataset.samples])
    _, counts = np.unique(labels, return_counts=True)
    freq = counts.astype(np.float32) / counts.sum()
    label_to_weight = {cls: 1.0 / f for cls, f in zip(np.unique(labels), freq)}
    weights = np.array([label_to_weight[label] for label in labels], dtype=np.float32)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(dataset),
        replacement=True,
    )


def train_phase1(cfg: PipelineConfig, log_dir: str | Path | None = None) -> dict:
    """Train the backbone on Chula-RBC-12 and return the Phase-1 report."""
    log_dir = Path(log_dir or Path(cfg.output_dir) / "phase1")
    log_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.train.seed)

    train_ds = ChulaRBC12Dataset(
        cfg.data.chula_rbc_root, split="train", transform=build_train_augmentations(cfg.data.input_size)
    )
    val_ds = ChulaRBC12Dataset(
        cfg.data.chula_rbc_root, split="val", transform=build_eval_augmentations(cfg.data.input_size)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.phase1_batch_size,
        sampler=_class_balanced_sampler(train_ds),
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.phase1_batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )

    backbone, feature_dim = build_backbone(cfg.model.backbone, pretrained=cfg.model.pretrained)
    head = MorphologyPretrainHead(feature_dim, num_classes=len(CHULA_CLASSES))
    model = nn.Sequential(backbone, head).to(cfg.train.device)

    criterion = FocalLoss(gamma=cfg.train.focal_gamma, label_smoothing=cfg.train.label_smoothing)
    optim = AdamW(model.parameters(), lr=cfg.train.phase1_lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=cfg.train.phase1_epochs)

    best_val_auc = 0.0
    for epoch in range(cfg.train.phase1_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Phase-1 epoch {epoch + 1}/{cfg.train.phase1_epochs}")
        for x, y in pbar:
            x, y = x.to(cfg.train.device, non_blocking=True), y.to(cfg.train.device, non_blocking=True)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)
            optim.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(cfg.train.device), y.to(cfg.train.device)
                preds = model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        print(f"[Phase 1] epoch {epoch + 1}: val_acc = {val_acc:.4f}")

        if val_acc > best_val_auc:
            best_val_auc = val_acc
            torch.save(
                {"backbone": backbone.state_dict(), "head": head.state_dict(),
                 "val_acc": val_acc, "epoch": epoch + 1},
                log_dir / "best_backbone.pt",
            )

    report = {"best_val_accuracy": best_val_auc, "epochs": cfg.train.phase1_epochs}
    with open(log_dir / "phase1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
