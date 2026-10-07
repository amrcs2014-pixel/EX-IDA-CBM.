"""Phase-2: AneRBC-I fine-tuning of the full Concept Bottleneck Model.

Produces the model that generates every number in Table 3 (and most of
Tables 5-8). Uses differential learning rates — a small rate on the
Phase-1-pretrained backbone, the full rate on the concept predictor and
IDA classifier heads.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import PipelineConfig
from ..data.augmentation import (
    build_eval_augmentations, build_train_augmentations,
    mixed_criterion, mixup_cutmix_batch,
)
from ..data.datasets import AneRBCIDataset
from ..evaluation.metrics import classification_report
from ..models.backbone import build_backbone
from ..models.cbm import ConceptBottleneckModel
from ._focal import FocalLoss


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_model(cfg: PipelineConfig, phase1_checkpoint: str | Path | None) -> ConceptBottleneckModel:
    backbone, feat_dim = build_backbone(cfg.model.backbone, pretrained=cfg.model.pretrained)
    if phase1_checkpoint is not None:
        state = torch.load(phase1_checkpoint, map_location="cpu")
        missing, unexpected = backbone.load_state_dict(state["backbone"], strict=False)
        if missing or unexpected:
            print(f"[Phase 2] loaded backbone with missing={len(missing)} unexpected={len(unexpected)}")
    return ConceptBottleneckModel(
        backbone=backbone,
        feature_dim=feat_dim,
        num_concepts=cfg.model.num_concepts,
        concept_hidden=cfg.model.concept_hidden,
        classifier_hidden=cfg.model.classifier_hidden,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )


def train_phase2(
    cfg: PipelineConfig,
    phase1_checkpoint: str | Path | None = None,
    log_dir: str | Path | None = None,
    seed: int | None = None,
) -> dict:
    """Train the CBM on AneRBC-I and return the Phase-2 report."""
    seed = seed if seed is not None else cfg.train.seed
    log_dir = Path(log_dir or Path(cfg.output_dir) / f"phase2_seed{seed}")
    log_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)

    train_ds = AneRBCIDataset(
        cfg.data.aner_bc_root, split="train",
        split_manifest=cfg.data.split_manifest,
        transform=build_train_augmentations(cfg.data.input_size),
    )
    val_ds = AneRBCIDataset(
        cfg.data.aner_bc_root, split="val",
        split_manifest=cfg.data.split_manifest,
        transform=build_eval_augmentations(cfg.data.input_size),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.train.phase2_batch_size,
                              shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.phase2_batch_size,
                            shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True)

    model = _build_model(cfg, phase1_checkpoint).to(cfg.train.device)
    criterion = FocalLoss(gamma=cfg.train.focal_gamma, label_smoothing=cfg.train.label_smoothing)

    # Differential learning rates: 0.1x on backbone, full on heads
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.concept_predictor.parameters())
                   + list(model.classifier.parameters()))
    optim = AdamW([
        {"params": backbone_params, "lr": cfg.train.phase2_lr * cfg.train.phase2_backbone_lr_mult},
        {"params": head_params, "lr": cfg.train.phase2_lr},
    ], weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)

    best_val_auc = 0.0
    patience_counter = 0
    history: list[dict] = []
    for epoch in range(cfg.train.phase2_epochs):
        # --------- train ----------
        model.train()
        pbar = tqdm(train_loader, desc=f"Phase-2 (seed {seed}) epoch {epoch + 1}/{cfg.train.phase2_epochs}")
        for x, y in pbar:
            x = x.to(cfg.train.device, non_blocking=True)
            y = y.to(cfg.train.device, non_blocking=True)
            x_mix, ya, yb, lam = mixup_cutmix_batch(
                x, y, alpha=cfg.train.mixup_alpha, apply_prob=cfg.train.mixup_cutmix_prob,
            )
            optim.zero_grad()
            logits = model(x_mix)
            loss = mixed_criterion(criterion, logits, ya, yb, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)
            optim.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        # --------- validate ----------
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(cfg.train.device)
                p = torch.softmax(model(x), dim=-1)[:, 1].cpu().numpy()
                probs.append(p)
                labels.append(y.numpy())
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        report = classification_report(probs, labels)
        history.append({"epoch": epoch + 1, **report.to_dict()})
        print(f"[Phase 2 seed {seed}] epoch {epoch + 1}: "
              f"val AUC = {report.auc_roc:.4f}, acc = {report.accuracy:.4f}")

        if report.auc_roc > best_val_auc:
            best_val_auc = report.auc_roc
            patience_counter = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "val_auc": report.auc_roc,
                "seed": seed,
            }, log_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.phase2_early_stop_patience:
                print(f"[Phase 2 seed {seed}] early stopping at epoch {epoch + 1}")
                break

    with open(log_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return {"best_val_auc": best_val_auc, "epochs_trained": epoch + 1, "seed": seed}
