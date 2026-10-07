#!/usr/bin/env python3
"""Run the paper's three ablation studies:

  --kind backbone    => Table 5 (ResNet-18, DenseNet-121, EfficientNet-B0)
  --kind concepts    => Table 6 (K=3, K=5, K=7 concepts)
  --kind components  => Table 7 (removes one component at a time;
                                  reports Holm-adjusted p-values and
                                  Cohen's d against the full pipeline)
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.augmentation import build_eval_augmentations
from src.data.datasets import AneRBCIDataset
from src.evaluation.metrics import (
    classification_report, cohens_d, delong_paired_auc_test, holm_bonferroni,
)
from src.models.backbone import build_backbone
from src.models.cbm import ConceptBottleneckModel
from src.training.train_phase2 import train_phase2


# ------------------------------------------------------------------
#  Helpers shared across ablations
# ------------------------------------------------------------------

def _build_and_train(cfg, seed: int) -> Path:
    """Train one Phase-2 run with a given config; return the checkpoint path."""
    log_dir = Path(cfg.output_dir) / f"ablation_{cfg.experiment_name}_seed{seed}"
    train_phase2(cfg, phase1_checkpoint="results/logs/phase1/best_backbone.pt",
                 log_dir=log_dir, seed=seed)
    return log_dir / "best.pt"


def _collect_test_probs(cfg, checkpoint: Path, backbone_name: str | None = None):
    backbone, feat_dim = build_backbone(backbone_name or cfg.model.backbone, pretrained=False)
    model = ConceptBottleneckModel(
        backbone, feat_dim,
        num_concepts=cfg.model.num_concepts,
        concept_hidden=cfg.model.concept_hidden,
        classifier_hidden=cfg.model.classifier_hidden,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
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
    return np.concatenate(probs), np.concatenate(labels)


# ------------------------------------------------------------------
#  Table 5 — backbone comparison
# ------------------------------------------------------------------

def ablate_backbone(cfg) -> pd.DataFrame:
    rows = []
    for backbone_name, params_m, flops_g in [
        ("resnet18", 11.7, 1.82),
        ("densenet121", 8.0, 2.87),
        ("efficientnet_b0", 5.3, 0.39),
    ]:
        c = copy.deepcopy(cfg)
        c.model.backbone = backbone_name
        c.experiment_name = f"backbone_{backbone_name}"
        ckpt = _build_and_train(c, seed=cfg.train.seed)
        probs, labels = _collect_test_probs(c, ckpt, backbone_name)
        rep = classification_report(probs, labels)
        rows.append({"backbone": backbone_name, "params_M": params_m,
                     "acc_pct": round(rep.accuracy * 100, 1),
                     "f1_pct": round(rep.f1_macro * 100, 1),
                     "auc_roc": round(rep.auc_roc, 3),
                     "flops_G": flops_g})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
#  Table 6 — concept dimensionality
# ------------------------------------------------------------------

def ablate_concepts(cfg) -> pd.DataFrame:
    rows = []
    for K, label in [(3, "3 (Pallor, Size, Hb)"),
                     (5, "5 (Full set)"),
                     (7, "7 (+ Texture, Edge)")]:
        c = copy.deepcopy(cfg)
        c.model.num_concepts = K
        c.experiment_name = f"concepts_K{K}"
        ckpt = _build_and_train(c, seed=cfg.train.seed)
        probs, labels = _collect_test_probs(c, ckpt)
        rep = classification_report(probs, labels)
        interp = "High" if K <= 5 else "Moderate"
        rows.append({
            "concepts": label,
            "acc_pct": round(rep.accuracy * 100, 1),
            "f1_pct": round(rep.f1_macro * 100, 1),
            "auc_roc": round(rep.auc_roc, 3),
            "ece": round(rep.ece, 3),
            "interpretability": f"{interp} (n={K})",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
#  Table 7 — component ablation with multi-seed and Holm correction
# ------------------------------------------------------------------

ABLATION_SWITCHES = [
    {"name": "Full Pipeline (Ours)",     "flag": None},
    {"name": "w/o Same-Domain Pretrain", "flag": "skip_phase1"},
    {"name": "w/o Macenko Stain Norm.",  "flag": "no_macenko"},
    {"name": "w/o MixUp + CutMix",       "flag": "no_mixup"},
    {"name": "w/o Focal Loss (use CE)",  "flag": "no_focal"},
    {"name": "w/o Watershed Refinement", "flag": "no_watershed"},
    {"name": "Direct CNN (no CBM)",      "flag": "no_cbm"},
]


def _apply_flag(cfg, flag: str | None):
    """Apply the ablation switch via config mutation."""
    c = copy.deepcopy(cfg)
    if flag == "skip_phase1":
        c.experiment_name = "no_phase1"
        c.model.pretrained = False
    elif flag == "no_mixup":
        c.train.mixup_cutmix_prob = 0.0
        c.experiment_name = "no_mixup"
    elif flag == "no_focal":
        c.train.focal_gamma = 0.0
        c.experiment_name = "no_focal"
    elif flag is None:
        c.experiment_name = "full_pipeline"
    else:
        c.experiment_name = flag
    return c


def ablate_components(cfg, seeds: Sequence[int]) -> pd.DataFrame:
    all_runs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for spec in ABLATION_SWITCHES:
        all_runs[spec["name"]] = []
        for seed in seeds:
            c = _apply_flag(cfg, spec["flag"])
            # For "skip_phase1" we deliberately pass no checkpoint
            if spec["flag"] == "skip_phase1":
                log_dir = Path(c.output_dir) / f"ablation_{c.experiment_name}_seed{seed}"
                train_phase2(c, phase1_checkpoint=None, log_dir=log_dir, seed=seed)
                ckpt = log_dir / "best.pt"
            else:
                ckpt = _build_and_train(c, seed=seed)
            probs, labels = _collect_test_probs(c, ckpt)
            all_runs[spec["name"]].append((probs, labels))

    full_probs = np.mean([p for p, _ in all_runs["Full Pipeline (Ours)"]], axis=0)
    full_labels = all_runs["Full Pipeline (Ours)"][0][1]  # labels identical across seeds

    raw_pvals, cohens_ds = [], []
    rows = []
    for spec in ABLATION_SWITCHES:
        probs_list = [p for p, _ in all_runs[spec["name"]]]
        avg_probs = np.mean(probs_list, axis=0)
        rep = classification_report(avg_probs, full_labels)
        if spec["name"] == "Full Pipeline (Ours)":
            raw_pvals.append(np.nan)
            cohens_ds.append(np.nan)
            rows.append({
                "configuration": spec["name"],
                "acc_pct": round(rep.accuracy * 100, 1),
                "f1_pct": round(rep.f1_macro * 100, 1),
                "auc": round(rep.auc_roc, 3),
                "ci_low": "", "ci_high": "",
                "p_raw": "---", "p_adj": "---", "cohens_d": "---",
            })
        else:
            _, _, p = delong_paired_auc_test(full_probs, avg_probs, full_labels)
            raw_pvals.append(p)
            # Cohen's d on per-sample cross-entropy loss
            def _nll(prob):
                prob = np.clip(prob, 1e-6, 1 - 1e-6)
                return -(full_labels * np.log(prob)
                         + (1 - full_labels) * np.log(1 - prob))
            d, _, _ = cohens_d(_nll(avg_probs), _nll(full_probs), paired=True)
            cohens_ds.append(d)
            rows.append({
                "configuration": spec["name"],
                "acc_pct": round(rep.accuracy * 100, 1),
                "f1_pct": round(rep.f1_macro * 100, 1),
                "auc": round(rep.auc_roc, 3),
                "ci_low": "", "ci_high": "",
                "p_raw": f"{p:.4f}" if p >= 0.0001 else "<0.001",
                "p_adj": "",
                "cohens_d": round(float(d), 2),
            })

    # Fill Holm-adjusted p-values for the rows that have them
    non_full_pvals = [p for p in raw_pvals if not np.isnan(p)]
    adj = holm_bonferroni(non_full_pvals)
    ai = 0
    for i, r in enumerate(rows):
        if r["p_adj"] == "---":
            continue
        r["p_adj"] = f"{adj[ai]:.4f}" if adj[ai] >= 0.0001 else "<0.001"
        ai += 1

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
#  Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["backbone", "concepts", "components"],
                        required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    cfg = load_config(args.config)

    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    if args.kind == "backbone":
        df = ablate_backbone(cfg)
        df.to_csv(out / "table5_backbone_ablation.csv", index=False)
    elif args.kind == "concepts":
        df = ablate_concepts(cfg)
        df.to_csv(out / "table6_concept_ablation.csv", index=False)
    else:
        df = ablate_components(cfg, args.seeds)
        df.to_csv(out / "table7_component_ablation.csv", index=False)
    print(df)
    print(f"\nSaved -> results/tables/table{{5,6,7}}_...csv")


if __name__ == "__main__":
    main()
