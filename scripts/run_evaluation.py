#!/usr/bin/env python3
"""Evaluate a trained CBM checkpoint on the AneRBC-I test split and
emit Table 3 (classification) + Table 2 (segmentation) as CSV."""

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
from src.evaluation.metrics import bootstrap_ci, classification_report
from src.models.backbone import build_backbone
from src.models.cbm import ConceptBottleneckModel


def _load_cbm(cfg, checkpoint_path: Path) -> ConceptBottleneckModel:
    backbone, feat_dim = build_backbone(cfg.model.backbone, pretrained=False)
    model = ConceptBottleneckModel(
        backbone=backbone, feature_dim=feat_dim,
        num_concepts=cfg.model.num_concepts,
        concept_hidden=cfg.model.concept_hidden,
        classifier_hidden=cfg.model.classifier_hidden,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    return model


def _collect_test_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CBM on AneRBC-I test set.")
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = _load_cbm(cfg, Path(args.checkpoint)).to(cfg.train.device)

    test_ds = AneRBCIDataset(cfg.data.aner_bc_root, split="test",
                             split_manifest=cfg.data.split_manifest,
                             transform=build_eval_augmentations(cfg.data.input_size))
    test_loader = DataLoader(test_ds, batch_size=cfg.train.phase2_batch_size,
                             num_workers=cfg.train.num_workers)

    probs, labels = _collect_test_probs(model, test_loader, cfg.train.device)
    report = classification_report(probs, labels)

    # Bootstrap CIs for the top-line metrics
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    rows = []
    metric_specs = [
        ("Accuracy", lambda p, y: accuracy_score(y, (p >= 0.5).astype(int))),
        ("Precision (macro)", lambda p, y: report.precision_macro),  # already in `report`
        ("Recall (macro)", lambda p, y: report.recall_macro),
        ("F1-Score (macro)", lambda p, y: report.f1_macro),
        ("AUC-ROC", lambda p, y: roc_auc_score(y, p)),
        ("AUC-PR", lambda p, y: average_precision_score(y, p)),
    ]
    from src.evaluation.calibration import (
        brier_score, expected_calibration_error, max_calibration_error,
    )
    # These calibration helpers accept (probs, labels) rather than the
    # sklearn (labels, probs) convention. The bootstrap_ci wrapper
    # handles both via `argument_order`.
    calibration_specs = [
        ("ECE", expected_calibration_error),
        ("Brier score", brier_score),
        ("MCE (worst bin)", max_calibration_error),
    ]

    for name, fn in metric_specs:
        try:
            # sklearn metrics expect (labels, probs)
            point, lo, hi = bootstrap_ci(
                probs, labels, fn,
                num_resamples=cfg.eval.bootstrap_resamples,
                argument_order="labels_first",
            )
        except Exception:
            point = fn(labels, probs)
            lo, hi = float("nan"), float("nan")
        rows.append({"metric": name, "value": round(point, 4),
                     "ci_low": round(lo, 4) if not np.isnan(lo) else "",
                     "ci_high": round(hi, 4) if not np.isnan(hi) else ""})

    for name, fn in calibration_specs:
        try:
            point, lo, hi = bootstrap_ci(
                probs, labels, fn,
                num_resamples=cfg.eval.bootstrap_resamples,
                argument_order="probs_first",
            )
        except Exception:
            point = fn(probs, labels)
            lo, hi = float("nan"), float("nan")
        rows.append({"metric": name, "value": round(point, 4),
                     "ci_low": round(lo, 4) if not np.isnan(lo) else "",
                     "ci_high": round(hi, 4) if not np.isnan(hi) else ""})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table3_classification.csv", index=False)

    with open(out_dir / "table3_classification.json", "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"[Eval] Table 3 written to {out_dir / 'table3_classification.csv'}")
    print(df)


if __name__ == "__main__":
    main()
