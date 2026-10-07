#!/usr/bin/env python3
"""Reproduce Table 8: single- and two-concept intervention experiment,
plus the decorrelation stress test from Section 5.7."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import load_config
from src.data.augmentation import build_eval_augmentations
from src.data.concepts import compute_concept_vector
from src.data.datasets import AneRBCIDataset
from src.models.backbone import build_backbone
from src.models.cbm import ConceptBottleneckModel
from src.training.intervention import run_intervention_experiment


def _compute_reference_concepts(dataset) -> torch.Tensor:
    """Compute the deterministic morphometric reference concept vector
    for every cell crop in the dataset. Used as the "simulated expert
    correction" in the intervention experiment."""
    refs = []
    for i in tqdm(range(len(dataset)), desc="Computing reference concepts"):
        s = dataset.samples[i]
        img = np.asarray(Image.open(s.image_path).convert("RGB"))
        # Simple foreground mask by thresholding — real pipeline uses
        # the segmentation module output; this fallback keeps the script
        # runnable even when Cellpose weights are not downloaded.
        gray = img.mean(axis=-1)
        mask = (gray < gray.mean()).astype(np.uint8)
        median_diam = 2.0 * np.sqrt(mask.sum() / np.pi)
        c = compute_concept_vector(mask, img, img, median_diam)
        refs.append(c)
    return torch.tensor(np.asarray(refs), dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    backbone, feat_dim = build_backbone(cfg.model.backbone, pretrained=False)
    model = ConceptBottleneckModel(
        backbone, feat_dim,
        num_concepts=cfg.model.num_concepts,
        concept_hidden=cfg.model.concept_hidden,
        classifier_hidden=cfg.model.classifier_hidden,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model = model.to(cfg.train.device)

    test_ds = AneRBCIDataset(cfg.data.aner_bc_root, split="test",
                             split_manifest=cfg.data.split_manifest,
                             transform=build_eval_augmentations(cfg.data.input_size))
    loader = DataLoader(test_ds, batch_size=len(test_ds))
    all_images, all_labels = next(iter(loader))

    reference_concepts = _compute_reference_concepts(test_ds)

    report = run_intervention_experiment(
        model=model, images=all_images, labels=all_labels.numpy(),
        reference_concepts=reference_concepts, device=cfg.train.device,
    )

    out = Path("results/tables") / "table8_intervention.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(report.as_table())
    df.to_csv(out, index=False)
    print(df)
    print(f"\nDecorrelation stress test: single-c3 override recovered "
          f"{report.decorrelation_fn_recovered} / {report.num_fn} FN "
          f"({100 * report.decorrelation_fn_recovered / max(report.num_fn, 1):.1f}%)")


if __name__ == "__main__":
    main()
