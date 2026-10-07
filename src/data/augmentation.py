"""Augmentation pipeline: Macenko-normalized photometric + geometric transforms
plus MixUp and CutMix for batch-level regularization.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T


def build_train_augmentations(input_size: int = 224) -> T.Compose:
    """Augmentations for Phase-1 and Phase-2 training.

    Geometric: full-rotation (±180°), horizontal & vertical flip, elastic.
    Photometric: color jitter, Gaussian noise, blur. Macenko stain
    normalization is applied upstream of the Dataset (see
    `scripts/build_split_manifests.py`), so it does not appear here.
    """
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.uint8, scale=False),
        T.Resize(input_size, antialias=True),
        T.CenterCrop(input_size),
        T.RandomRotation(degrees=180),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ElasticTransform(alpha=120.0, sigma=10.0),
        T.ColorJitter(brightness=0.15, contrast=0.10, saturation=0.08),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_augmentations(input_size: int = 224) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.uint8, scale=False),
        T.Resize(input_size, antialias=True),
        T.CenterCrop(input_size),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------
#  MixUp + CutMix
# ------------------------------------------------------------------

def _rand_bbox(size: tuple[int, int, int, int], lam: float) -> tuple[int, int, int, int]:
    _, _, H, W = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def mixup_cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    apply_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply either MixUp or CutMix with probability `apply_prob`.

    Returns the possibly-mixed images, the primary labels, the secondary
    labels, and the mixing coefficient lambda. If no mixing was applied,
    lambda=1.0 and labels_b == labels_a.
    """
    if random.random() > apply_prob:
        return images, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    labels_b = labels[perm]

    if random.random() < 0.5:
        # MixUp: convex combination over the whole image
        images = lam * images + (1.0 - lam) * images[perm]
    else:
        # CutMix: splice a rectangular region
        x1, y1, x2, y2 = _rand_bbox(tuple(images.size()), lam)
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))

    return images, labels, labels_b, lam


def mixed_criterion(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Loss for a mixed batch: lambda * L(y_a) + (1 - lambda) * L(y_b)."""
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
