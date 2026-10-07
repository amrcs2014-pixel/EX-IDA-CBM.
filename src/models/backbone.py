"""Backbone CNN for feature extraction.

The paper reports EfficientNet-B0 as the final choice because it gave
the best classification performance at the lowest parameter count
(Table 5 — 98.9% / 0.993 AUC with only 5.3M parameters). We expose the
alternatives (ResNet-18, DenseNet-121) for the backbone-ablation
experiment in Section 4.5.1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover
    _HAS_TIMM = False


# ------------------------------------------------------------------
#  Factory
# ------------------------------------------------------------------

_FEATURE_DIM = {
    "efficientnet_b0": 1280,
    "resnet18": 512,
    "densenet121": 1024,
}


def build_backbone(
    name: str = "efficientnet_b0",
    pretrained: bool = True,
    global_pool: str = "avg",
) -> tuple[nn.Module, int]:
    """Create a feature-extraction backbone that returns pooled features.

    Returns
    -------
    (backbone_module, feature_dim)
        `backbone_module(x)` -> tensor of shape (B, feature_dim).
    """
    if not _HAS_TIMM:
        raise ImportError(
            "`timm` is required. `pip install timm` or use the conda env."
        )
    if name not in _FEATURE_DIM:
        raise ValueError(
            f"Unknown backbone '{name}'. Must be one of {list(_FEATURE_DIM)}."
        )

    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,         # strip classifier head
        global_pool=global_pool,
    )
    return model, _FEATURE_DIM[name]


class MorphologyPretrainHead(nn.Module):
    """Simple 12-class linear head used during Phase-1 pretraining.

    Phase-1 trains the backbone to classify the 12 Chula-RBC-12
    morphology classes. After Phase 1 the head is discarded and the
    backbone is coupled to the concept predictor for Phase 2.
    """

    def __init__(self, feature_dim: int, num_classes: int = 12, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(features))
