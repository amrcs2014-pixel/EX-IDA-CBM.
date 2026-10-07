"""Concept Bottleneck Model (CBM).

Architecture (paper Section 3.5):

    backbone features (B, 1280)
        │
    ConceptPredictor:  (B, 1280) -> (B, 5)       # sigmoid outputs
        │  this is the AUDITABLE bottleneck
    IDAClassifier:     (B, 5)    -> (B, 2)       # 2-layer MLP, softmax

Because the IDA decision is a deterministic function of the 5-vector
`c`, a clinician who disagrees with any concept score can overwrite it
and the prediction updates deterministically (see
`src/training/intervention.py` for the measured recovery rate).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
#  Concept predictor (features -> 5 concepts in [0,1])
# ------------------------------------------------------------------

class ConceptPredictor(nn.Module):
    """3-layer MLP mapping backbone features to the concept vector.

    The depth is chosen to be large enough for accurate concept
    prediction while remaining shallow enough to minimize *concept
    leakage* — the phenomenon whereby task-relevant information bypasses
    the semantic concept definitions.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        hidden_dims: tuple[int, int] = (512, 256),
        num_concepts: int = 5,
        dropout: tuple[float, float] = (0.3, 0.2),
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        d1, d2 = dropout
        self.block1 = nn.Sequential(
            nn.Linear(feature_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(d1),
        )
        self.block2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(d2),
        )
        self.out = nn.Linear(h2, num_concepts)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns concept vector c ∈ [0, 1]^K."""
        h = self.block1(features)
        h = self.block2(h)
        return torch.sigmoid(self.out(h))


# ------------------------------------------------------------------
#  IDA classifier (concepts -> 2-class logits)
# ------------------------------------------------------------------

class IDAClassifier(nn.Module):
    """2-layer MLP from the concept vector to IDA logits.

    This is intentionally compact: the decision depends only on `c`, and
    the small hidden layer makes the learned weights W_h and W_cls
    directly inspectable (useful for the audit trail required by the EU
    AI Act, Section 5.12 of the paper).
    """

    def __init__(
        self,
        num_concepts: int = 5,
        hidden_dim: int = 32,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(num_concepts, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, num_classes)."""
        return self.out(F.relu(self.hidden(concepts)))

    def per_concept_contribution(self, concepts: torch.Tensor) -> torch.Tensor:
        """Signed per-concept contribution to the IDA logit.

        Approximation: ignores the ReLU gating and uses the
        combined-weights path W_cls @ W_hidden, giving a (B, K) tensor
        where each entry is the signed contribution of concept k to the
        IDA (class-1) logit. Exact on the inputs where the ReLU is
        active for all 32 hidden units.
        """
        combined = self.out.weight @ self.hidden.weight  # (num_classes, K)
        # Take the IDA (positive) row minus the Normal row for a signed value
        delta_weights = combined[1] - combined[0]  # (K,)
        return concepts * delta_weights[None, :]


# ------------------------------------------------------------------
#  Composite model
# ------------------------------------------------------------------

class ConceptBottleneckModel(nn.Module):
    """End-to-end CBM: backbone + ConceptPredictor + IDAClassifier."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_concepts: int = 5,
        concept_hidden: tuple[int, int] = (512, 256),
        classifier_hidden: int = 32,
        num_classes: int = 2,
        dropout: tuple[float, float] = (0.3, 0.2),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.concept_predictor = ConceptPredictor(
            feature_dim=feature_dim,
            hidden_dims=concept_hidden,
            num_concepts=num_concepts,
            dropout=dropout,
        )
        self.classifier = IDAClassifier(
            num_concepts=num_concepts,
            hidden_dim=classifier_hidden,
            num_classes=num_classes,
        )

    # ------------- forward -------------
    def forward(
        self, images: torch.Tensor, return_concepts: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        concepts = self.concept_predictor(features)
        logits = self.classifier(concepts)
        if return_concepts:
            return logits, concepts
        return logits

    # ------------- intervention -------------
    def forward_with_concept_override(
        self,
        images: torch.Tensor,
        concept_override: torch.Tensor,
        override_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward with specific concepts overwritten.

        Parameters
        ----------
        images : (B, 3, H, W)
        concept_override : (B, K) replacement values in [0, 1]
        override_mask : (B, K) boolean; True means use the override, False
            means keep the model-predicted value.
        """
        features = self.backbone(images)
        concepts_pred = self.concept_predictor(features)
        concepts = torch.where(override_mask, concept_override, concepts_pred)
        return self.classifier(concepts)
