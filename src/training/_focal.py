"""Focal loss with label smoothing — used in both Phase-1 and Phase-2."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017) with label smoothing.

    L = -alpha * (1 - p_t)^gamma * log(p_t)

    We implement the version that accepts raw logits so there is no
    silent sigmoid/softmax double-pass.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Label-smoothed one-hot targets
        if self.label_smoothing > 0:
            smooth_target = torch.full_like(
                log_probs, self.label_smoothing / (num_classes - 1)
            )
            smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_target = F.one_hot(targets, num_classes).float()

        p_t = (probs * smooth_target).sum(dim=-1)
        log_p_t = (log_probs * smooth_target).sum(dim=-1)
        focal = (1.0 - p_t).pow(self.gamma) * (-log_p_t)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal
