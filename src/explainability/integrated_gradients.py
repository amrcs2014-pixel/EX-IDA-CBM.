"""Integrated Gradients (Sundararajan et al. 2017).

IG is implemented from scratch rather than via `captum` so that the code
remains legible as a reference implementation. We integrate along a
straight-line path from a black-image baseline to the input with
`steps=200` Riemann segments (matches the paper's N=200).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class IntegratedGradients:
    """Axiom-satisfying pixel-level attributions."""

    def __init__(self, model: nn.Module, steps: int = 200) -> None:
        self.model = model
        self.steps = steps

    def __call__(
        self,
        images: torch.Tensor,
        target_class: int = 1,
        baseline: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Return per-pixel attributions, shape (B, 3, H, W)."""
        if baseline is None:
            baseline = torch.zeros_like(images)

        self.model.eval()
        alphas = torch.linspace(0.0, 1.0, self.steps, device=images.device)
        total_grads = torch.zeros_like(images)

        for alpha in alphas:
            interpolated = baseline + alpha * (images - baseline)
            interpolated.requires_grad_(True)
            logits = self.model(interpolated)
            score = logits[:, target_class].sum()
            score.backward()
            total_grads = total_grads + interpolated.grad.detach()
            interpolated.grad.zero_()

        avg_grads = total_grads / self.steps
        attribution = (images - baseline) * avg_grads
        return attribution.detach().cpu().numpy()

    def normalized_magnitude(self, attribution: np.ndarray) -> np.ndarray:
        """Collapse channels and min-max normalize for display."""
        mag = np.abs(attribution).sum(axis=1)  # (B, H, W)
        out = np.empty_like(mag)
        for i in range(mag.shape[0]):
            lo, hi = mag[i].min(), mag[i].max()
            out[i] = (mag[i] - lo) / (hi - lo + 1e-8)
        return out
