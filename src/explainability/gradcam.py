"""Grad-CAM++ saliency maps.

Implements the formulation of Chattopadhay et al. 2018, with the
second-order gradient term that distinguishes it from the original
Grad-CAM. We bind the forward/backward hooks to the final convolutional
feature map of the backbone (configurable).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMPlusPlus:
    """Compute Grad-CAM++ heatmaps for an arbitrary CNN classifier.

    Example
    -------
    >>> cam = GradCAMPlusPlus(model, model.backbone.conv_head)
    >>> heatmap = cam(image_batch, target_class=1)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    # ------------- hooks -------------
    def _save_activations(self, _module, _input, output):  # noqa: D401
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):  # noqa: D401
        self._gradients = grad_out[0].detach()

    # ------------- public API -------------
    def __call__(self, images: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """Return a (B, H, W) heatmap, min-max normalized to [0, 1]."""
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(images)
        if logits.ndim == 2:
            score = logits[:, target_class].sum()
        else:
            raise RuntimeError("Expected classifier logits of shape (B, C).")
        score.backward(retain_graph=True)

        activations = self._activations        # (B, C, h, w)
        gradients = self._gradients            # (B, C, h, w)
        if activations is None or gradients is None:
            raise RuntimeError("Hooks did not fire; verify target_layer.")

        # Second-order weighting (Grad-CAM++ formula)
        grads_power_2 = gradients.pow(2)
        grads_power_3 = gradients.pow(3)
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        eps = 1e-6
        alpha = grads_power_2 / (2.0 * grads_power_2 + sum_activations * grads_power_3 + eps)
        alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))

        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3))  # (B, C)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=1)
        cam = F.relu(cam)

        # Upsample to input resolution and min-max normalize
        cam = F.interpolate(
            cam.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        cam_np = cam.cpu().numpy()
        for i in range(cam_np.shape[0]):
            lo, hi = cam_np[i].min(), cam_np[i].max()
            cam_np[i] = (cam_np[i] - lo) / (hi - lo + 1e-8)
        return cam_np

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
