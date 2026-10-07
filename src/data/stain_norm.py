"""Macenko stain normalization.

Reference-based color normalization for H&E / Romanowsky-stained images.
Based on Macenko et al. 2009, "A method for normalizing histology slides
for quantitative analysis." We implement a dependency-light version so
the pipeline doesn't force users to install the full `staintools` stack
when they only need normalization at inference time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


# ------------------------------------------------------------------
#  Core algorithm
# ------------------------------------------------------------------

def _rgb_to_od(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert an 8-bit RGB image to optical density space."""
    image = image.astype(np.float32)
    image[image == 0] = 1  # avoid log(0)
    return -np.log(image / 255.0 + eps)


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    rgb = 255.0 * np.exp(-od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _get_stain_matrix(od: np.ndarray, alpha: float = 1.0, beta: float = 0.15) -> np.ndarray:
    """Estimate the 2 principal stain vectors by PCA on the OD-space pixels."""
    # Discard very transparent pixels (background)
    od_flat = od.reshape(-1, 3)
    mask = (od_flat > beta).any(axis=1)
    od_hat = od_flat[mask]
    if od_hat.shape[0] < 10:
        # Fallback: identity-like matrix to avoid crashing on blank crops
        return np.eye(3, 2, dtype=np.float32)

    # Covariance and eigendecomposition
    cov = np.cov(od_hat.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    # Keep the two largest eigenvectors
    top2 = eig_vecs[:, -2:]
    # Project pixels onto the plane spanned by those two vectors
    proj = od_hat @ top2
    # Find the extreme angles — those give the two stain vectors
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo = np.percentile(phi, alpha)
    hi = np.percentile(phi, 100.0 - alpha)
    v1 = top2 @ np.array([np.cos(lo), np.sin(lo)])
    v2 = top2 @ np.array([np.cos(hi), np.sin(hi)])
    # Order stains so the first column is Hematoxylin (higher red absorbance)
    if v1[0] > v2[0]:
        stain = np.column_stack([v1, v2])
    else:
        stain = np.column_stack([v2, v1])
    # Normalize columns
    stain = stain / (np.linalg.norm(stain, axis=0, keepdims=True) + 1e-6)
    return stain.astype(np.float32)


def _get_concentrations(od: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Solve the least-squares deconvolution OD = Stain @ Conc."""
    od_flat = od.reshape(-1, 3).T
    conc, *_ = np.linalg.lstsq(stain_matrix, od_flat, rcond=None)
    return conc  # (2, N)


# ------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------

class MacenkoNormalizer:
    """Fit on a reference image, transform to normalize arbitrary slides."""

    def __init__(self) -> None:
        self.reference_stain: np.ndarray | None = None
        self.reference_max_conc: np.ndarray | None = None

    # ------------- fit -------------
    def fit(self, reference_image: np.ndarray | Path | str) -> "MacenkoNormalizer":
        img = self._load_image(reference_image)
        od = _rgb_to_od(img)
        self.reference_stain = _get_stain_matrix(od)
        conc = _get_concentrations(od, self.reference_stain)
        # Use the 99th percentile rather than the max to be robust to outliers
        self.reference_max_conc = np.percentile(conc, 99, axis=1).astype(np.float32)
        return self

    # ------------- transform -------------
    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.reference_stain is None or self.reference_max_conc is None:
            raise RuntimeError("MacenkoNormalizer must be fit() before transform().")
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Expected an (H, W, 3) RGB image.")

        od = _rgb_to_od(image)
        stain = _get_stain_matrix(od)
        conc = _get_concentrations(od, stain)
        src_max = np.percentile(conc, 99, axis=1).astype(np.float32)
        # Rescale source concentrations to match the reference's dynamic range
        src_max = np.where(src_max < 1e-6, 1.0, src_max)
        conc *= (self.reference_max_conc / src_max)[:, None]
        # Reconstruct using the REFERENCE stain matrix
        od_norm = (self.reference_stain @ conc).T.reshape(image.shape)
        return _od_to_rgb(od_norm)

    # ------------- convenience -------------
    @staticmethod
    def _load_image(src: np.ndarray | Path | str) -> np.ndarray:
        if isinstance(src, np.ndarray):
            return src
        return np.asarray(Image.open(src).convert("RGB"))
