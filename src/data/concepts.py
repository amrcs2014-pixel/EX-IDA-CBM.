"""Deterministic morphological concept operators.

Each of the five concepts the CBM is built around is computed by a
fixed operator from a segmented single-cell mask and its intensity
image. All operators are applied AFTER Macenko stain normalization, so
colour-dependent concepts are measured on a common stain reference.

The operators defined here are the paper's canonical definitions
(Section 3.2.3 of the manuscript). They are intentionally deterministic
so that two runs on the same cell produce the same concept vector; any
clinically meaningful audit of the pipeline therefore reduces to
auditing these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu


# ------------------------------------------------------------------
#  Low-level helpers
# ------------------------------------------------------------------

def _ensure_single_channel(image: np.ndarray) -> np.ndarray:
    """Return a 2-D grayscale view of a possibly 3-D image."""
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3 and image.shape[-1] == 3:
        # Standard luminance weights for RGB -> grayscale
        return (0.2989 * image[..., 0] +
                0.5870 * image[..., 1] +
                0.1140 * image[..., 2]).astype(np.float32)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 1e-12 else default


# ------------------------------------------------------------------
#  Individual concept operators
# ------------------------------------------------------------------

def central_pallor_ratio(cell_mask: np.ndarray, intensity: np.ndarray) -> float:
    """c1: Area fraction of the interior pale region.

    The pale region is defined as pixels inside the cell mask whose
    (grayscale) intensity exceeds the per-cell Otsu threshold. In an
    IDA-affected cell this ratio enlarges because hemoglobin is scarce
    and the cell becomes centrally translucent.
    """
    gray = _ensure_single_channel(intensity)
    cell_pixels = gray[cell_mask > 0]
    if cell_pixels.size < 10:
        return 0.0
    try:
        thresh = threshold_otsu(cell_pixels)
    except ValueError:
        return 0.0
    pale = ((gray > thresh) & (cell_mask > 0)).sum()
    total = (cell_mask > 0).sum()
    return float(np.clip(_safe_divide(pale, total), 0.0, 1.0))


def relative_cell_diameter(cell_mask: np.ndarray, image_median_diameter: float) -> float:
    """c2: Equivalent-circle diameter normalized by the per-image median.

    `image_median_diameter` is the median equivalent-circle diameter
    across every retained cell in the same smear. Microcytic cells in
    IDA yield values well below 1.
    """
    props = measure.regionprops((cell_mask > 0).astype(np.uint8))
    if not props:
        return 0.0
    area = props[0].area
    diameter = 2.0 * np.sqrt(area / np.pi)
    # Normalize so that c2 lives in [0, 1]: 0.5 means "median size", values
    # below 0.5 mean microcytosis (relative to this smear).
    rel = _safe_divide(diameter, 2.0 * image_median_diameter)
    return float(np.clip(rel, 0.0, 1.0))


def hemoglobinization_score(cell_mask: np.ndarray, stain_normalized: np.ndarray) -> float:
    """c3: hypochromia-ness proxy from cytoplasmic brightness.

    Under Romanowsky staining, hemoglobin absorbs eosin: a well-
    hemoglobinized cell therefore stains more intensely (its cytoplasm
    appears DARKER / lower mean grayscale intensity), while a
    hypochromic cell exhibits reduced eosin uptake (its cytoplasm
    appears BRIGHTER / higher mean intensity). We return the mean
    cytoplasmic brightness directly so that *higher* scores correspond
    to *more* hypochromia — matching the paper's convention that both
    IDA cells in Fig. 7 exhibit high `Hemoglobinization Score`.

    Operates on the stain-normalized image so that cross-laboratory
    staining-intensity differences do not confound the measurement.
    """
    gray = _ensure_single_channel(stain_normalized)
    cell_pixels = gray[cell_mask > 0]
    if cell_pixels.size < 10:
        return 0.0
    mean_intensity = float(cell_pixels.mean()) / 255.0
    return float(np.clip(mean_intensity, 0.0, 1.0))


def shape_regularity_index(cell_mask: np.ndarray) -> float:
    """c4: Deviation from a circular biconcave profile.

    Combines eccentricity and solidity: an elliptocyte has high
    eccentricity, a target-cell / schistocyte has low solidity. The
    returned value is high for irregular cells (poikilocytosis) and low
    for the near-circular shapes of healthy erythrocytes.
    """
    props = measure.regionprops((cell_mask > 0).astype(np.uint8))
    if not props:
        return 0.0
    p = props[0]
    eccentricity = float(p.eccentricity)
    solidity = float(p.solidity)
    # Average two complementary irregularity measures. `eccentricity`
    # already lives in [0, 1]; `1 - solidity` likewise lives in [0, 1].
    irregular = 0.5 * eccentricity + 0.5 * (1.0 - solidity)
    return float(np.clip(irregular, 0.0, 1.0))


def colour_uniformity(cell_mask: np.ndarray, stain_normalized: np.ndarray) -> float:
    """c5: 1 - normalized intra-cell standard deviation of staining.

    High when the cell stains uniformly; low when the cytoplasm is
    anisochromic. Anisochromia across a population is a classic marker
    of early or partially-treated IDA.
    """
    gray = _ensure_single_channel(stain_normalized)
    cell_pixels = gray[cell_mask > 0]
    if cell_pixels.size < 10:
        return 0.0
    std = float(cell_pixels.std())
    # Normalize the std by its theoretical maximum (half of the 0..255
    # dynamic range), then invert so uniform cells score near 1.
    norm_std = min(std / 127.5, 1.0)
    return float(np.clip(1.0 - norm_std, 0.0, 1.0))


# ------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------

@dataclass
class ConceptOperators:
    """Bundle of the five canonical operators.

    The operator functions themselves are module-level and stateless; this
    dataclass is a thin wrapper so that callers can pass a single `ops`
    object around and so that we have an obvious hook if a future version
    of the paper needs to swap operators in or out.
    """

    pallor = staticmethod(central_pallor_ratio)
    diameter = staticmethod(relative_cell_diameter)
    hemoglobinization = staticmethod(hemoglobinization_score)
    shape = staticmethod(shape_regularity_index)
    colour = staticmethod(colour_uniformity)

    @classmethod
    def all(cls) -> Sequence[str]:
        return (
            "central_pallor_ratio",
            "relative_cell_diameter",
            "hemoglobinization_score",
            "shape_regularity_index",
            "colour_uniformity",
        )


def compute_concept_vector(
    cell_mask: np.ndarray,
    raw_intensity: np.ndarray,
    stain_normalized: np.ndarray,
    image_median_diameter: float,
) -> np.ndarray:
    """Compute the 5-dim concept vector for a single cell crop.

    Parameters
    ----------
    cell_mask : (H, W) binary mask of the cell interior
    raw_intensity : (H, W) or (H, W, 3) un-normalized crop (used for
        `central_pallor_ratio`, since Otsu benefits from the original
        contrast)
    stain_normalized : (H, W, 3) Macenko-normalized crop (used for
        `hemoglobinization_score` and `colour_uniformity`)
    image_median_diameter : median equivalent-circle diameter across the
        parent smear (for `relative_cell_diameter`)

    Returns
    -------
    np.ndarray of shape (5,), each entry in [0, 1], in the order
    (`central_pallor_ratio`, `relative_cell_diameter`,
    `hemoglobinization_score`, `shape_regularity_index`,
    `colour_uniformity`).
    """
    return np.asarray([
        central_pallor_ratio(cell_mask, raw_intensity),
        relative_cell_diameter(cell_mask, image_median_diameter),
        hemoglobinization_score(cell_mask, stain_normalized),
        shape_regularity_index(cell_mask),
        colour_uniformity(cell_mask, stain_normalized),
    ], dtype=np.float32)
