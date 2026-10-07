"""Unit tests for the deterministic concept operators.

The paper's audit protocol depends on these operators producing the same
output for the same input, every time. These tests pin the behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.concepts import (
    central_pallor_ratio,
    colour_uniformity,
    compute_concept_vector,
    hemoglobinization_score,
    relative_cell_diameter,
    shape_regularity_index,
)


# ------------------------------------------------------------------
#  Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def round_cell() -> tuple[np.ndarray, np.ndarray]:
    """A nicely rounded, uniformly-stained synthetic cell."""
    H = W = 128
    mask = np.zeros((H, W), dtype=np.uint8)
    cx, cy, r = W // 2, H // 2, 40
    yy, xx = np.mgrid[:H, :W]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
    mask[circle] = 1
    intensity = np.full((H, W, 3), 128, dtype=np.uint8)
    intensity[circle] = (120, 80, 100)  # eosinophilic
    return mask, intensity


@pytest.fixture
def pale_cell() -> tuple[np.ndarray, np.ndarray]:
    """A cell with a very pronounced central pallor."""
    H = W = 128
    mask = np.zeros((H, W), dtype=np.uint8)
    cx, cy = W // 2, H // 2
    yy, xx = np.mgrid[:H, :W]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 < 40 ** 2
    inner = (xx - cx) ** 2 + (yy - cy) ** 2 < 25 ** 2
    mask[circle] = 1
    intensity = np.full((H, W, 3), 220, dtype=np.uint8)
    intensity[circle] = (120, 80, 100)
    intensity[inner] = (240, 230, 240)  # pale center
    return mask, intensity


# ------------------------------------------------------------------
#  Tests
# ------------------------------------------------------------------

def test_concepts_output_shape_and_range(round_cell):
    mask, intensity = round_cell
    v = compute_concept_vector(mask, intensity, intensity, image_median_diameter=40.0)
    assert v.shape == (5,)
    assert v.dtype == np.float32
    assert ((v >= 0) & (v <= 1)).all()


def test_central_pallor_higher_for_pale(round_cell, pale_cell):
    r_mask, r_img = round_cell
    p_mask, p_img = pale_cell
    assert central_pallor_ratio(p_mask, p_img) > central_pallor_ratio(r_mask, r_img)


def test_relative_cell_diameter_normalization(round_cell):
    mask, _ = round_cell
    # If we tell the operator the "image-median diameter" is enormous, our
    # cell should look much smaller than 1 on the normalized scale.
    small = relative_cell_diameter(mask, image_median_diameter=1000.0)
    large = relative_cell_diameter(mask, image_median_diameter=10.0)
    assert small < large


def test_shape_regularity_is_low_for_circle(round_cell):
    mask, _ = round_cell
    assert shape_regularity_index(mask) < 0.4  # near-circular -> low irregularity


def test_colour_uniformity_is_high_for_flat_cell(round_cell):
    mask, img = round_cell
    assert colour_uniformity(mask, img) > 0.7  # uniformly stained


def test_hemoglobinization_higher_for_pale(round_cell, pale_cell):
    r_mask, r_img = round_cell
    p_mask, p_img = pale_cell
    # Lower stain intensity -> higher "hypochromia-ness"
    assert hemoglobinization_score(p_mask, p_img) > hemoglobinization_score(r_mask, r_img)
