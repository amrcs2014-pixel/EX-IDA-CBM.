"""Stage-1 segmentation: Cellpose + marker-controlled watershed + artifact rejection.

See paper Section 3.3 for the full algorithmic description. The pipeline
is implemented against the public `cellpose` package (we use `cyto2`
pretrained weights by default; the dependency is optional at import
time so the rest of the project is still importable on machines without
Cellpose).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

try:
    from cellpose import models as cp_models
    _HAS_CELLPOSE = True
except ImportError:  # pragma: no cover
    _HAS_CELLPOSE = False


# ------------------------------------------------------------------
#  Dataclasses
# ------------------------------------------------------------------

@dataclass
class SegmentationResult:
    label_map: np.ndarray             # (H, W) int32, 0 = background
    cell_count: int
    mean_cell_area: float
    artifact_rejection_rate: float    # fraction of initial Cellpose
                                      # detections rejected by the
                                      # eccentricity+area filter
    median_diameter: float            # used downstream by `compute_concept_vector`


# ------------------------------------------------------------------
#  Pipeline
# ------------------------------------------------------------------

class SegmentationPipeline:
    """Cellpose instance segmentation + watershed + artifact rejection."""

    def __init__(
        self,
        model_type: str = "cyto2",
        gpu: bool = True,
        min_marker_distance: int = 10,
        max_eccentricity: float = 0.85,
        area_z_threshold: float = 3.0,
    ) -> None:
        if not _HAS_CELLPOSE:
            raise ImportError("cellpose is required: `pip install cellpose`.")
        self.model = cp_models.Cellpose(gpu=gpu, model_type=model_type)
        self.min_marker_distance = min_marker_distance
        self.max_eccentricity = max_eccentricity
        self.area_z_threshold = area_z_threshold

    # ------------- step 1: Cellpose -------------
    def _cellpose(self, image: np.ndarray) -> np.ndarray:
        masks, _flows, _styles, _diams = self.model.eval(
            image, channels=[0, 0], diameter=None,
        )
        return masks.astype(np.int32)

    # ------------- step 2: watershed refinement -------------
    def _watershed_refine(self, masks: np.ndarray) -> np.ndarray:
        binary = masks > 0
        distance = ndi.distance_transform_edt(binary)
        local_max = peak_local_max(
            distance,
            min_distance=self.min_marker_distance,
            labels=binary,
        )
        markers = np.zeros_like(masks, dtype=np.int32)
        for i, (y, x) in enumerate(local_max, start=1):
            markers[y, x] = i
        if markers.max() == 0:
            return masks
        refined = watershed(-distance, markers, mask=binary)
        return refined.astype(np.int32)

    # ------------- step 3: artifact rejection -------------
    def _reject_artifacts(self, masks: np.ndarray) -> tuple[np.ndarray, float]:
        props = regionprops(masks)
        if not props:
            return masks, 0.0
        areas = np.array([p.area for p in props], dtype=np.float32)
        mu, sigma = areas.mean(), areas.std()

        keep = np.zeros_like(masks, dtype=np.int32)
        next_id = 1
        rejected = 0
        for p in props:
            if p.eccentricity > self.max_eccentricity:
                rejected += 1
                continue
            if p.area < mu - self.area_z_threshold * sigma:
                rejected += 1
                continue
            keep[masks == p.label] = next_id
            next_id += 1
        rejection_rate = rejected / len(props) if props else 0.0
        return keep, rejection_rate

    # ------------- public API -------------
    def __call__(self, image: np.ndarray) -> SegmentationResult:
        cellpose_masks = self._cellpose(image)
        refined = self._watershed_refine(cellpose_masks)
        cleaned, rej_rate = self._reject_artifacts(refined)

        props = regionprops(cleaned)
        if not props:
            return SegmentationResult(cleaned, 0, 0.0, rej_rate, 0.0)

        areas = np.array([p.area for p in props], dtype=np.float32)
        diameters = 2.0 * np.sqrt(areas / np.pi)
        return SegmentationResult(
            label_map=cleaned,
            cell_count=len(props),
            mean_cell_area=float(areas.mean()),
            artifact_rejection_rate=rej_rate,
            median_diameter=float(np.median(diameters)),
        )

    # ------------- fallback for environments without cellpose -------------
    @staticmethod
    def available() -> bool:
        return _HAS_CELLPOSE
