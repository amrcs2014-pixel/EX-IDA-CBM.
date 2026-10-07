"""EX-IDA-CBM: Explainable IDA detection via Concept Bottleneck Models.

The top-level package intentionally does NOT re-export anything from
submodules — submodules have varying external dependencies (torch,
cellpose, timm), and importing them eagerly here would force every
caller to install the full stack even if they only want the stateless
helpers in `src.data.concepts` or `src.evaluation.calibration`.
Import what you need directly, e.g.::

    from src.data.concepts import compute_concept_vector
    from src.models.cbm import ConceptBottleneckModel
"""

__version__ = "1.0.0"
__author__ = "Amr H. Abdelhaliem et al."

CONCEPT_NAMES = (
    "Central Pallor Ratio",
    "Relative Cell Diameter",
    "Hemoglobinization Score",
    "Shape Regularity Index",
    "Colour Uniformity",
)
NUM_CONCEPTS = len(CONCEPT_NAMES)
