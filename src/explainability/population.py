"""Population-level smear analysis.

Aggregates per-cell IDA probabilities and per-cell concept vectors
across every cell in a smear to produce a smear-level severity grade
and a structured morphological profile (paper Eq. 13 / Section 3.6.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------
#  Severity grading
# ------------------------------------------------------------------

def severity_from_score(
    score: float,
    thresholds: tuple[float, float, float] = (0.3, 0.5, 0.7),
) -> str:
    """Map an aggregate IDA score to one of 4 severity grades."""
    t1, t2, t3 = thresholds
    if score < t1:
        return "Normal"
    if score < t2:
        return "Mild"
    if score < t3:
        return "Moderate"
    return "Severe"


# ------------------------------------------------------------------
#  Smear-level summary
# ------------------------------------------------------------------

@dataclass
class PopulationSummary:
    """Smear-level IDA summary.

    Attributes
    ----------
    aggregate_score : mean of per-cell IDA probabilities over the smear
    severity : "Normal" | "Mild" | "Moderate" | "Severe"
    concept_mean : (K,) mean concept activations across the smear
    concept_std : (K,) per-concept standard deviation
    cell_count : number of cells after artifact rejection
    """
    aggregate_score: float
    severity: str
    concept_mean: np.ndarray
    concept_std: np.ndarray
    cell_count: int

    @classmethod
    def from_per_cell(
        cls,
        ida_probs: np.ndarray,       # (N,) in [0, 1]
        concept_vectors: np.ndarray, # (N, K) in [0, 1]
        thresholds: tuple[float, float, float] = (0.3, 0.5, 0.7),
    ) -> "PopulationSummary":
        if ida_probs.ndim != 1 or concept_vectors.ndim != 2:
            raise ValueError("ida_probs must be (N,) and concept_vectors (N, K).")
        aggregate = float(ida_probs.mean())
        return cls(
            aggregate_score=aggregate,
            severity=severity_from_score(aggregate, thresholds),
            concept_mean=concept_vectors.mean(axis=0),
            concept_std=concept_vectors.std(axis=0),
            cell_count=int(ida_probs.shape[0]),
        )

    # ------------- display helpers -------------
    def to_dict(self) -> dict[str, object]:
        return {
            "aggregate_score": self.aggregate_score,
            "severity": self.severity,
            "concept_mean": self.concept_mean.tolist(),
            "concept_std": self.concept_std.tolist(),
            "cell_count": self.cell_count,
        }
