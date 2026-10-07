"""Calibration metrics: ECE, MCE, Brier score.

All three are reported in Table 3 of the paper; they are *complementary*
because ECE collapses an entire reliability diagram into a single
average-case number, MCE identifies the worst-case bin, and Brier score
penalizes both calibration and refinement jointly.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _validate(probs: np.ndarray, labels: np.ndarray) -> None:
    if probs.ndim != 1 or labels.ndim != 1:
        raise ValueError("probs and labels must be 1-D arrays.")
    if probs.shape != labels.shape:
        raise ValueError("probs and labels must have the same shape.")
    if ((probs < 0) | (probs > 1)).any():
        raise ValueError("probs must live in [0, 1].")


def _reliability_bins(
    probs: np.ndarray, labels: np.ndarray, num_bins: int
) -> list[tuple[int, float, float]]:
    """Return (count, mean_confidence, empirical_accuracy) per bin."""
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Include the right edge in the final bin so prob=1.0 is counted
        if hi == edges[-1]:
            in_bin = (probs >= lo) & (probs <= hi)
        else:
            in_bin = (probs >= lo) & (probs < hi)
        n = int(in_bin.sum())
        if n == 0:
            out.append((0, 0.0, 0.0))
        else:
            out.append((n, float(probs[in_bin].mean()), float(labels[in_bin].mean())))
    return out


# ------------------------------------------------------------------
#  Metrics
# ------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, num_bins: int = 15
) -> float:
    """Expected Calibration Error (Guo et al. 2017)."""
    _validate(probs, labels)
    bins = _reliability_bins(probs, labels, num_bins)
    N = len(probs)
    ece = sum((n / N) * abs(conf - acc) for n, conf, acc in bins if n > 0)
    return float(ece)


def max_calibration_error(
    probs: np.ndarray, labels: np.ndarray, num_bins: int = 15
) -> float:
    """Maximum (worst-bin) calibration error."""
    _validate(probs, labels)
    bins = _reliability_bins(probs, labels, num_bins)
    return float(max((abs(conf - acc) for n, conf, acc in bins if n > 0), default=0.0))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary labels."""
    _validate(probs, labels)
    return float(np.mean((probs - labels) ** 2))
