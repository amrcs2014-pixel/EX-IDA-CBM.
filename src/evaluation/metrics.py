"""Classification metrics, bootstrap CIs, statistical tests.

Supports the full statistical pipeline reported in the paper:
  * Table 3 classification metrics with 95% bootstrap CIs
  * Table 7 ablation table with DeLong p-values, Holm-corrected p-values,
    and Cohen's d
  * Table 2 panoptic / segmentation / recognition quality
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .calibration import brier_score, expected_calibration_error, max_calibration_error


# ------------------------------------------------------------------
#  Classification report
# ------------------------------------------------------------------

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    auc_roc: float
    auc_pr: float
    ece: float
    brier: float
    mce: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def classification_report(
    probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> ClassificationMetrics:
    """Compute every row that appears in Table 3 of the paper."""
    preds = (probs >= threshold).astype(np.int32)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(labels, preds)),
        precision_macro=float(precision_score(labels, preds, average="macro", zero_division=0)),
        recall_macro=float(recall_score(labels, preds, average="macro", zero_division=0)),
        f1_macro=float(f1_score(labels, preds, average="macro", zero_division=0)),
        auc_roc=float(roc_auc_score(labels, probs)),
        auc_pr=float(average_precision_score(labels, probs)),
        ece=float(expected_calibration_error(probs, labels)),
        brier=float(brier_score(probs, labels)),
        mce=float(max_calibration_error(probs, labels)),
    )


# ------------------------------------------------------------------
#  Bootstrap CI
# ------------------------------------------------------------------

def bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    argument_order: str = "labels_first",
) -> tuple[float, float, float]:
    """Return (point_estimate, lower, upper) by paired bootstrap over test items.

    `argument_order` controls how `metric_fn` is invoked:
      - "labels_first" (default, sklearn convention): metric_fn(labels, probs)
      - "probs_first":                                  metric_fn(probs, labels)
    """
    rng = np.random.default_rng(seed)
    if argument_order == "labels_first":
        call = lambda p, y: metric_fn(y, p)
    elif argument_order == "probs_first":
        call = lambda p, y: metric_fn(p, y)
    else:
        raise ValueError(f"Unknown argument_order: {argument_order}")

    point = call(probs, labels)
    n = len(labels)
    stats_ = np.empty(num_resamples)
    for i in range(num_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            stats_[i] = call(probs[idx], labels[idx])
        except Exception:
            stats_[i] = np.nan
    lo = float(np.nanpercentile(stats_, 100 * (1 - confidence) / 2))
    hi = float(np.nanpercentile(stats_, 100 * (1 + confidence) / 2))
    return float(point), lo, hi


# ------------------------------------------------------------------
#  Panoptic / segmentation quality
# ------------------------------------------------------------------

def panoptic_quality(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Compute (PQ, SQ, RQ) for one image (Kirillov et al. 2019).

    pred_mask and gt_mask are integer label maps (0 = background).
    """
    pred_ids = [i for i in np.unique(pred_mask) if i != 0]
    gt_ids = [i for i in np.unique(gt_mask) if i != 0]

    if not pred_ids and not gt_ids:
        return 1.0, 1.0, 1.0

    # IoU matrix between every (pred, gt) pair
    iou = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    for i, pid in enumerate(pred_ids):
        p = pred_mask == pid
        for j, gid in enumerate(gt_ids):
            g = gt_mask == gid
            inter = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            iou[i, j] = inter / union if union > 0 else 0.0

    # Greedy matching
    tp_pairs = []
    used_p, used_g = set(), set()
    while True:
        i, j = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[i, j] <= iou_threshold:
            break
        if i in used_p or j in used_g:
            iou[i, j] = 0.0
            continue
        tp_pairs.append((i, j, float(iou[i, j])))
        used_p.add(i)
        used_g.add(j)
        iou[i, :] = 0.0
        iou[:, j] = 0.0

    tp = len(tp_pairs)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp

    sq = (sum(x[2] for x in tp_pairs) / tp) if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    return float(sq * rq), float(sq), float(rq)


# ------------------------------------------------------------------
#  Statistical tests
# ------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Internal: midrank for DeLong's test (handles ties)."""
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float32)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float32)
    T2[J] = T
    return T2


def delong_paired_auc_test(
    probs_a: np.ndarray, probs_b: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    """DeLong's test for the difference of two paired AUCs.

    Returns (auc_a, auc_b, p_value).
    """
    labels = labels.astype(np.int32)
    pos = labels == 1
    neg = labels == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return (float("nan"), float("nan"), 1.0)

    def _delong_components(y_pred: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Return (AUC, V10, V01) components for a single predictor."""
        tx = _compute_midrank(y_pred[pos])
        ty = _compute_midrank(y_pred[neg])
        tz = _compute_midrank(y_pred)
        auc = (tz[pos].sum() - m * (m + 1) / 2) / (m * n)
        v10 = (tz[pos] - tx) / n
        v01 = 1.0 - (tz[neg] - ty) / m
        return float(auc), v10, v01

    auc_a, v10_a, v01_a = _delong_components(probs_a)
    auc_b, v10_b, v01_b = _delong_components(probs_b)

    sx = np.cov(np.vstack([v10_a, v10_b]))
    sy = np.cov(np.vstack([v01_a, v01_b]))
    s = sx / m + sy / n
    # Variance of (AUC_a - AUC_b)
    var = s[0, 0] + s[1, 1] - 2.0 * s[0, 1]
    if var <= 0:
        return auc_a, auc_b, 1.0
    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, float(p)


def mcnemar_paired_test(
    preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray
) -> float:
    """Return the p-value of the paired McNemar test."""
    a_correct = preds_a == labels
    b_correct = preds_b == labels
    b_only = int(np.logical_and(a_correct, np.logical_not(b_correct)).sum())
    a_only = int(np.logical_and(np.logical_not(a_correct), b_correct).sum())
    n = a_only + b_only
    if n == 0:
        return 1.0
    # Exact binomial test (B(n, 0.5))
    k = min(a_only, b_only)
    return float(2.0 * stats.binom.cdf(k, n, 0.5))


def cohens_d(
    sample_a: np.ndarray, sample_b: np.ndarray, paired: bool = True
) -> tuple[float, float, float]:
    """Return (d, lower, upper) with a 95% CI (bootstrap, 1000 resamples)."""
    if paired:
        diff = sample_a - sample_b
        d = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    else:
        pooled = np.sqrt(0.5 * (sample_a.var(ddof=1) + sample_b.var(ddof=1)))
        d = float((sample_a.mean() - sample_b.mean()) / (pooled + 1e-12))

    # Bootstrap CI
    rng = np.random.default_rng(42)
    n = len(sample_a)
    ds = np.empty(1000)
    for i in range(1000):
        idx = rng.integers(0, n, size=n)
        if paired:
            diff_b = sample_a[idx] - sample_b[idx]
            ds[i] = diff_b.mean() / (diff_b.std(ddof=1) + 1e-12)
        else:
            jdx = rng.integers(0, n, size=n)
            pooled = np.sqrt(0.5 * (sample_a[idx].var(ddof=1) + sample_b[jdx].var(ddof=1)))
            ds[i] = (sample_a[idx].mean() - sample_b[jdx].mean()) / (pooled + 1e-12)
    lo = float(np.percentile(ds, 2.5))
    hi = float(np.percentile(ds, 97.5))
    return d, lo, hi


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Step-down Holm-Bonferroni correction.

    Returns an array of adjusted p-values of the same length as `p_values`.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = (m - rank) * p[idx]
        corrected = min(corrected, 1.0)
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max
    return adjusted
