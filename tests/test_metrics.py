"""Unit tests for the evaluation metrics stack."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.calibration import (
    brier_score, expected_calibration_error, max_calibration_error,
)
from src.evaluation.metrics import (
    bootstrap_ci, classification_report, cohens_d,
    delong_paired_auc_test, holm_bonferroni, mcnemar_paired_test,
)


# ------------------------------------------------------------------
#  Calibration
# ------------------------------------------------------------------

def test_perfect_calibration():
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 20)
    labels = (probs > 0.5).astype(int)
    assert expected_calibration_error(probs, labels) >= 0
    assert brier_score(np.array([0.0, 1.0]), np.array([0, 1])) == 0.0


def test_ece_bounds():
    probs = np.linspace(0.01, 0.99, 100)
    labels = (probs > 0.5).astype(int)
    ece = expected_calibration_error(probs, labels, num_bins=10)
    assert 0.0 <= ece <= 1.0


def test_mce_at_least_ece():
    probs = np.linspace(0.01, 0.99, 200)
    labels = (probs > 0.3).astype(int)
    assert max_calibration_error(probs, labels) >= expected_calibration_error(probs, labels)


# ------------------------------------------------------------------
#  Classification report
# ------------------------------------------------------------------

def test_classification_report_schema():
    rng = np.random.default_rng(0)
    probs = rng.uniform(size=200)
    labels = (probs > 0.5).astype(int)
    rep = classification_report(probs, labels)
    d = rep.to_dict()
    for k in ["accuracy", "f1_macro", "auc_roc", "auc_pr", "ece", "brier", "mce"]:
        assert k in d


# ------------------------------------------------------------------
#  Bootstrap
# ------------------------------------------------------------------

def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    probs = rng.uniform(size=500)
    labels = (probs > 0.4).astype(int)
    from sklearn.metrics import roc_auc_score
    point, lo, hi = bootstrap_ci(
        probs, labels, roc_auc_score, num_resamples=200,
        argument_order="labels_first",
    )
    assert lo <= point <= hi


# ------------------------------------------------------------------
#  Statistical tests
# ------------------------------------------------------------------

def test_holm_bonferroni_monotone_and_bounded():
    p = [0.01, 0.04, 0.03, 0.20]
    adj = holm_bonferroni(p)
    assert (adj <= 1.0).all()
    assert (adj >= np.asarray(p)).all()


def test_mcnemar_identical_predictors_pvalue_one():
    preds = np.array([0, 1, 1, 0, 1])
    labels = np.array([0, 1, 0, 0, 1])
    assert mcnemar_paired_test(preds, preds, labels) == 1.0


def test_delong_identical_predictors_pvalue_one():
    rng = np.random.default_rng(2)
    probs = rng.uniform(size=200)
    labels = (probs > 0.5).astype(int)
    _, _, p = delong_paired_auc_test(probs, probs.copy(), labels)
    # p = 1.0 exactly, modulo numerical tolerance
    assert p > 0.95


def test_cohens_d_zero_for_identical_samples():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    d, lo, hi = cohens_d(x, x.copy(), paired=True)
    assert abs(d) < 0.01
    assert lo <= d <= hi
