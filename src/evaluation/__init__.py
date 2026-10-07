"""Evaluation metrics, bootstrap CIs, calibration metrics, statistical tests.

All submodules are torch-free, so they can be imported standalone.
"""

from .calibration import brier_score, expected_calibration_error, max_calibration_error
from .metrics import (
    bootstrap_ci, classification_report, cohens_d, delong_paired_auc_test,
    holm_bonferroni, mcnemar_paired_test, panoptic_quality,
)

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "max_calibration_error",
    "bootstrap_ci",
    "classification_report",
    "cohens_d",
    "delong_paired_auc_test",
    "holm_bonferroni",
    "mcnemar_paired_test",
    "panoptic_quality",
]
