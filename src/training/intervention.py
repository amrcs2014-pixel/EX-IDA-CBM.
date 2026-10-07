"""Concept-level intervention experiment (Section 4.6.4 of the paper).

For each misclassified test cell, we identify the "most erroneous"
concept as the one whose predicted value is furthest from the
deterministic morphometric reference. We then overwrite that single
concept's score with the reference value and re-run the frozen
classifier head. Table 8 of the paper reports:
    - 1-concept override: 72.7% error recovery (acc 98.9% -> 99.7%)
    - 2-concept override: 90.9% error recovery (acc 98.9% -> 99.9%)

A decorrelation stress test is also supported: for each misclassified
cell, overwrite ONLY c_3 (hemoglobinization) with its reference value
and keep the three correlated concepts fixed. Section 5.7 of the paper
reports 64.7% false-negative recovery under this stricter protocol —
the 23.5-point gap vs. standard single-concept recovery quantifies
correlation-mediated vs genuine single-concept recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models.cbm import ConceptBottleneckModel


@dataclass
class InterventionReport:
    baseline_accuracy: float
    one_concept_accuracy: float
    two_concept_accuracy: float
    one_concept_recovered: int
    two_concept_recovered: int
    num_errors: int
    # Stratified
    one_concept_fn_recovered: int
    one_concept_fp_recovered: int
    two_concept_fn_recovered: int
    two_concept_fp_recovered: int
    num_fn: int
    num_fp: int
    # Decorrelation stress test
    decorrelation_fn_recovered: int

    def as_table(self) -> list[dict[str, object]]:
        return [
            {"setting": "No intervention (baseline)",
             "fn_recovered": f"0 / {self.num_fn} (0%)",
             "fp_recovered": f"0 / {self.num_fp} (0%)",
             "accuracy": f"{self.baseline_accuracy * 100:.1f}%"},
            {"setting": "Correct 1 concept per cell",
             "fn_recovered": (f"{self.one_concept_fn_recovered} / {self.num_fn} "
                              f"({100 * self.one_concept_fn_recovered / max(self.num_fn, 1):.1f}%)"),
             "fp_recovered": (f"{self.one_concept_fp_recovered} / {self.num_fp} "
                              f"({100 * self.one_concept_fp_recovered / max(self.num_fp, 1):.1f}%)"),
             "accuracy": f"{self.one_concept_accuracy * 100:.1f}%"},
            {"setting": "Correct up to 2 concepts",
             "fn_recovered": (f"{self.two_concept_fn_recovered} / {self.num_fn} "
                              f"({100 * self.two_concept_fn_recovered / max(self.num_fn, 1):.1f}%)"),
             "fp_recovered": (f"{self.two_concept_fp_recovered} / {self.num_fp} "
                              f"({100 * self.two_concept_fp_recovered / max(self.num_fp, 1):.1f}%)"),
             "accuracy": f"{self.two_concept_accuracy * 100:.1f}%"},
        ]


def _single_concept_intervention(
    model: ConceptBottleneckModel,
    images: torch.Tensor,
    reference: torch.Tensor,
    top_k: int,
    device: str,
    fixed_concept: int | None = None,
) -> np.ndarray:
    """Return predicted labels after overriding `top_k` most-erroneous concepts."""
    model.eval()
    with torch.no_grad():
        features = model.backbone(images.to(device))
        concepts_pred = model.concept_predictor(features)       # (B, K)
        err = (concepts_pred - reference.to(device)).abs()      # (B, K)
        if fixed_concept is not None:
            mask = torch.zeros_like(err, dtype=torch.bool)
            mask[:, fixed_concept] = True
        else:
            _, top_idx = err.topk(top_k, dim=-1)
            mask = torch.zeros_like(err, dtype=torch.bool)
            mask.scatter_(1, top_idx, True)
        concepts_corrected = torch.where(mask, reference.to(device), concepts_pred)
        logits = model.classifier(concepts_corrected)
        return logits.argmax(dim=-1).cpu().numpy()


def run_intervention_experiment(
    model: ConceptBottleneckModel,
    images: torch.Tensor,
    labels: np.ndarray,
    reference_concepts: torch.Tensor,
    device: str = "cuda",
) -> InterventionReport:
    """Produce the entries of Table 8 for a given model and test set."""
    model.eval()
    with torch.no_grad():
        baseline_preds = model(images.to(device)).argmax(dim=-1).cpu().numpy()

    baseline_correct = baseline_preds == labels
    err_idx = np.where(~baseline_correct)[0]
    num_errors = len(err_idx)
    num_fn = int(np.sum((~baseline_correct) & (labels == 1)))
    num_fp = int(np.sum((~baseline_correct) & (labels == 0)))

    if num_errors == 0:
        return InterventionReport(1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # ----- 1-concept intervention on misclassified cells only -----
    corrected_preds_1 = _single_concept_intervention(
        model, images[err_idx], reference_concepts[err_idx], top_k=1, device=device,
    )
    one_c_recovered = int((corrected_preds_1 == labels[err_idx]).sum())
    one_c_fn = int(((corrected_preds_1 == labels[err_idx]) & (labels[err_idx] == 1)).sum())
    one_c_fp = int(((corrected_preds_1 == labels[err_idx]) & (labels[err_idx] == 0)).sum())

    # ----- 2-concept intervention -----
    corrected_preds_2 = _single_concept_intervention(
        model, images[err_idx], reference_concepts[err_idx], top_k=2, device=device,
    )
    two_c_recovered = int((corrected_preds_2 == labels[err_idx]).sum())
    two_c_fn = int(((corrected_preds_2 == labels[err_idx]) & (labels[err_idx] == 1)).sum())
    two_c_fp = int(((corrected_preds_2 == labels[err_idx]) & (labels[err_idx] == 0)).sum())

    # ----- Decorrelation stress test: fix c_3 (index 2) only -----
    decorr_preds = _single_concept_intervention(
        model, images[err_idx], reference_concepts[err_idx], top_k=1,
        device=device, fixed_concept=2,
    )
    decorr_fn = int(
        ((decorr_preds == labels[err_idx]) & (labels[err_idx] == 1)).sum()
    )

    baseline_acc = float(baseline_correct.mean())
    one_c_acc = baseline_acc + one_c_recovered / len(labels)
    two_c_acc = baseline_acc + two_c_recovered / len(labels)

    return InterventionReport(
        baseline_accuracy=baseline_acc,
        one_concept_accuracy=one_c_acc,
        two_concept_accuracy=two_c_acc,
        one_concept_recovered=one_c_recovered,
        two_concept_recovered=two_c_recovered,
        num_errors=num_errors,
        one_concept_fn_recovered=one_c_fn,
        one_concept_fp_recovered=one_c_fp,
        two_concept_fn_recovered=two_c_fn,
        two_concept_fp_recovered=two_c_fp,
        num_fn=num_fn,
        num_fp=num_fp,
        decorrelation_fn_recovered=decorr_fn,
    )
