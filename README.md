# EX-IDA-CBM: Explainable Iron Deficiency Anemia Detection via Concept Bottleneck Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

Official implementation of the paper:

> **Explainable Iron Deficiency Anemia Detection from Peripheral Blood Smears via Concept Bottleneck Models and Attention-Guided Morphological Maps**


A four-stage, inherently-interpretable pipeline that classifies individual red blood cells (RBCs) from peripheral blood smears (PBS) as **normal** or **Iron-Deficiency-Anemia (IDA) positive** by routing every prediction through five clinically grounded morphological concepts.

---

## Headline results (AneRBC-I test partition)

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| Accuracy | **98.9%** | [98.4%, 99.3%] |
| Macro F1-score | **97.8%** | [97.1%, 98.4%] |
| AUC-ROC | **0.993** | [0.989, 0.996] |
| AUC-PR | 0.991 | [0.986, 0.995] |
| Expected Calibration Error (ECE) | 0.043 | [0.035, 0.051] |
| Brier score | 0.0097 | [0.0082, 0.0114] |
| Maximum Calibration Error (MCE) | 0.072 | [0.055, 0.091] |

Single-concept clinician-in-the-loop intervention recovers **72.7%** of misclassified cells; two-concept intervention recovers **90.9%** of misclassifications, lifting overall accuracy from 98.9% to 99.9%.

---

## Pipeline overview

```
     ┌─────────────────────────────────────────────────────────────┐
     │  Raw PBS image I ∈ ℝ^{H×W×3}                                │
     └──────────────────────────┬──────────────────────────────────┘
                                │
   Stage 1 ────────────────────▼───────────────────────
    Cellpose 3.0 instance segmentation
    + marker-controlled watershed
    + 2-criterion artifact rejection (eccentricity, area)
                                │
   Stage 2 ────────────────────▼───────────────────────
    EfficientNet-B0 backbone
    Phase 1: pretrain on Chula-RBC-12 (12-class morphology)
    Phase 2: fine-tune on AneRBC-I (binary IDA)
                                │
   Stage 3 ────────────────────▼───────────────────────
    Concept Bottleneck Model (CBM):
      c₁  Central Pallor Ratio
      c₂  Relative Cell Diameter
      c₃  Hemoglobinization Score
      c₄  Shape Regularity Index
      c₅  Colour Uniformity
    → 2-layer classifier over c ∈ [0,1]^5
                                │
   Stage 4 ────────────────────▼───────────────────────
    Multi-scale explanations:
      cell level:        Grad-CAM++ + Integrated Gradients
      concept level:     per-concept signed contributions
      population level:  smear severity grade + concept summary
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/amrogomma-web/EX-IDA-CBM.git
cd EX-IDA-CBM

# Create a conda environment (recommended)
conda env create -f environment.yml
conda activate ex-ida-cbm

# OR use pip
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA 11.8+ (optional — the pipeline also runs on CPU). Full inference on a 1,000-cell smear takes ~30 s on a single NVIDIA T4 GPU, ~90 s on CPU.

---

## Datasets

Both datasets are publicly available:

| Dataset | Role | Access |
|---|---|---|
| **AneRBC-I** (1,000 PBS images, 500 healthy / 500 IDA) | Primary IDA fine-tuning + evaluation | Via original publication: Shahzad et al., *Database* 2024, baae120, [doi:10.1093/database/baae120](https://doi.org/10.1093/database/baae120) |
| **Chula-RBC-12** (706 images, >20,000 cells, 12 morphology classes) | Phase-1 same-domain pretraining | <https://github.com/Chula-PIC-Lab/Chula-RBC-12-Dataset> |

Place both corpora under `data/`:

```
data/
├── AneRBC-I/
│   ├── healthy/
│   └── anemic/
└── Chula-RBC-12/
    ├── images/
    └── annotations/
```

Then run the one-off split-manifest builder (seeded for reproducibility):

```bash
python scripts/build_split_manifests.py --seed 42
```

---

## Reproducing the paper's results

Every table and figure in the paper is produced by a dedicated script.

### 1. Phase-1 pretraining (Table 1)

```bash
python scripts/run_phase1_pretraining.py --config configs/phase1.yaml
```

Writes to `results/logs/phase1/` and `results/tables/table1_phase1_morph.csv`.

### 2. Phase-2 fine-tuning + headline metrics (Tables 2, 3)

```bash
python scripts/run_phase2_finetuning.py --config configs/phase2.yaml --seed 42
python scripts/run_evaluation.py --checkpoint results/logs/phase2/best.pt
```

Writes to `results/tables/table2_segmentation.csv` and `results/tables/table3_classification.csv`.

### 3. Backbone comparison ablation (Table 5)

```bash
python scripts/run_ablation.py --kind backbone --config configs/phase2.yaml
```

### 4. Concept-dimensionality ablation (Table 6)

```bash
python scripts/run_ablation.py --kind concepts --config configs/phase2.yaml
```

### 5. Component ablation with Holm-corrected p-values and Cohen's d (Table 7)

```bash
python scripts/run_ablation.py --kind components --config configs/phase2.yaml --seeds 7 13 21 42 101
```

### 6. Intervention experiment (Table 8)

```bash
python scripts/run_intervention.py --checkpoint results/logs/phase2/best.pt
```

### 7. Hyperparameter sensitivity sweep (Table 9)

```bash
python scripts/run_hp_sensitivity.py --config configs/phase2.yaml
```

### 8. Five-seed reproducibility sweep (Section 5.9)

```bash
python scripts/run_multiseed.py --seeds 7 13 21 42 101 --config configs/phase2.yaml
```

### Reproduce everything at once

```bash
bash scripts/run_all.sh
```

Full end-to-end reproduction (one T4 GPU) takes ~24 h for Phase-1 + Phase-2 + five seeds + all ablations.

---

## Repository layout

```
EX-IDA-CBM/
├── configs/                   # YAML configurations for every experiment
├── docs/                      # Reproducibility and concept-protocol docs
├── notebooks/                 # Figure-reproduction notebooks
├── results/                   # Where tables, figures, and logs land
├── scripts/                   # Entry points for every experiment
├── src/
│   ├── data/                  # Datasets, Macenko stain normalization,
│   │                          # deterministic concept operators, augmentation
│   ├── models/                # Backbone, CBM, Cellpose+watershed segmentation
│   ├── explainability/        # Grad-CAM++, Integrated Gradients, population
│   ├── evaluation/            # ECE, Brier, MCE, Cohen's d, Holm-Bonferroni
│   └── training/              # Phase-1 pretrain, Phase-2 finetune, intervention
└── tests/                     # Unit tests for concept operators and metrics
```

---

## Clinical concept operators

Each of the five concepts is computed deterministically from the segmented single-cell mask and its intensity image, so that two runs on the same cell produce the same concept vector:

| Concept | Operator |
|---|---|
| `Central Pallor Ratio` (c₁) | Area fraction of the interior pale region (intensity above a per-cell Otsu threshold within the cell mask) relative to total cell area |
| `Relative Cell Diameter` (c₂) | Equivalent-circle diameter from `skimage.measure.regionprops` normalized by the per-image median cell diameter |
| `Hemoglobinization Score` (c₃) | Mean stain-normalized cytoplasmic intensity (inverted so hypochromia yields high scores) |
| `Shape Regularity Index` (c₄) | Function of eccentricity and solidity quantifying deviation from a circular biconcave profile |
| `Colour Uniformity` (c₅) | 1 − normalized intra-cell standard deviation of staining intensity |

All operators are applied after Macenko stain normalization against a fixed reference. Full definitions and the two-hematologist audit protocol are in [`docs/CONCEPT_PROTOCOL.md`](docs/CONCEPT_PROTOCOL.md).

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{abdelhaliem2026exidacbm,
    title   = {Explainable Iron Deficiency Anemia Detection from Peripheral Blood Smears
               via Concept Bottleneck Models and Attention-Guided Morphological Maps},
    author  = {Abdelhaliem, Amr H. and Tawfik, Mohammed and Shaban, Ahmed S. and
               Fathi, Islam S. and Heshmat, Mohamed},
    journal = {Neural Computing and Applications},
    year    = {2026},
    note    = {Under review}
}
```

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).

The AneRBC-I and Chula-RBC-12 datasets have their own licences; please consult the original publications.

---

## Correspondence

For questions or collaboration enquiries, please contact the corresponding author.

- **Amr H. Abdelhaliem** — Head, Cyber Security Department, Faculty of Science and Information Technology, Irbid National University, Jordan.
- GitHub: [@amrogomma-web](https://github.com/amrogomma-web)
