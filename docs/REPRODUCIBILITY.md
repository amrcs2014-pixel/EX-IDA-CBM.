# Reproducibility Guide

This document is the operator's manual for reproducing every number
reported in the manuscript.

## Hardware & software baseline

| Item | Setting |
|---|---|
| GPU | 1 × NVIDIA T4 (16 GB VRAM) |
| CPU | 4 vCPUs, 16 GB RAM |
| OS | Ubuntu 22.04 LTS |
| Python | 3.10 |
| PyTorch | 2.1.* with CUDA 11.8 |
| cuDNN | deterministic kernels enabled |
| RNG seed | 42 for the headline model; {7, 13, 21, 42, 101} for the multi-seed sweep |

If you have no GPU, the same code paths run on CPU — expect ~90 s per smear instead of ~30 s.

## Dataset layout

```
data/
├── AneRBC-I/
│   ├── healthy/*.png     # 500 PBS images
│   └── anemic/*.png      # 500 PBS images
├── Chula-RBC-12/
│   ├── images/*.jpg
│   └── annotations.json  # {image, class, bbox, split}
└── stain_reference/
    └── reference_slide.png
```

The split manifest (`data/splits/aner_bc_split_42.json`) is built once with:

```bash
python scripts/build_split_manifests.py --seed 42
```

That file freezes the 70/15/15 image-level stratified split. All downstream training/evaluation reads from it, so re-runs are bit-reproducible given the same seed.

## End-to-end reproduction

Shortest path:

```bash
bash scripts/run_all.sh
```

Longer path (useful for staged checkpointing):

```bash
# 1. Build the frozen split manifest
python scripts/build_split_manifests.py --seed 42

# 2. Phase-1 pretraining  (produces Table 1)
python scripts/run_phase1_pretraining.py --config configs/phase1.yaml

# 3. Phase-2 fine-tuning on the seed-42 run  (produces checkpoint used by every ablation)
python scripts/run_phase2_finetuning.py --config configs/phase2.yaml --seed 42

# 4. Headline metrics  (produces Table 3)
python scripts/run_evaluation.py --config configs/phase2.yaml \
    --checkpoint results/logs/phase2_seed42/best.pt

# 5. Each table from its own script
python scripts/run_ablation.py --kind backbone   --config configs/phase2.yaml
python scripts/run_ablation.py --kind concepts   --config configs/phase2.yaml
python scripts/run_ablation.py --kind components --config configs/phase2.yaml --seeds 7 13 21 42 101
python scripts/run_hp_sensitivity.py --config configs/phase2.yaml
python scripts/run_intervention.py --config configs/phase2.yaml \
    --checkpoint results/logs/phase2_seed42/best.pt
python scripts/run_multiseed.py --config configs/phase2.yaml --seeds 7 13 21 42 101
```

Every script writes CSV + JSON outputs under `results/tables/`.

## Expected runtimes (one T4)

| Stage | Wall-clock |
|---|---|
| Phase-1 pretraining (100 epochs) | ~6 h |
| Phase-2 fine-tuning (one seed) | ~1 h |
| Backbone ablation (3 backbones) | ~3 h |
| Concept-dimensionality ablation (3 configs) | ~3 h |
| Component ablation (7 configs × 5 seeds) | ~35 h |
| HP sensitivity sweep (12 settings) | ~12 h |
| Multi-seed sweep (5 seeds) | ~5 h |
| **Total end-to-end** | **~24-72 h depending on skips** |

For a quick sanity check, run `pytest -q` — the unit tests in `tests/` cover the concept operators and the metric stack and complete in under a minute on CPU.

## Reproducibility checklist

* [x] Fixed random seed (42) with `torch.backends.cudnn.deterministic = True`
* [x] Image-level (= subject-level) stratified split, frozen in a JSON manifest
* [x] Validation split used for early stopping / checkpoint selection
* [x] Test split scored once with the frozen checkpoint
* [x] Bootstrap 95% CIs (1,000 resamples) for every reported point estimate
* [x] 5-seed sweep confirms the single-seed figure is stable
* [x] One-at-a-time HP sensitivity sweep covering the four hyperparameters not manually tuned
* [x] Holm-Bonferroni correction over the 9-member family of ablation + intervention tests
* [x] Cohen's d reported alongside p-values for every paired comparison
