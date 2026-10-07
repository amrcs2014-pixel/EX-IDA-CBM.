#!/usr/bin/env bash
# End-to-end reproduction of every table and figure in the paper.
# Assumes AneRBC-I and Chula-RBC-12 are already under data/.
# Runtime: ~24 h on a single NVIDIA T4 GPU.

set -euo pipefail

CONFIG_P1="configs/phase1.yaml"
CONFIG_P2="configs/phase2.yaml"
LOG_DIR="results/logs"

echo "============================================================"
echo "  EX-IDA-CBM — end-to-end reproduction"
echo "============================================================"

echo "[0/8] Building AneRBC-I split manifest ..."
python scripts/build_split_manifests.py --seed 42

echo "[1/8] Phase-1 Chula-RBC-12 pretraining (Table 1) ..."
python scripts/run_phase1_pretraining.py --config "$CONFIG_P1"

echo "[2/8] Phase-2 AneRBC-I fine-tuning (seed 42) ..."
python scripts/run_phase2_finetuning.py --config "$CONFIG_P2" --seed 42

echo "[3/8] Headline evaluation (Table 3) ..."
python scripts/run_evaluation.py --config "$CONFIG_P2" \
    --checkpoint "$LOG_DIR/phase2_seed42/best.pt"

echo "[4/8] Backbone ablation (Table 5) ..."
python scripts/run_ablation.py --kind backbone --config "$CONFIG_P2"

echo "[5/8] Concept-dimensionality ablation (Table 6) ..."
python scripts/run_ablation.py --kind concepts --config "$CONFIG_P2"

echo "[6/8] Component ablation with 5 seeds (Table 7) ..."
python scripts/run_ablation.py --kind components --config "$CONFIG_P2" \
    --seeds 7 13 21 42 101

echo "[7/8] Hyperparameter sensitivity sweep (Table 9) ..."
python scripts/run_hp_sensitivity.py --config "$CONFIG_P2"

echo "[8/8] Intervention experiment (Table 8) and 5-seed sweep ..."
python scripts/run_intervention.py --config "$CONFIG_P2" \
    --checkpoint "$LOG_DIR/phase2_seed42/best.pt"
python scripts/run_multiseed.py --config "$CONFIG_P2" --seeds 7 13 21 42 101

echo
echo "All tables written under results/tables/:"
ls -1 results/tables/
