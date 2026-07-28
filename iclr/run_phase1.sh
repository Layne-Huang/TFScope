#!/usr/bin/env bash
# ICLR 2026 Phase-I necessity audit launcher (plan §3, §9).
#
# Runs the training-free baselines locally, then prints the exact train.py
# commands for the trained variants B2–B7 across seeds (launch these on the
# GPU node / DDP as appropriate). B8 is the FROZEN v24 checkpoint — never
# retrained. Nothing here tunes on the test set (plan §2 rule 8).
#
# Usage:
#   bash iclr/run_phase1.sh baselines           # run B0, B1 now (CPU, minutes)
#   bash iclr/run_phase1.sh commands            # print B2–B8 train/eval commands
#   SEEDS="42 1 7" bash iclr/run_phase1.sh commands
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

PY="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
DATA="${DATA:-data/processed/tf_pwm_training_v23.parquet}"
SPLIT="${SPLIT:-data/processed/splits/train_v22/split.json}"
OUT_ROOT="${OUT_ROOT:-checkpoints/iclr_phase1}"
SEEDS="${SEEDS:-42 1 7}"                       # ≥3 seeds (plan §2 rule 4)

cmd="${1:-commands}"

case "$cmd" in
  baselines)
    for V in B0 B1; do
      env PYTHONPATH=.:src "$PY" -m iclr.baselines --variant "$V" \
        --train-data "$DATA" --split "$SPLIT" \
        --test-data "$DATA" --test-split "$SPLIT" \
        --out "$OUT_ROOT/$V"
    done
    ;;
  commands)
    echo "# ---- Phase-I trained variants (launch on GPU/DDP) ----"
    for V in B2 B3 B4 B5 B6 B7; do
      for S in $SEEDS; do
        echo
        env PYTHONPATH=.:src "$PY" -m iclr.variants "$V" --seed "$S" --out-root "$OUT_ROOT"
      done
    done
    echo
    echo "# ---- B8: frozen v24 reference (evaluate only) ----"
    env PYTHONPATH=.:src "$PY" -m iclr.variants B8 --out-root "$OUT_ROOT"
    ;;
  *)
    echo "usage: bash iclr/run_phase1.sh [baselines|commands]"; exit 1;;
esac
