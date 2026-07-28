#!/usr/bin/env bash
# v23 N-chain, SINGLE-GPU (one seed per GPU). Same config as run_v23_nchain_ddp.sh
# but no DDP -- for running many seeds / the full-data model in parallel across GPUs.
# Env: GPU_UUID, SEED, DATA, SPLIT, RUN_NAME, OUT_ROOT.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

GPU_UUID="${GPU_UUID:?set GPU_UUID}"
SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-225}"; BATCH_SIZE="${BATCH_SIZE:-12}"; ACCUM="${ACCUM:-3}"
DATA="${DATA:-data/processed/tf_pwm_training_v23.parquet}"
SPLIT="${SPLIT:-data/processed/splits/train_v22/split.json}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v23_seeds}"
RUN_NAME="${RUN_NAME:-nchain_v23_seed${SEED}}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
OUT="$OUT_ROOT/$RUN_NAME"; LOG_DIR="$OUT_ROOT/logs"; LOG="$LOG_DIR/$RUN_NAME.log"
mkdir -p "$OUT" "$LOG_DIR"

args=(
  scripts/train.py --data "$DATA" --split "$SPLIT" --out "$OUT"
  --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --grad-accum-steps "$ACCUM"
  --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6
  --warmup-steps 150 --workers 2 --save-every 25
  --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0
  --pwm-head-v18 --group-balanced-sampling
  --v18-contact-supervision --v18-contact-weight 0.3 --v18-contact-bias-scale 0.0
  --recognition-prior-path data/contact_maps/recognition_residues_cluster40trainonly.json
  --family-embedding-path none
  --moe-granularity residue --num-experts 8 --n-shared-experts 2 --top-k 2
  --expert-hidden-dim 512 --balance-loss-weight 0.01 --diversity-loss-weight 0.0
  --gate-length-weight 0.05 --latent-registration
  --gate-mode span --max-motif-length 42 --motif-overflow-policy error
  --pwm-cov-r-weight 0.25 --pwm-core-ic-thresh 0.25
  --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 0
  --oracle-aggregation gene --early-stop-patience 30
  --two-chain-input --chain-id-embedding --max-chains 4
  --precision bf16 --tf32 --no-wandb
)
echo "v23 single-GPU: seed=$SEED gpu=$GPU_UUID data=$DATA split=$SPLIT"
echo "Log: $LOG"
env CUDA_VISIBLE_DEVICES="$GPU_UUID" TORCH_HOME="$TORCH_HOME" PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$LOG"
