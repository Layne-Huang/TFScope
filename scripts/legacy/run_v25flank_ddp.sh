#!/usr/bin/env bash
# v25flank: EXACT v24 recipe, but trained on DBD+20aa-flank data (dbd marked
# internally via dbd_start:dbd_end) with flank-reindexed contact/recognition
# targets. Tests whether flanking protein context restores mutation sensitivity.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
# 3 free GPUs, numeric indices (UUID + CUDA_DEVICE_ORDER broke CUDA enumeration)
GPUS="7,8,9"
NPROC=3; MASTER_PORT=29608; BATCH_SIZE=12; ACCUM=1; SEED=42; EPOCHS=225
OUT="checkpoints/iclr_phase1/v25flank/seed${SEED}"; mkdir -p "$OUT"
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python

args=(
  scripts/train.py
  --data data/processed/tf_pwm_training_v25flank.parquet
  --split data/processed/splits/train_v22/split.json --out "$OUT"
  --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --grad-accum-steps "$ACCUM"
  --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6
  --warmup-steps 150 --workers 2 --save-every 25
  --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0
  --pwm-head-v18 --group-balanced-sampling
  --v18-contact-supervision --v18-contact-weight 0.3
  --v18-contact-bias-scale 1.0 --v18-contact-bias-learnable 1
  --contact-distill-weight 0.2 --contact-targets-path data/contact_maps/contact_targets_v25flank.json
  --recognition-prior-path data/contact_maps/recognition_residues_v25flank.json
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
echo "[$(date +%T)] v25flank DDP $NPROC GPUs | global batch=$((NPROC*BATCH_SIZE*ACCUM))"
env CUDA_VISIBLE_DEVICES="$GPUS" TORCH_HOME=/data1/leihuang/.cache/torch PYTHONUNBUFFERED=1 PYTHONPATH=src \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" "${args[@]}"
echo "[$(date +%T)] v25flank DONE" | tee "$OUT/DONE"
