#!/usr/bin/env bash
# v23: N-chain (order-aware) multichain, 6-GPU DDP.
# = v22 "multichain" stage config (repaired data + span gate + covR loss +
#   chain-id embedding) BUT with max_chains=4 and the N-chain v23 data, so
#   p53/HSF/NF-Y/IRF get their full multimer (up to tetramer), not a dimer.
# Global batch held at 36 (6 x batch6 x accum1) to match the v22 single-GPU runs.
# DDP fixes for this node: UUID pin (skip broken GPU0) + NCCL_P2P_DISABLE (A6000
# no-NVLink hang). If NCCL stalls, fall back to single-GPU (BATCH_SIZE=12 ACCUM=3).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

# 6 good GPUs by UUID (nvidia-smi indices 4-9; excludes broken GPU0)
GPUS="${GPUS:-GPU-26df3b25-f077-10ed-57eb-47e5a71c0cef,GPU-348cc075-8a3d-0689-3039-395677fd3431,GPU-bbd43cba-1653-dfdd-df8b-5f5e0845f298,GPU-c9320d8a-9633-cf7a-f341-ac8096c6fc3a,GPU-2cf50fdc-4677-47ce-37fe-bb29d4932668,GPU-3e792bc3-d868-a8f5-ea8c-806dde5cdd80}"
NPROC="${NPROC:-6}"; MASTER_PORT="${MASTER_PORT:-29601}"
BATCH_SIZE="${BATCH_SIZE:-6}"; ACCUM="${ACCUM:-1}"; SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-225}"
DATA="${DATA:-data/processed/tf_pwm_training_v23.parquet}"
SPLIT="${SPLIT:-data/processed/splits/train_v22/split.json}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v24_contact}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
RUN_NAME="${RUN_NAME:-contact_v24_seed${SEED}}"
OUT="$OUT_ROOT/$RUN_NAME"; LOG_DIR="$OUT_ROOT/logs"; LOG="$LOG_DIR/$RUN_NAME.log"
mkdir -p "$OUT" "$LOG_DIR"

args=(
  scripts/train.py
  --data "$DATA" --split "$SPLIT" --out "$OUT"
  --seed "$SEED" --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE" --grad-accum-steps "$ACCUM"
  --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6
  --warmup-steps 150 --workers 2 --save-every 25
  --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0
  --pwm-head-v18 --group-balanced-sampling
  --v18-contact-supervision --v18-contact-weight 0.3 \
  --v18-contact-bias-scale 1.0 --v18-contact-bias-learnable 1 \
  --contact-distill-weight 0.2 --contact-targets-path data/contact_maps/contact_targets_v23.json
  --recognition-prior-path data/contact_maps/recognition_residues_v23.json
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

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT ${args[*]}"; exit 0
fi
echo "v23 N-chain DDP: $NPROC GPUs | global batch=$((NPROC*BATCH_SIZE*ACCUM)) | data=$DATA"
echo "Log: $LOG"
env CUDA_VISIBLE_DEVICES="$GPUS" TORCH_HOME="$TORCH_HOME" PYTHONUNBUFFERED=1 PYTHONPATH=src \
  CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" "${args[@]}" 2>&1 | tee "$LOG"
