#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7,8}"   # 9 GPUs; GPU 9 is inaccessible on this host
BATCH_SIZE="${BATCH_SIZE:-11}"              # 9 x 11 = 99 global batch ≈ original 96
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-150}"                     # step-matched to E2's 1400 steps (1345 genes / 99 batch = 14 steps/epoch)
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_deploy_v2_norag}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/torchrun}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"
RUN_NAME="deploy_rag_seed${SEED}"
OUT_DIR="$OUT_ROOT/$RUN_NAME"
LOG_DIR="$OUT_ROOT/logs"
LOG="$LOG_DIR/$RUN_NAME.log"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NPROC="${#GPUS[@]}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"

# Verify GPUs before launch
preflight="$(
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -c '
import torch
count = torch.cuda.device_count()
for index in range(count):
    with torch.cuda.device(index):
        free_bytes, _ = torch.cuda.mem_get_info()
    print(f"DEVICE {index} FREE_GIB {free_bytes / 2**30:.2f}")
print(f"COUNT {count}")
'
)"
echo "$preflight"
visible_count="$(awk '/^COUNT / {print $2}' <<< "$preflight")"
if [[ "$visible_count" != "$NPROC" ]]; then
    echo "Expected $NPROC GPUs but got $visible_count. Check GPU_IDS=$GPU_IDS" >&2
    exit 2
fi
low_memory="$(awk -v min="$MIN_FREE_GIB" '/^DEVICE / && $4 < min {print $2":"$4}' <<< "$preflight")"
if [[ -n "$low_memory" ]]; then
    echo "Low memory on: $low_memory" >&2
    exit 2
fi

command=(
    "$TORCHRUN_BIN"
    --standalone
    --nproc_per_node="$NPROC"
    scripts/train.py
    --data   data/processed/tf_pwm_aug_dbd_canon_trim.parquet
    --split  data/processed/splits/cluster40_deploy_v2/split.json
    --out    "$OUT_DIR"
    --seed   "$SEED"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --grad-accum-steps 1
    --lr 4.5e-4
    --lora-lr 7.5e-6
    --lora-rank 16
    --lora-alpha 32
    --lora-n-layers 6
    --warmup-steps 667
    --workers 2
    --save-every 10
    --early-stop-patience 40
    --eval-oracle-r
    --oracle-r-every 5
    --pwm-head-v18
    --family-embedding-path none
    --gene-balanced-sampling
    --precision bf16
    --tf32
    --no-wandb
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"; printf '%q ' "${command[@]}"; printf '\n'
    exit 0
fi

echo "Starting deploy retrain seed $SEED on GPUs $GPU_IDS  (${NPROC} x ${BATCH_SIZE} = $((NPROC * BATCH_SIZE)) global batch)"
echo "Log: $LOG"

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
TORCH_HOME="$TORCH_HOME" \
PYTHONUNBUFFERED=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_P2P_DISABLE=1 \
"${command[@]}" 2>&1 | tee "$LOG"
