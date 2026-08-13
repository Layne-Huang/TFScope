#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7,8}"
BATCH_SIZE="${BATCH_SIZE:-12}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_e8_aligned_fusion_lora_fixed_ddp9}"
INIT_MODEL="${INIT_MODEL:-/data1/leihuang/project/TFScope/checkpoints/v19_e6_reranker_lora_fixed_ddp8/e6_reranker_seed42/ckpt_best.pt}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/torchrun}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
RUN_NAME="${RUN_NAME:-e8_aligned_fusion_seed${SEED}}"
OUT_DIR="$OUT_ROOT/$RUN_NAME"
LOG_DIR="$OUT_ROOT/logs"
LOG="$LOG_DIR/$RUN_NAME.log"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NPROC="${#GPUS[@]}"
if (( NPROC < 1 )); then
    echo "GPU_IDS must contain at least one GPU ID; got: $GPU_IDS" >&2
    exit 2
fi
if [[ ! -f "$INIT_MODEL" ]]; then
    echo "Missing E6 initialization checkpoint: $INIT_MODEL" >&2
    exit 2
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"

command=(
    "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC"
    scripts/train.py
    --data data/processed/tf_pwm_aug_dbd_canon_trim.parquet
    --split data/processed/splits/cluster40_clean/split.json
    --out "$OUT_DIR"
    --seed "$SEED"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --grad-accum-steps 1
    --lr 1e-3
    --lora-lr 7.5e-6
    --lora-rank 16
    --lora-alpha 32
    --lora-n-layers 6
    --warmup-steps 90
    --workers 2
    --save-every 10
    --ic-pcc-weight 0.5
    --topbase-weight 0.1
    --topbase-margin 2.0
    --early-stop-patience 12
    --eval-oracle-r
    --oracle-r-every 1
    --oracle-r-n-tfs 1000
    --pwm-head-v18
    --family-embedding-path none
    --gene-balanced-sampling
    --precision bf16
    --tf32
    --no-wandb
    --use-retrieval
    --retrieval-k 16
    --retrieval-dropout 0.15
    --retrieval-index-path data/processed/tf_nn_index_cluster40_clean.json
    --aligned-trust-target
    --trust-rank-weight 0.5
    --trust-rank-margin 0.1
    --align-retrieved-pwms
    --retrieval-alignment-max-shift 10
    --retrieval-alignment-min-overlap 4
    --full-retrieval-dropout 0.10
    --neighbor-dropout 0.20
    --hard-negative-rate 0.50
    --hard-negative-per-sample 2
    --all-bad-case-rate 0.10
    --retrieval-reranker-only
    --init-model "$INIT_MODEL"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

echo "Starting E8 aligned retrieval fusion seed $SEED on GPUs $GPU_IDS"
echo "Log: $LOG"
CUDA_VISIBLE_DEVICES="$GPU_IDS" \
TORCH_HOME="$TORCH_HOME" \
PYTHONUNBUFFERED=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_P2P_DISABLE=1 \
"${command[@]}" 2>&1 | tee "$LOG"
