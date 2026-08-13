#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7,8}"
BATCH_SIZE="${BATCH_SIZE:-11}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_e5b_family_register_bf16_ddp9}"
INIT_MODEL="${INIT_MODEL:-/data1/leihuang/project/TFScope/checkpoints/v19_e5_structural_register_bf16_ddp9/e5_structural_rag_seed42/ckpt_best.pt}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/torchrun}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"
RUN_NAME="e5b_family_register_rag_seed${SEED}"
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
    echo "Missing E5 initialization checkpoint: $INIT_MODEL" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -x "$TORCHRUN_BIN" ]]; then
    echo "Missing Python or torchrun in the TFScope environment" >&2
    exit 2
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"

preflight="$(
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -c '
import torch
count = torch.cuda.device_count()
for index in range(count):
    torch.cuda.get_device_properties(index)
    with torch.cuda.device(index):
        free_bytes, _ = torch.cuda.mem_get_info()
    print(f"DEVICE {index} FREE_GIB {free_bytes / 2**30:.2f}")
print(f"COUNT {count}")
'
)"
visible_count="$(awk '/^COUNT / {print $2}' <<< "$preflight")"
if [[ "$visible_count" != "$NPROC" ]]; then
    echo "GPU_IDS=$GPU_IDS exposes $visible_count CUDA devices, expected $NPROC." >&2
    exit 2
fi
echo "$preflight" | sed '/^COUNT /d'
low_memory="$(
    awk -v minimum="$MIN_FREE_GIB" '
        /^DEVICE / && $4 < minimum {print $2 ":" $4}
    ' <<< "$preflight"
)"
if [[ -n "$low_memory" ]]; then
    echo "At least one selected device has less than ${MIN_FREE_GIB} GiB free:" >&2
    echo "$low_memory" >&2
    exit 2
fi

command=(
    "$TORCHRUN_BIN"
    --standalone
    --nproc_per_node="$NPROC"
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
    --latent-registration
    --registration-max-shift 10
    --registration-min-overlap 4
    --registration-temperature 0.1
    --registration-coverage-penalty 0.5
    --registration-anchor-path
    results/v19_e5b_registration/family_anchors_train.tsv
    --register-head
    --register-loss-weight 1.0
    --register-head-only
    --init-model "$INIT_MODEL"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

echo "Starting E5b register fine-tuning seed $SEED on GPUs $GPU_IDS"
echo "Log: $LOG"
CUDA_VISIBLE_DEVICES="$GPU_IDS" \
TORCH_HOME="$TORCH_HOME" \
PYTHONUNBUFFERED=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_P2P_DISABLE=1 \
"${command[@]}" 2>&1 | tee "$LOG"
