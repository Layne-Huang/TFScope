#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-6,7,8}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_e2_gene_balanced_bf16_ddp3}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/torchrun}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
EPOCHS="${EPOCHS:-200}"
MIN_FREE_GIB="${MIN_FREE_GIB:-30}"
OUT_DIR="$OUT_ROOT/norag_seed42"
LOG_DIR="$OUT_ROOT/logs"
LOG="$LOG_DIR/norag_seed42.log"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} != 3 )); then
    echo "GPU_IDS must contain exactly three GPU IDs; got: $GPU_IDS" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -x "$TORCHRUN_BIN" ]]; then
    echo "Missing Python or torchrun in the TFScope environment" >&2
    exit 2
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"

if ! preflight="$(
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
)"; then
    echo "GPU_IDS=$GPU_IDS contains a CUDA device unusable by this PyTorch build." >&2
    echo "On this machine physical GPU 9 is visible to nvidia-smi but unusable by PyTorch." >&2
    echo "Use the known-good free set GPU_IDS=6,7,8." >&2
    exit 2
fi
visible_count="$(awk '/^COUNT / {print $2}' <<< "$preflight")"
if [[ "$visible_count" != "3" ]]; then
    echo "GPU_IDS=$GPU_IDS exposes $visible_count CUDA devices to PyTorch, expected 3." >&2
    echo "On this machine physical GPU 9 is visible to nvidia-smi but unusable by PyTorch." >&2
    echo "Use the known-good free set GPU_IDS=6,7,8." >&2
    exit 2
fi
echo "$preflight" | sed '/^COUNT /d'
low_memory="$(
    awk -v minimum="$MIN_FREE_GIB" '
        /^DEVICE / && $4 < minimum {print $2 ":" $4}
    ' <<< "$preflight"
)"
if [[ -n "$low_memory" ]]; then
    echo "At least one selected CUDA device has less than ${MIN_FREE_GIB} GiB free:" >&2
    echo "$low_memory" >&2
    echo "Choose three genuinely free devices or wait for an E1 job to finish." >&2
    exit 2
fi

command=(
    "$TORCHRUN_BIN"
    --standalone
    --nproc_per_node=3
    scripts/train.py
    --data data/processed/tf_pwm_aug_dbd_canon_trim.parquet
    --split data/processed/splits/cluster40_clean/split.json
    --out "$OUT_DIR"
    --seed 42
    --epochs "$EPOCHS"
    --batch-size 32
    --grad-accum-steps 1
    --lr 4.5e-4
    --lora-lr 7.5e-6
    --lora-rank 16
    --lora-alpha 32
    --lora-n-layers 6
    --warmup-steps 667
    --workers 2
    --save-every 25
    --ic-pcc-weight 0.5
    --topbase-weight 0.1
    --topbase-margin 2.0
    --early-stop-patience 30
    --eval-oracle-r
    --oracle-r-every 5
    --oracle-r-n-tfs 100
    --pwm-head-v18
    --family-embedding-path none
    --gene-balanced-sampling
    --precision bf16
    --tf32
    --no-wandb
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

echo "Starting three-GPU DDP no-RAG seed 42 on GPUs $GPU_IDS"
echo "Log: $LOG"
CUDA_VISIBLE_DEVICES="$GPU_IDS" \
TORCH_HOME="$TORCH_HOME" \
PYTHONUNBUFFERED=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_P2P_DISABLE=1 \
"${command[@]}" 2>&1 | tee "$LOG"
