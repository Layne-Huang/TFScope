#!/usr/bin/env bash
# Train TFScope on DeepPBS's structure-level split for direct comparison.
#
# Split: data/processed/splits/deeppbs_cluster40/split.json
#   train=471 rows (same 520-row DeepPBS fold union minus val-49)
#   val=49 rows, test=130 rows (= DeepPBS blind id.txt)
#
# Step-matched to V18a baseline: 471/32 × 200 = 2944 steps
#   → 3 GPUs × 12 = 36 global batch → 471/36 ≈ 13 steps/epoch → 225 epochs
#
# No gene-balanced sampling (same as V18a — lets model see all 471 rows/epoch).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2}"
BATCH_SIZE="${BATCH_SIZE:-12}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-225}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_deeppbs_cluster40_semfam34}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/torchrun}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
MIN_FREE_GIB="${MIN_FREE_GIB:-25}"
RUN_NAME="${RUN_NAME:-rag_seed${SEED}}"
OUT_DIR="$OUT_ROOT/$RUN_NAME"
LOG_DIR="$OUT_ROOT/logs"
LOG="$LOG_DIR/$RUN_NAME.log"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NPROC="${#GPUS[@]}"
if (( NPROC < 1 )); then echo "GPU_IDS must contain at least one GPU ID" >&2; exit 2; fi
if [[ ! -x "$PYTHON_BIN" || ! -x "$TORCHRUN_BIN" ]]; then
    echo "Missing Python or torchrun in the TFScope environment" >&2; exit 2
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
    echo "GPU_IDS=$GPU_IDS contains a CUDA device unusable by this PyTorch build." >&2; exit 2
fi
visible_count="$(awk '/^COUNT / {print $2}' <<< "$preflight")"
if [[ "$visible_count" != "$NPROC" ]]; then
    echo "GPU_IDS=$GPU_IDS exposes $visible_count CUDA devices, expected $NPROC." >&2; exit 2
fi
echo "$preflight" | sed '/^COUNT /d'
low_memory="$(awk -v minimum="$MIN_FREE_GIB" '/^DEVICE / && $4 < minimum {print $2":"$4}' <<< "$preflight")"
if [[ -n "$low_memory" ]]; then
    echo "Low memory on: $low_memory (need ${MIN_FREE_GIB} GiB free)" >&2; exit 2
fi

command=(
    "$TORCHRUN_BIN"
    --standalone
    --nproc_per_node="$NPROC"
    scripts/train.py
    --data data/processed/tf_pwm_deeppbs_rebin34.parquet
    --split data/processed/splits/deeppbs_cluster40/split.json
    --out "$OUT_DIR"
    --seed "$SEED"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --grad-accum-steps 1
    --lr 4.5e-4
    --lora-lr 7.5e-6
    --lora-rank 16
    --lora-alpha 32
    --lora-n-layers 6
    --warmup-steps 150
    --workers 2
    --save-every 25
    --ic-pcc-weight 0.5
    --topbase-weight 0.1
    --topbase-margin 2.0
    --early-stop-patience 30
    --eval-oracle-r
    --oracle-r-every 5
    --oracle-r-n-tfs 40
    --pwm-head-v18
    --num-families 34
    --family-embedding-path /afs/csail.mit.edu/u/l/leihuang/project/TFScope/data/processed/family_embeddings_rebin34.pt
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

echo "Starting DeepPBS benchmark seed $SEED on GPUs $GPU_IDS  (${NPROC} × ${BATCH_SIZE} = $((NPROC * BATCH_SIZE)) global batch)"
echo "Log: $LOG"
CUDA_VISIBLE_DEVICES="$GPU_IDS" \
TORCH_HOME="$TORCH_HOME" \
PYTHONUNBUFFERED=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_P2P_DISABLE=1 \
"${command[@]}" 2>&1 | tee "$LOG"
