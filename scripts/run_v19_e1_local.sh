#!/usr/bin/env bash
set -uo pipefail

# Run the six V19 E1 baselines concurrently on six local GPUs.
#
# Usage:
#   bash scripts/run_v19_e1_local.sh
#   DRY_RUN=1 bash scripts/run_v19_e1_local.sh
#   GPU_IDS=0,1,2,3,4,5 OUT_ROOT=/data/checkpoints/v19_e1 \
#     bash scripts/run_v19_e1_local.sh

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '3,10p' "$0"
    exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT/checkpoints/v19_e1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
WANDB_MODE="${WANDB_MODE:-disabled}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} < 6 )); then
    echo "GPU_IDS must contain at least six GPU IDs; got: $GPU_IDS" >&2
    exit 2
fi

mkdir -p "$OUT_ROOT/logs"
cd "$ROOT"

MODES=(norag norag norag rag rag rag)
SEEDS=(42 43 44 42 43 44)
PIDS=()
NAMES=()

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for task_id in "${!MODES[@]}"; do
    mode="${MODES[$task_id]}"
    seed="${SEEDS[$task_id]}"
    gpu="${GPUS[$task_id]}"
    name="${mode}_seed${seed}"
    out="$OUT_ROOT/$name"
    log="$OUT_ROOT/logs/$name.log"
    rag_args=()

    if [[ "$mode" == "rag" ]]; then
        rag_args=(
            --use-retrieval
            --retrieval-k 16
            --retrieval-dropout 0.15
            --retrieval-index-path data/processed/tf_nn_index_cluster40_clean.json
        )
    fi

    mkdir -p "$out"
    command=(
        python scripts/train.py
        --data data/processed/tf_pwm_aug_dbd_canon_trim.parquet
        --split data/processed/splits/cluster40_clean/split.json
        --out "$out"
        --seed "$seed"
        --epochs 200
        --batch-size 128
        --lr 6e-4
        --lora-lr 1e-5
        --lora-rank 16
        --lora-alpha 32
        --lora-n-layers 6
        --warmup-steps 500
        --workers 4
        --save-every 25
        --ic-pcc-weight 0.5
        --topbase-weight 0.1
        --topbase-margin 2.0
        --early-stop-patience 30
        --eval-oracle-r
        --oracle-r-every 5
        --oracle-r-n-tfs 100
        --pwm-head-v18
        --no-wandb
        "${rag_args[@]}"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'GPU %s: ' "$gpu"
        printf '%q ' "${command[@]}"
        printf '\n'
        continue
    fi

    echo "Starting $name on GPU $gpu; log: $log"
    CUDA_VISIBLE_DEVICES="$gpu" \
    WANDB_MODE="$WANDB_MODE" \
    PYTHONUNBUFFERED=1 \
    "${command[@]}" >"$log" 2>&1 &

    PIDS+=("$!")
    NAMES+=("$name")
done

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

failed=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "Completed ${NAMES[$index]}"
    else
        echo "FAILED ${NAMES[$index]} (see $OUT_ROOT/logs/${NAMES[$index]}.log)" >&2
        failed=1
    fi
done

exit "$failed"
