#!/usr/bin/env bash
set -uo pipefail

# Run the paired seed-42 V19 E2 gene-balanced baselines on two local GPUs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_e2_gene_balanced_bf16}"
GPU_IDS="${GPU_IDS:-7,8}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
WANDB_MODE="${WANDB_MODE:-disabled}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} != 2 )); then
    echo "GPU_IDS must contain exactly two GPU IDs; got: $GPU_IDS" >&2
    exit 2
fi

mkdir -p "$OUT_ROOT/logs"
cd "$ROOT"

MODES=(norag rag)
PIDS=()

for task_id in "${!MODES[@]}"; do
    mode="${MODES[$task_id]}"
    gpu="${GPUS[$task_id]}"
    name="${mode}_seed42"
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
        "$PYTHON_BIN" scripts/train.py
        --data data/processed/tf_pwm_aug_dbd_canon_trim.parquet
        --split data/processed/splits/cluster40_clean/split.json
        --out "$out"
        --seed 42
        --epochs 200
        --batch-size 64
        --grad-accum-steps 2
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
        --family-embedding-path none
        --gene-balanced-sampling
        --precision bf16
        --tf32
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
    nohup env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        WANDB_MODE="$WANDB_MODE" \
        TORCH_HOME="$TORCH_HOME" \
        PYTHONUNBUFFERED=1 \
        "${command[@]}" >"$log" 2>&1 </dev/null &
    PIDS+=("$!")
done

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

printf '%s\n' "${PIDS[@]}" >"$OUT_ROOT/launcher.pids"
echo "Started PIDs: ${PIDS[*]}"

failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
exit "$failed"
