#!/bin/bash
#SBATCH -J v19_e1
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-16:00:00
#SBATCH --array=0-5

set -eo pipefail

ROOT=${TFSCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BASE=${TFSCOPE_E1_BASE:-$ROOT/checkpoints/v19_e1}
SEEDS=(42 43 44)
MODES=(norag rag)
MODE=${MODES[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
OUT=$BASE/${MODE}_seed${SEED}

mkdir -p "$BASE/logs" "$OUT"
cd "$ROOT"

if [[ -n "${TFSCOPE_ENV_BIN:-}" ]]; then
    export PATH="$TFSCOPE_ENV_BIN:$PATH"
fi
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export PYTHONUNBUFFERED=1

RAG_ARGS=()
if [[ "$MODE" == "rag" ]]; then
    RAG_ARGS=(
        --use-retrieval
        --retrieval-k 16
        --retrieval-dropout 0.15
        --retrieval-index-path data/processed/tf_nn_index_cluster40_clean.json
    )
fi

python scripts/train.py \
    --data data/processed/tf_pwm_aug_dbd_canon_trim.parquet \
    --split data/processed/splits/cluster40_clean/split.json \
    --out "$OUT" \
    --seed "$SEED" \
    --epochs 200 \
    --batch-size 64 \
    --grad-accum-steps 2 \
    --lr 6e-4 \
    --lora-lr 1e-5 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-n-layers 6 \
    --warmup-steps 500 \
    --workers 4 \
    --save-every 25 \
    --ic-pcc-weight 0.5 \
    --topbase-weight 0.1 \
    --topbase-margin 2.0 \
    --early-stop-patience 30 \
    --eval-oracle-r \
    --oracle-r-every 5 \
    --oracle-r-n-tfs 100 \
    --pwm-head-v18 \
    --wandb-project TFScope \
    --wandb-name "v19_e1_${MODE}_seed${SEED}" \
    "${RAG_ARGS[@]}"
