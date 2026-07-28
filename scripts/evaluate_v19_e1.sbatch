#!/bin/bash
#SBATCH -J v19_e1_eval
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00
#SBATCH --array=0-5

set -eo pipefail

ROOT=${TFSCOPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BASE=${TFSCOPE_E1_BASE:-$ROOT/checkpoints/v19_e1}
SEEDS=(42 43 44)
MODES=(norag rag)
MODE=${MODES[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
RUN_DIR=$BASE/${MODE}_seed${SEED}

cd "$ROOT"
if [[ -n "${TFSCOPE_ENV_BIN:-}" ]]; then
    export PATH="$TFSCOPE_ENV_BIN:$PATH"
fi
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export PYTHONUNBUFFERED=1

python scripts/evaluate_v19_e1.py \
    --run-dir "$RUN_DIR" \
    --data data/processed/tf_pwm_aug_dbd_canon_trim.parquet \
    --split data/processed/splits/cluster40_clean/split.json \
    --out "$RUN_DIR/test_metrics.json"
