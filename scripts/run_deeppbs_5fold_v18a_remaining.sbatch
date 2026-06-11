#!/bin/bash
#SBATCH -J v18a_5fold_seq
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_5fold_v18a/logs/seq_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_5fold_v18a/logs/seq_%j.err

# Run DeepPBS-5fold v18a folds 1-4 sequentially on a single gpu_test node.
# v18a head + canonical registration + fixed-weight loss (no Kendall-Gal).
# Fold 0 already complete; this script handles the remaining four.

CKPT_ROOT="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_5fold_v18a"
DATA="data/processed/tf_pwm_deeppbs_only_canon.parquet"
PROJECT="/n/home13/leihuang/project/TFScope"

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

nvidia-smi || true
cd $PROJECT

for FOLD in 1 2 3 4; do
    CKPT="$CKPT_ROOT/fold_${FOLD}/ckpt_best.pt"
    if [ -f "$CKPT" ]; then
        echo "=== fold $FOLD already done, skipping ==="
        continue
    fi

    mkdir -p "$CKPT_ROOT/fold_${FOLD}/logs"
    echo "=== Starting fold $FOLD at $(date) | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="

    python scripts/train.py \
        --data   $DATA \
        --split  data/processed/splits/deeppbs_5fold/fold${FOLD}.json \
        --out    $CKPT_ROOT/fold_${FOLD} \
        --epochs 250 --batch-size 32 \
        --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
        --warmup-steps 150 --workers 4 --save-every 25 \
        --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
        --early-stop-patience 100 \
        --pwm-head-v18 \
        --no-wandb

    echo "=== Fold $FOLD complete at $(date) ==="
done

echo "=== All folds done at $(date) ==="
