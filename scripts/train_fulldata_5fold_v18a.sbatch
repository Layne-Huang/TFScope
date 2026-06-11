#!/bin/bash
#SBATCH -J tfscope_full5fold_v18a
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-12:00:00
#SBATCH --array=0-4
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_5fold_v18a/fold_%a/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_5fold_v18a/fold_%a/logs/train_%j.err

# =============================================================
# TFScope full-data 5-fold — v18a head, canonical registration, fixed-weight loss
#
# Key changes vs previous fulldata_5fold:
#   --pwm-head-v18       : contact-aware cosine cross-attention residual branch
#   fixed-weight loss    : removed Kendall-Gal learnable sigma (val loss now stable)
#   --early-stop-patience 100 : allows L_pwm to converge (~epoch 87 from v14_noRAG)
#   --epochs 300         : ceiling above patience to avoid truncation
#   --batch-size 128     : gpu_h200
# =============================================================

FOLD=$SLURM_ARRAY_TASK_ID
OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_5fold_v18a/fold_${FOLD}

mkdir -p $OUT/logs
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== Full-data v18a fold ${FOLD} | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd_canon.parquet \
    --split  data/processed/splits/fulldata_5fold/fold_${FOLD}.json \
    --out    $OUT \
    --epochs 300 \
    --batch-size 128 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 500 --workers 4 --save-every 25 \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --early-stop-patience 100 \
    --pwm-head-v18 \
    --no-wandb

echo "Full-data v18a fold ${FOLD} complete at $(date)"
