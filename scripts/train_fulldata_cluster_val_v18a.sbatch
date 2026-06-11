#!/bin/bash
#SBATCH -J v18a_full_clval
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_clusterval_v18a/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_clusterval_v18a/logs/train_%j.err

# TFScope v18a — full augmented dataset (DeepPBS + JASPAR/HOCOMOCO/CIS-BP),
# cluster-based val split (train/val clusters disjoint).
#
# Split: data/processed/splits/fulldata_cluster_val/split.json
#   train=3523, val=594, test=130
# Architecture: --pwm-head-v18 (contact-aware cosine cross-attention residual)
# Loss: fixed-weight (no Kendall-Gal)
# 200 epochs, patience=100, batch=128, gpu_h200

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_clusterval_v18a
mkdir -p $OUT/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== FullData cluster-val v18a | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd_canon.parquet \
    --split  data/processed/splits/fulldata_cluster_val/split.json \
    --out    $OUT \
    --epochs 200 \
    --batch-size 128 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 500 --workers 4 --save-every 25 \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --early-stop-patience 100 \
    --pwm-head-v18 \
    --no-wandb

echo "Done at $(date)"
