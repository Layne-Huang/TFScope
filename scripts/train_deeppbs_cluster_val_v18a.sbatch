#!/bin/bash
#SBATCH -J v18a_dpbs_clval
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_clusterval_v18a/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_clusterval_v18a/logs/train_%j.err

# TFScope v18a — full DeepPBS train set, cluster-based val split.
#
# Split: data/processed/splits/deeppbs_cluster_val/split.json
#   train=409, val=111, test=130  (val clusters disjoint from train)
# Architecture: --pwm-head-v18 (contact-aware cosine cross-attention residual)
# Loss: fixed-weight (no Kendall-Gal), total = gate_weight*L_gate + L_pwm
# 200 epochs, patience=100, batch=32, gpu_test

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_clusterval_v18a
mkdir -p $OUT/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== DeepPBS cluster-val v18a | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_deeppbs_only_canon.parquet \
    --split  data/processed/splits/deeppbs_cluster_val/split.json \
    --out    $OUT \
    --epochs 200 \
    --batch-size 32 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 150 --workers 4 --save-every 25 \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --early-stop-patience 100 \
    --pwm-head-v18 \
    --no-wandb

echo "Done at $(date)"
