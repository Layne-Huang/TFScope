#!/bin/bash
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v7_full/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v7_full/slurm_%j.err
#SBATCH -J tfscope_v7

# TFScope v7: same as v6 but with ALL 520 CV entries in training (no held-out val).
# Validation step still runs on a 16-sample subset of train (placeholder, ignore val_loss).
# Saves checkpoint every 25 epochs.

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
conda activate tfscope

set -eo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
nvidia-smi

CKPT_DIR=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v7_full
mkdir -p $CKPT_DIR

cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data          data/processed/tf_pwm_deeppbs_only.parquet \
    --split         data/processed/splits/deeppbs_only/benchmark_no_val.json \
    --out           $CKPT_DIR \
    --epochs        200 \
    --batch-size    32 \
    --lr            3e-4 \
    --lora-rank     16 \
    --lora-lr       1e-5 \
    --lora-alpha    32.0 \
    --lora-n-layers 6 \
    --warmup-steps  200 \
    --workers       4 \
    --save-every    25 \
    --wandb-name    deeppbs_v7_full

echo "Training complete at $(date)"
