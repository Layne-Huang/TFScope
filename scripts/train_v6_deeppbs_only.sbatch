#!/bin/bash
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v6_only/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v6_only/slurm_%j.err
#SBATCH -J tfscope_v6

# TFScope v6 DeepPBS-only: strictly fair comparison with DeepPBS.
# Dataset: sequences from PDB crystal chains, PWMs from NPZ Y_pwm.
# Training set = DeepPBS CV training entries (train0-4.txt union).
# Test set     = DeepPBS blind benchmark (id.txt, 130 entries).
# No original TFScope dataset used.
# Architecture and loss identical to v4/v5 (L1-only, LoRA rank=16, 6 layers).

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
conda activate tfscope

set -eo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
nvidia-smi

CKPT_DIR=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v6_only
mkdir -p $CKPT_DIR

cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data          data/processed/tf_pwm_deeppbs_only.parquet \
    --split         data/processed/splits/deeppbs_only/benchmark.json \
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
    --save-every    5 \
    --wandb-name    deeppbs_v6_only

echo "Training complete at $(date)"
