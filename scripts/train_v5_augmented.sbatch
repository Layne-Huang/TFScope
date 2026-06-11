#!/bin/bash
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v5_augmented/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v5_augmented/slurm_%j.err
#SBATCH -J tfscope_v5

# TFScope v5 augmented: v4 architecture + 46 new DeepPBS CV training TFs added to parquet
# Same loss (L1-only), same arch (v3), same hyperparams as v4.
# Augmented dataset: data/processed/tf_pwm_augmented.parquet (+46 rows)
# New split: data/processed/splits/deeppbs_blind/benchmark_augmented.json
# train=3299, val=361, test=129

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
conda activate tfscope

set -eo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
nvidia-smi

CKPT_DIR=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v5_augmented
mkdir -p $CKPT_DIR

cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data          data/processed/tf_pwm_augmented.parquet \
    --split         data/processed/splits/deeppbs_blind/benchmark_augmented.json \
    --out           $CKPT_DIR \
    --epochs        200 \
    --batch-size    64 \
    --lr            3e-4 \
    --lora-rank     16 \
    --lora-lr       1e-5 \
    --lora-alpha    32.0 \
    --lora-n-layers 6 \
    --warmup-steps  500 \
    --workers       4 \
    --save-every    5 \
    --wandb-name    deeppbs_v5_augmented

echo "Training complete at $(date)"
