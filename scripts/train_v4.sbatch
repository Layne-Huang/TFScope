#!/bin/bash
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark_v4/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark_v4/slurm_%j.err
#SBATCH -J tfscope_v4

# TFScope v4: v3 architecture + plain L1 primary loss (drop KL, match DeepPBS formula)

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
conda activate tfscope

set -eo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data          data/processed/tf_pwm.parquet \
    --split         data/processed/splits/deeppbs/benchmark.json \
    --out           /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark_v4 \
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
    --wandb-name    deeppbs_benchmark_v4

echo "Training complete at $(date)"
