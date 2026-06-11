#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_semantic/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_semantic/logs/train_%j.err
#SBATCH -J tfscope_train

# =============================================================
# TFScope: PWM prediction with semantic family embeddings
# LOFO split: Homeodomain held out
# SemanticFamilyEmbedding (ProTrek text + ESM-2 seq) + MoE routing
# =============================================================

mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_semantic/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
conda activate tfscope

set -eo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm.parquet \
    --split  data/processed/splits/lofo/Homeodomain.json \
    --out    /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_semantic_long \
    --epochs 100 \
    --batch-size 128 \
    --lr 3e-4 \
    --lora-lr 1e-5 \
    --warmup-steps 500 \
    --workers 4 \
    --save-every 10 \
    --no-wandb

echo "Training complete at $(date)"
