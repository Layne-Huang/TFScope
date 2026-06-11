#!/bin/bash
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v8_rag/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v8_rag/slurm_%j.err
#SBATCH -J tfscope_v8

# TFScope v8 RAG-TFScope: same architecture as v7 + retrieval cross-attention.
# Trains on tf_pwm_deeppbs_only.parquet, benchmark_no_val.json (520 train, 130 test).
# Each sample is augmented with top-3 nearest neighbour PWMs from tf_nn_index.json.
# Pure-retrieval baseline: 0.55 mean / 0.78 median Pearson r (target to beat).

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
conda activate tfscope

set -eo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
nvidia-smi

CKPT_DIR=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v8_rag
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
    --use-retrieval \
    --retrieval-k   3 \
    --retrieval-dropout 0.15 \
    --retrieval-index-path data/processed/tf_nn_index.json \
    --wandb-name    deeppbs_v8_rag

echo "Training complete at $(date)"
