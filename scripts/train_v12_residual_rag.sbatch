#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v12_residual_rag/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v12_residual_rag/logs/train_%j.err
#SBATCH -J tfscope_v12

# =============================================================
# TFScope v12: Residual-prior RAG
#
# Architecture change from v11:
#   v11: output = de_novo_logits + beta * log(retrieved_prior)
#   v12: output = log(retrieved_prior) + alpha * delta(protein, prior)
#
# Key differences:
#   - Starts from retrieval prior (not de-novo); alpha initialised to 0.13
#   - Prior cross-attention: model sees what retrieval suggests per position
#   - Leave-source-out NN index: excludes same source_id from neighbours
#   - 50% retrieval dropout (CFG) so de-novo pathway remains functional
#
# NN index: tf_nn_index_lso.json (leave-source-out, K=3)
# Training data: tf_pwm_aug_dbd.parquet (4247 TFs)
# Split: benchmark_no_val.json (520 train / 16 val / 130 test)
# =============================================================

mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v12_residual_rag/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"

set -eo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd.parquet \
    --split  data/processed/splits/deeppbs_only/benchmark_no_val.json \
    --out    /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v12_residual_rag \
    --epochs 150 \
    --batch-size 64 \
    --lr 6e-4 \
    --lora-lr 1e-5 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-n-layers 6 \
    --warmup-steps 150 \
    --workers 4 \
    --save-every 25 \
    --use-retrieval \
    --residual-prior \
    --retrieval-k 3 \
    --retrieval-dropout 0.5 \
    --retrieval-index-path data/processed/tf_nn_index_lso.json

echo "Training complete at $(date)"
