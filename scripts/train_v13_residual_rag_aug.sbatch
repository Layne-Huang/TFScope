#!/bin/bash
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v13_residual_rag_aug/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v13_residual_rag_aug/logs/train_%j.err
#SBATCH -J tfscope_v13

# =============================================================
# TFScope v13: Residual-prior RAG + augmented training set
#
# Changes from v12:
#   - Training set: 4117 samples (520 struct + 3597 ORIG__ CIS-BP/RCADE/MEME)
#   - Split: benchmark_no_val_aug.json
#   - Val/test sets unchanged (same 16 val, 130 test)
#
# No information leakage: 0 ORIG__ entries share source_id or gene_symbol
# with any test TF (verified 2026-05-29).
# =============================================================

mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v13_residual_rag_aug/logs

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
    --split  data/processed/splits/deeppbs_only/benchmark_no_val_aug.json \
    --out    /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v13_residual_rag_aug \
    --epochs 150 \
    --batch-size 64 \
    --lr 6e-4 \
    --lora-lr 1e-5 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-n-layers 6 \
    --warmup-steps 500 \
    --workers 4 \
    --save-every 25 \
    --use-retrieval \
    --residual-prior \
    --retrieval-k 3 \
    --retrieval-dropout 0.5 \
    --retrieval-index-path data/processed/tf_nn_index_lso.json

echo "Training complete at $(date)"
