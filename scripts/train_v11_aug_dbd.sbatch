#!/bin/bash
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v11_aug_dbd/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v11_aug_dbd/slurm_%j.err
#SBATCH -J tfscope_v11_aug

# TFScope v8 SINGLE — same recipe as v7 single but WITH retrieval cross-attention.
# Isolates the effect of RAG without the 5-fold protocol cost.
# Train recipe (mirrors v7 single):
#   - data: tf_pwm_aug_dbd.parquet (650 entries)
#   - split: benchmark_no_val.json (520 train, 16 placeholder val, 130 test)
#   - 200 epochs, no early stopping
#   - --no-save-best because val is a placeholder (useless signal)
#   - save every 25 epochs
# Retrieval uses the full 520-donor NN index (data/processed/tf_nn_index_aug.json).

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"

set -eo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
nvidia-smi

CKPT_DIR=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v11_aug_dbd
mkdir -p $CKPT_DIR

cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data          data/processed/tf_pwm_aug_dbd.parquet \
    --split         data/processed/splits/deeppbs_aug_dbd/benchmark.json \
    --out           $CKPT_DIR \
    --epochs        200 \
    --batch-size    64 \
    --lr            6e-4 \
    --lora-rank     16 \
    --lora-lr       1e-5 \
    --lora-alpha    32.0 \
    --lora-n-layers 6 \
    --warmup-steps  150 \
    --workers       4 \
    --save-every    25 \
    --no-save-best \
    --use-retrieval \
    --retrieval-k   3 \
    --retrieval-dropout 0.15 \
    --retrieval-index-path data/processed/tf_nn_index_aug.json \
    --no-wandb

echo "v8 single training complete at $(date)"
