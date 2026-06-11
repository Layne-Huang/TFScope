#!/bin/bash
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v8_fold%a/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v8_fold%a/slurm_%j.err
#SBATCH -J tfscope_v8f
#SBATCH --array=0-4

# TFScope v8 RAG-TFScope, 5-fold cross-validation (mirrors DeepPBS's setup).
# Each fold trains on ~416 structures (4 of 5 DeepPBS folds), with the held-out
# fold (~104) used as a real validation set. Early stopping (patience=15, max=100).
# At inference, the 5 models are ensembled by averaging softmax outputs.

FOLD=$SLURM_ARRAY_TASK_ID
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
conda activate tfscope

set -eo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Fold:     $FOLD"
echo "Node:     $SLURM_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
nvidia-smi

CKPT_DIR=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v8_fold${FOLD}
mkdir -p $CKPT_DIR

cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data          data/processed/tf_pwm_deeppbs_only.parquet \
    --split         data/processed/splits/deeppbs_5fold/fold${FOLD}.json \
    --out           $CKPT_DIR \
    --epochs        100 \
    --batch-size    32 \
    --lr            3e-4 \
    --lora-rank     16 \
    --lora-lr       1e-5 \
    --lora-alpha    32.0 \
    --lora-n-layers 6 \
    --warmup-steps  100 \
    --workers       4 \
    --save-every    10 \
    --early-stop-patience 15 \
    --use-retrieval \
    --retrieval-k   3 \
    --retrieval-dropout 0.15 \
    --retrieval-index-path data/processed/nn_index_5fold/fold${FOLD}.json \
    --wandb-name    deeppbs_v8_fold${FOLD}

echo "Training fold $FOLD complete at $(date)"
