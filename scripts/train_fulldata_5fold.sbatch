#!/bin/bash
#SBATCH -J tfscope_full5fold
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-08:00:00
#SBATCH --array=0-4
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_5fold/fold_%a/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_5fold/fold_%a/logs/train_%j.err

# =============================================================
# TFScope full-data 5-fold ensemble — NO RAG, canonical registration
#
# Training data : JASPAR + HOCOMOCO + CIS-BP + DeepPBS (3403 entries total)
#                 leakage-filtered: gene-level + 40% DBD identity threshold
# Folds         : our own gene-stratified splits (data/splits/fulldata_5fold/)
#                 ~2700 train / ~680 val per fold
# Targets       : canonical PWMs (tf_pwm_aug_dbd_canon.parquet)
# Retrieval     : NONE
# Test set      : same 130 DeepPBS blind benchmark
#
# Compare with: scripts/train_5fold_norag.sbatch (DeepPBS-exact 520-entry folds)
# Ensemble eval: python scripts/eval_ensemble.py --mode fulldata
# =============================================================

FOLD=$SLURM_ARRAY_TASK_ID
OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_5fold/fold_${FOLD}

mkdir -p $OUT/logs
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== Full-data fold ${FOLD} | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd_canon.parquet \
    --split  data/processed/splits/fulldata_5fold/fold_${FOLD}.json \
    --out    $OUT \
    --epochs 200 \
    --batch-size 128 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 500 --workers 4 --save-every 25 \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --early-stop-patience 30 \
    --no-wandb

echo "Full-data fold ${FOLD} complete at $(date)"
