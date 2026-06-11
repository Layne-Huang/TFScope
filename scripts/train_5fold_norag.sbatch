#!/bin/bash
#SBATCH -J tfscope_5fold
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-08:00:00
#SBATCH --array=0-4
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_5fold_norag/fold_%a/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_5fold_norag/fold_%a/logs/train_%j.err

# =============================================================
# TFScope 5-fold ensemble — NO RAG, canonical registration
#
# Mimics DeepPBS's 5-fold CV:
#   - Uses DeepPBS's exact fold files (data/splits/deeppbs_5fold/fold{i}.json)
#   - Trains on canonical PWM targets (model outputs in canonical frame)
#   - No retrieval (pure de-novo sequence → PWM)
#   - v14-style architecture: LoRA ESM-2 + MoE + IC-PCC + top-base loss
#
# Submit: sbatch scripts/train_5fold_norag.sbatch
# Five jobs run in parallel (array 0-4).
# Ensemble eval: python scripts/eval_ensemble.py
# =============================================================

FOLD=$SLURM_ARRAY_TASK_ID
OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_5fold_norag/fold_${FOLD}

mkdir -p $OUT/logs
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== Fold ${FOLD} | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_deeppbs_only_canon.parquet \
    --split  data/processed/splits/deeppbs_5fold/fold${FOLD}.json \
    --out    $OUT \
    --epochs 200 \
    --batch-size 128 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 150 --workers 4 --save-every 25 \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --early-stop-patience 30 \
    --no-wandb

echo "Fold ${FOLD} complete at $(date)"
