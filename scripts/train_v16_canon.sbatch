#!/bin/bash
#SBATCH -J tfscope_v16
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v16/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v16/logs/train_%j.err

# =============================================================
# TFScope v16: canonical registration + pretrain->finetune
#
# Identical to v15 BUT trained on CANONICALLY-REGISTERED targets
# (left-anchored IC core + canonical strand) for BOTH stages, so the model
# learns ONE consistent motif placement. Targets registration — v14's biggest
# remaining gap (70/130 predictions offset; motif-level r 0.855 vs exact 0.651).
#
# Stage 1: pretrain on 3297 canonical augmented TFs (RAG, aug index, big val)
# Stage 2: finetune on 520 canonical structure-derived TFs (RAG, deeppbs index)
#
# Goal: convert v14's motif-recovery strength (0.855) into the EXACT metric,
# potentially pushing exact mean r above DeepPBS (0.705).
# =============================================================

mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v16/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

PRE=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v16_pretrain
FT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v16
mkdir -p $PRE/logs

echo "=== STAGE 1: pretrain on canonical augmented ==="
python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd_canon.parquet \
    --split  data/processed/splits/deeppbs_only/v15_pretrain.json \
    --out    $PRE \
    --epochs 60 --batch-size 128 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 300 --workers 4 --save-every 20 \
    --use-retrieval --retrieval-k 3 --retrieval-dropout 0.15 \
    --retrieval-index-path data/processed/tf_nn_index_aug.json \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --no-wandb

echo "=== STAGE 2: finetune on canonical structure-derived ==="
python scripts/train.py \
    --data   data/processed/tf_pwm_deeppbs_only_canon.parquet \
    --split  data/processed/splits/deeppbs_only/v15_finetune.json \
    --out    $FT \
    --init-from $PRE/ckpt_best.pt \
    --epochs 100 --batch-size 32 \
    --lr 2e-4 --lora-lr 5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 100 --workers 4 --save-every 25 \
    --use-retrieval --retrieval-k 3 --retrieval-dropout 0.15 \
    --retrieval-index-path data/processed/tf_nn_index.json \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --no-wandb

echo "v16 complete at $(date)"
