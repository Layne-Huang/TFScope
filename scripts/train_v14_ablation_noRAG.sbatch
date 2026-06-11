#!/bin/bash
#SBATCH -J v14_noRAG
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 0-12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_noRAG/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_noRAG/logs/train_%j.err

# =============================================================
# TFScope v14 ABLATION: abandon RAG (pure de-novo + v14 loss)
#
# Identical to v14 EXCEPT no retrieval (--use-retrieval removed). Isolates how
# much RAG contributes vs the de-novo pathway with the IC-PCC + top-base loss.
# Given retrieval was shown to be leakage-driven (v14-LGO drop), this is the
# honest "sequence-only, no motif-copying" baseline.
#
# Same data (520 structure-derived), same loss, same hyperparameters as v14.
# v14 (with RAG, same-source-leaky) = 0.651 strand-sym mean r.
# =============================================================

mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_noRAG/logs
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_deeppbs_only.parquet \
    --split  data/processed/splits/deeppbs_only/benchmark_no_val.json \
    --out    /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_noRAG \
    --epochs 100 \
    --batch-size 32 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 150 --workers 4 --save-every 25 \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --no-wandb

echo "v14_noRAG complete at $(date)"
