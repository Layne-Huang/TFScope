#!/bin/bash
#SBATCH -J tfscope_v14
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 0-12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_icpcc/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_icpcc/logs/train_%j.err

# =============================================================
# TFScope v14: v10 recipe + de-novo base-composition loss
#
# Identical to v10 (additive RAG, 15% dropout, tf_nn_index.json, 520 TFs)
# but adds two loss terms targeting the base-composition gap:
#   --ic-pcc-weight 0.5   IC-weighted per-column (1-Pearson) loss
#   --topbase-weight 0.1  top-base margin loss on high-IC positions
#
# Rationale: v10 IC-r=0.92 (knows WHICH positions matter) but column-r=0.54.
# Retrieval is exhausted under clean LSO (deployable ceiling ~0.55); the lever
# is the de-novo prediction. This directly optimises the reported metric.
# Baseline to beat: v10 mean per-column r = 0.542.
# =============================================================

mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_icpcc/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1

set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_deeppbs_only.parquet \
    --split  data/processed/splits/deeppbs_only/benchmark_no_val.json \
    --out    /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_icpcc \
    --epochs 100 \
    --batch-size 16 \
    --lr 6e-4 \
    --lora-lr 1e-5 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-n-layers 6 \
    --warmup-steps 150 \
    --workers 4 \
    --save-every 25 \
    --use-retrieval \
    --retrieval-k 3 \
    --retrieval-dropout 0.15 \
    --retrieval-index-path data/processed/tf_nn_index.json \
    --ic-pcc-weight 0.5 \
    --topbase-weight 0.1 \
    --topbase-margin 2.0 \
    --no-wandb

echo "v14 training complete at $(date)"
