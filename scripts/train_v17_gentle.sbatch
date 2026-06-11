#!/bin/bash
#SBATCH -J tfscope_v17_gentle
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-16:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v17_gentle/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v17_gentle/logs/train_%j.err

# =============================================================
# TFScope v17: robust-RAG training (trust-gate hardening)
#
# v14 recipe (de-novo IC-PCC + top-base loss, 520 TFs, normal index) PLUS
# training-time robustness augmentations that teach the trust gate to DISTRUST
# bad retrieval and fall back to de-novo:
#   --full-retrieval-dropout 0.30   disable all retrieval for 30% of samples
#   --neighbor-dropout       0.20   drop individual neighbours
#   --hard-negative-rate     0.10   inject same-family / low-PWM-sim negatives
#   --all-bad-case-rate      0.10   replace all neighbours with wrong-family donors
# (augmentations TRAIN-only; off at eval.)
#
# Goal: a model whose beta gate shuts off for bad/novel retrieval, so the
# leakage-free (LGO) number rises toward the leaky number. Compare v14 vs v17
# under BOTH the normal and LGO indices.
# =============================================================

mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v17_gentle/logs
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_deeppbs_only.parquet \
    --split  data/processed/splits/deeppbs_only/benchmark_no_val.json \
    --out    /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v17_gentle \
    --epochs 200 --batch-size 32 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 150 --workers 4 --save-every 50 \
    --use-retrieval --retrieval-k 3 \
    --retrieval-index-path data/processed/tf_nn_index_lgo_deeppbs.json \
    --full-retrieval-dropout 0.15 --neighbor-dropout 0.10 \
    --hard-negative-rate 0.05 --hard-negative-per-sample 1 --all-bad-case-rate 0.05 \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --no-wandb

echo "v17 complete at $(date)"
