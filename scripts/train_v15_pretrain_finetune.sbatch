#!/bin/bash
#SBATCH -J tfscope_v15
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v15/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v15/logs/train_%j.err

# =============================================================
# TFScope v15: pretrain on augmented → finetune on structure-derived
#
# Stage 1 (pretrain): v14 architecture + de-novo loss on 3297 augmented TFs
#   (CIS-BP/RCADE/MEME/HOCOMOCO/JASPAR), big 300-TF val. Learns broad family
#   grammar. Test-clean (0 shared source_id/gene with the 130 benchmark).
# Stage 2 (finetune): re-init weights, train on 520 structure-derived TFs,
#   same 300-TF val, lower LR. Snaps back to high-quality target distribution.
#
# Goal: improve the de-novo pathway on the HARD TAIL (where DeepPBS still wins).
# v14 baseline (strand-symmetric): mean r 0.651, beats DeepPBS on median + 54% of TFs.
# =============================================================

mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v15/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

PRE=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v15_pretrain
FT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v15
mkdir -p $PRE/logs

# ── Stage 1: pretrain on augmented (RAG enabled, aug index — matches v14 arch) ──
echo "=== STAGE 1: pretrain on 3297 augmented TFs ==="
python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd.parquet \
    --split  data/processed/splits/deeppbs_only/v15_pretrain.json \
    --out    $PRE \
    --epochs 60 \
    --batch-size 128 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 300 --workers 4 --save-every 20 \
    --use-retrieval --retrieval-k 3 --retrieval-dropout 0.15 \
    --retrieval-index-path data/processed/tf_nn_index_aug.json \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --no-wandb

# ── Stage 2: finetune on 520 structure-derived (re-init weights, lower LR) ──────
echo "=== STAGE 2: finetune on 520 structure-derived TFs ==="
python scripts/train.py \
    --data   data/processed/tf_pwm_deeppbs_only.parquet \
    --split  data/processed/splits/deeppbs_only/v15_finetune.json \
    --out    $FT \
    --init-from $PRE/ckpt_best.pt \
    --epochs 100 \
    --batch-size 32 \
    --lr 2e-4 --lora-lr 5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 100 --workers 4 --save-every 25 \
    --use-retrieval --retrieval-k 3 --retrieval-dropout 0.15 \
    --retrieval-index-path data/processed/tf_nn_index.json \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --no-wandb

echo "v15 complete at $(date)"
