#!/bin/bash
#SBATCH -J pretrain_dpac
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/pretrain_dpac/logs/%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/pretrain_dpac/logs/%j.err

# Stage-A contrastive pretraining of the TFScope protein tower on DPAC's
# BioLip2 (protein, bound-DNA) pairs. Output encoder warm-starts Stage-B PWM
# finetuning via:  python scripts/train.py --init-from-pretrain <out>/ckpt_best.pt
#
# SOURCES controls the corpus: "dpac" (default), or "dpac htselex" once the
# full HT-SELEX k-mer landscapes are built.

OUT=${OUT:-/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/pretrain_dpac}
mkdir -p $OUT/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== Stage-A contrastive pretrain | sources=${SOURCES:-dpac} | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/pretrain_contrastive.py \
    --sources ${SOURCES:-dpac} \
    --out $OUT \
    --epochs 40 --batch-size 128 \
    --lr 3e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --contrastive-dim 256 --dna-max-len 50 --workers 4

echo "Done at $(date)"
