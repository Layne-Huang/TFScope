#!/bin/bash
#SBATCH -J v18a_c40_contrast
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-16:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_contrast/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_contrast/logs/train_%j.err

# TFScope v18a + DPAC-style in-batch contrastive loss — cluster40 ablation.
#
# Identical to fulldata_cluster40_v18a EXCEPT --contrastive-weight is set.
# Tests whether the anti-collapse InfoNCE term improves OOD oracle-r,
# especially on collapse-prone families (Homeodomain, Forkhead, bHLH).
# Direct A/B vs c40_best (same split, same seed, same everything else).

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_contrast
mkdir -p $OUT/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== cluster40 v18a + contrastive (w=${CW:-0.3} tau=${CTAU:-0.1}) | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd_canon_trim.parquet \
    --split  data/processed/splits/cluster40/split.json \
    --out    $OUT \
    --epochs 200 \
    --batch-size 128 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 500 --workers 4 --save-every 25 \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --contrastive-weight ${CW:-0.3} --contrastive-tau ${CTAU:-0.1} \
    --early-stop-patience 30 \
    --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 100 \
    --pwm-head-v18 \
    --no-wandb

echo "Done at $(date)"
