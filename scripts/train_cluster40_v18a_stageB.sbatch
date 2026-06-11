#!/bin/bash
#SBATCH -J v18a_c40_stageB
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-16:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_stageB/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_stageB/logs/train_%j.err

# Stage-B: PWM finetuning warm-started from Stage-A contrastive pretrain.
#
# PRETRAIN_CKPT selects which Stage-A encoder to use:
#   dpac       → pretrain_dpac/ckpt_best.pt         (breadth only, 11k pairs)
#   dpac_htselex → pretrain_dpac_htselex/ckpt_best.pt (breadth+depth, 50k pairs)
#
# A/B vs c40_best (0.535): identical split/seed/hyperparams, only encoder init differs.

CKPT_BASE=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints
PRETRAIN=${PRETRAIN:-dpac}
PRETRAIN_CKPT=$CKPT_BASE/pretrain_${PRETRAIN}/ckpt_best.pt
OUT=$CKPT_BASE/cluster40_v18a_stageB_${PRETRAIN}
mkdir -p $OUT/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== Stage-B warm-start | pretrain=${PRETRAIN} | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
echo "    encoder ckpt: $PRETRAIN_CKPT"
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
    --early-stop-patience 30 \
    --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 100 \
    --pwm-head-v18 \
    --init-from-pretrain $PRETRAIN_CKPT \
    --no-wandb

echo "Done at $(date)"
