#!/bin/bash
#SBATCH -J lofo_v18a
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-16:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_v18a/%x_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_v18a/%x_%j.err

# Leave-family-out v18a training.
# Usage: sbatch --job-name=lofo_<Family> --export=ALL,FAMILY=<Family> train_lofo_v18a.sbatch
#
# Holds out one entire TF family as test. Train/val from remaining families,
# val is CD-HIT-40% cluster-separated. Same v18a config + oracle-r early stop
# as fulldata_cluster40_v18a.

: "${FAMILY:?Must set FAMILY env var (e.g. C2H2_long)}"

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_v18a/${FAMILY}
SPLIT=data/processed/splits/lofo_v2/${FAMILY}.json
mkdir -p $OUT/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== LFO v18a | held-out=${FAMILY} | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd_canon_trim.parquet \
    --split  $SPLIT \
    --out    $OUT \
    --epochs 200 \
    --batch-size 128 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 500 --workers 4 --save-every 25 \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --early-stop-patience 30 \
    --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 100 \
    --pwm-head-v18 \
    --no-wandb

echo "Done at $(date)"
