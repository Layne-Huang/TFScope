#!/bin/bash
#SBATCH -J v18a_clval_trim
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-16:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_clval_v18a_trim/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_clval_v18a_trim/logs/train_%j.err

# TFScope v18a — full augmented dataset (IC>=0.5 flank-trimmed PWMs),
# cluster-based val split, oracle-r early stopping.
#
# Key changes vs fulldata_clusterval_v18a:
#   - Data: tf_pwm_aug_dbd_canon_trim.parquet (IC>=0.5 retrimmed)
#   - Early stopping: oracle-aligned Pearson r on 100 val TFs every 5 epochs
#     instead of total val loss (fixes L_gate dominating early stopping)
# Split: data/processed/splits/fulldata_cluster_val/split.json
#   train=3523, val=594, test=130
# Architecture: --pwm-head-v18
# 200 epochs, patience=30 oracle-r epochs (= up to 150 real epochs w/o improvement)

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_clval_v18a_trim
mkdir -p $OUT/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== FullData clval v18a trim | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd_canon_trim.parquet \
    --split  data/processed/splits/fulldata_cluster_val/split.json \
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
