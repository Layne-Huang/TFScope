#!/bin/bash
#SBATCH -J v18a_c40_rag
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-16:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_rag/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_rag/logs/train_%j.err

# TFScope v18a + RAG (retrieval-augmented PWM cross-attention) — cluster40 split.
#
# NN index: tf_nn_index_cluster40.json — donors restricted to cluster40 train+val,
# so test TFs never see test-set PWMs. Top-1 sim for test TFs: mean=0.981 (very high,
# so retrieval quality is nearly as good at test time as at train time).
#
# Direct A/B vs c40_best (0.5987 oracle-r) — same split/seed/hyperparams, only RAG added.
# retrieval_dropout=0.15: 15% of samples train without retrieval (CFG-style robustness).

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_rag
mkdir -p $OUT/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== cluster40 v18a + RAG | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
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
    --use-retrieval \
    --retrieval-k 3 \
    --retrieval-dropout 0.15 \
    --retrieval-index-path data/processed/tf_nn_index_cluster40.json \
    --no-wandb

echo "Done at $(date)"
