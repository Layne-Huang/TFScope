#!/bin/bash
#SBATCH -J lofo_v18a_rag
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-16:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_v18a_rag/%x_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_v18a_rag/%x_%j.err

# Leave-family-out v18a + RAG (cross-family retrieval, dropout=0.15).
# Usage: sbatch --job-name=lofo_<Family> --export=ALL,FAMILY=<Family> train_lofo_v18a_rag.sbatch
#
# NN index is per-family: donors = all train+val TFs EXCEPT the held-out family.
# Test TFs always retrieve cross-family neighbors (same-family donors don't exist).
# retrieval_dropout=0.15: teaches model to handle absent/weak retrieval gracefully.
# Direct A/B vs lofo_v18a (no RAG, macro oracle-r=0.479).

: "${FAMILY:?Must set FAMILY env var (e.g. bZIP)}"

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_v18a_rag/${FAMILY}
SPLIT=data/processed/splits/lofo_v2/${FAMILY}.json
NN_INDEX=data/processed/tf_nn_index_lofo_${FAMILY}.json
mkdir -p $OUT/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== LFO v18a + RAG | held-out=${FAMILY} | dropout=0.15 | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
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
    --use-retrieval \
    --retrieval-k 3 \
    --retrieval-dropout 0.15 \
    --retrieval-index-path $NN_INDEX \
    --no-wandb

echo "Done at $(date)"
