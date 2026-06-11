#!/bin/bash
#SBATCH -J v18a_rebin34_e16
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-16:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_rebin34_e16/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_rebin34_e16/logs/train_%j.err

# STEP 1 of the contact-sparse-attention plan: retrain v18a + RAG on the FULL-PFAM REBIN
# taxonomy (34 families) with a FINE-GRAINED 16-expert MoE at MATCHED total capacity
# (16 x 1536 == 12 x 2048, no extra params). Everything else identical to the
# cluster40_v18a_rag baseline, so this run isolates the (taxonomy + expert-granularity) effect.
#
#   - data  : tf_pwm_aug_dbd_canon_trim_rebin34.parquet  (family_id = rebin id 0..33)
#   - split : cluster40/split.json  (SAME TF partition — only family labels changed)
#   - MoE   : num_families 34, num_experts 16, top_k 4, expert_hidden 1536, 1 shared (config default)
#   - family embedding: LEARNED (the precomputed semantic file is fixed at 10 families)
#   - RAG   : identical to baseline (tf_nn_index_cluster40.json, k=3, dropout 0.15)
#
# v18 PWM head is the BASELINE softmax head here (entmax hard-sparsity is Step 2, a separate run).

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_rebin34_e16
mkdir -p $OUT/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== cluster40 v18a + RAG | REBIN34 / 16-expert | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd_canon_trim_rebin34.parquet \
    --split  data/processed/splits/cluster40/split.json \
    --out    $OUT \
    --num-families 34 --num-experts 16 --top-k 4 --expert-hidden-dim 1536 \
    --family-embedding-path /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/family_embeddings_rebin34.pt \
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
