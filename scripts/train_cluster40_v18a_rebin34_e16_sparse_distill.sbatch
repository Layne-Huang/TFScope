#!/bin/bash
#SBATCH -J reb34_sparse_distill
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-16:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_rebin34_e16_sparse_distill/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_rebin34_e16_sparse_distill/logs/train_%j.err

# STEPS 2+3 of the contact-sparse-attention plan, layered on the Step-1 rebin/16-expert base:
#   * v18_attn_sparse = entmax_learn  -> HARD-SPARSE residue->base recognition attention (learnable alpha)
#   * contact_distill_weight = 0.3    -> distill REAL co-crystal contacts into the attention
#                                        (data/contact_maps/contact_targets.json, 542 structured train TFs)
# Compare against cluster40_v18a_rebin34_e16 (softmax, no distill) to attribute the
# (sparse attention + structure supervision) effect. For the clean ablation ladder also run an
# entmax-only variant (drop --contact-distill-weight).

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_rebin34_e16_sparse_distill
mkdir -p $OUT/logs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== rebin34/16e + entmax + contact-distill | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd_canon_trim_rebin34.parquet \
    --split  data/processed/splits/cluster40/split.json \
    --out    $OUT \
    --num-families 34 --num-experts 16 --top-k 4 --expert-hidden-dim 1536 \
    --family-embedding-path /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/family_embeddings_rebin34.pt \
    --v18-attn-sparse entmax_learn --v18-attn-alpha-init 1.5 \
    --contact-distill-weight 0.3 \
    --contact-targets-path data/contact_maps/contact_targets.json \
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
