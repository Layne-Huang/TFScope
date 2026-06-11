#!/bin/bash
#SBATCH -J tfscope_v18a
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-08:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18a_attnrepair/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18a_attnrepair/logs/train_%j.err

# =============================================================
# TFScope v18a — ATTENTION REPAIR (build on v17_200ep prior)
#
# Freeze the v17 prior branch (RAG + de-novo) and train ONLY a new
# contact-aware residual branch:
#   - cosine cross-attention (+ LayerNorm on K/V) -> kills hub/sink collapse
#   - amino-acid-identity values -> mutations move the value vector
#   - residual gate lambda (init 0.1) -> prior preserved early
#   - row-diversity + hub penalties -> break rank-1 collapse
# NO contact supervision yet (that is v18b).
#
# Success: KLF4 attention rank>1, mutated residue mass>0, WT-vs-mut PWM r<1,
#          mean PWM r within ~0.03 of v17.
# =============================================================
OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18a_attnrepair
mkdir -p $OUT/logs
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_deeppbs_only.parquet \
    --split  data/processed/splits/deeppbs_only/benchmark_no_val.json \
    --out    $OUT \
    --epochs 80 --batch-size 16 \
    --lr 6e-4 --lora-rank 0 \
    --warmup-steps 150 --workers 4 --save-every 20 \
    --use-retrieval --retrieval-k 3 --retrieval-dropout 0.0 \
    --retrieval-index-path data/processed/tf_nn_index_lgo_deeppbs.json \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --pwm-head-v18 --v18-freeze-prior \
    --v18-row-div-weight 0.05 --v18-hub-weight 0.05 --v18-delta-scale-init 0.1 \
    --recognition-prior-path data/contact_maps/recognition_residues.json \
    --init-from /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v17_200ep/ckpt_best.pt \
    --no-wandb

echo "v18a complete at $(date)"
