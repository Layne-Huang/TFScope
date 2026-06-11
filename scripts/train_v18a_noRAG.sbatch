#!/bin/bash
#SBATCH -J tfscope_v18noRAG
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-08:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18a_noRAG/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18a_noRAG/logs/train_%j.err

# =============================================================
# TFScope v18a_noRAG — contact-aware repair on a NO-RETRIEVAL prior.
#
# Same v18a attention repair (cosine cross-attn + LN + aa-value + row-div + hub),
# but the frozen prior branch is v14_noRAG (use_retrieval=False) — so there is NO
# retrieval at all, no trust gate, no β. Pure de-novo prior + contact residual.
# Clean ablation: does the contact branch help without any retrieval crutch?
# =============================================================
OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18a_noRAG
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
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --pwm-head-v18 --v18-freeze-prior \
    --v18-row-div-weight 0.05 --v18-hub-weight 0.05 --v18-delta-scale-init 0.1 \
    --init-from /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_noRAG/ckpt_best.pt \
    --no-wandb

echo "v18a_noRAG complete at $(date)"
