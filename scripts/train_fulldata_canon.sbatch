#!/bin/bash
#SBATCH -J tfscope_fulldata
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-08:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_fulldata_canon/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_fulldata_canon/logs/train_%j.err

# =============================================================
# TFScope full-data canonical — NO RAG
#
# Training data : all JASPAR + HOCOMOCO + CIS-BP + DeepPBS entries
#                 (4247 total → 3783 train after excluding 130 test genes)
# Targets       : canonical PWMs (tf_pwm_aug_dbd_canon.parquet)
#                 same deterministic trim+strand rule used for evaluation
# Retrieval     : NONE (pure sequence → PWM)
# Architecture  : v14-style (LoRA ESM-2, MoE, IC-PCC + top-base loss)
# Val set       : 182 gene-stratified entries for early stopping
# Test set      : 130 DeepPBS blind benchmark (canonical-fixed r)
#
# After training: python scripts/eval_canonical_registration.py --models fulldata deeppbs
# =============================================================

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_fulldata_canon
mkdir -p $OUT/logs
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail

echo "=== fulldata_canon | Job $SLURM_JOB_ID on $SLURM_NODENAME ==="
nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_aug_dbd_canon.parquet \
    --split  data/processed/splits/fulldata_canon/train_val.json \
    --out    $OUT \
    --epochs 200 \
    --batch-size 32 \
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
    --warmup-steps 500 --workers 4 --save-every 25 \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --early-stop-patience 30 \
    --no-wandb

echo "fulldata_canon complete at $(date)"
