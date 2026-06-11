#!/bin/bash
#SBATCH -J eval_c40_oracle
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-01:00:00
#SBATCH -o /n/home13/leihuang/project/TFScope/logs/eval_c40_oracle_%j.out
#SBATCH -e /n/home13/leihuang/project/TFScope/logs/eval_c40_oracle_%j.err

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail
cd /n/home13/leihuang/project/TFScope

CKPT_BASE=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints
SPLIT=data/processed/splits/cluster40/split.json
LOG=logs/oracle_r_comparison.log

{
echo
echo "=== cluster40 RAG (20155996) ==="
python scripts/eval_oracle_r_testset.py $CKPT_BASE/cluster40_v18a_rag $SPLIT 2>&1 | grep -E "oracle-r|use_retrieval|CKPT"
echo
echo "=== Stage-B dpac (20088713) ==="
python scripts/eval_oracle_r_testset.py $CKPT_BASE/cluster40_v18a_stageB_dpac $SPLIT 2>&1 | grep -E "oracle-r|use_retrieval|CKPT"
} | tee -a $LOG

echo "Done at $(date)"
