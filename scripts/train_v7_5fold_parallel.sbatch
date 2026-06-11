#!/bin/bash
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=160G
#SBATCH -t 1-12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v7_5fold/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v7_5fold/slurm_%j.err
#SBATCH -J tfscope_v7_5fold

# Run all 5 v7 folds in PARALLEL on a single H200 (144GB VRAM, ~13GB per fold).
# v7 = same as v8 architecture but WITHOUT retrieval cross-attention.
# Provides a non-RAG ensemble baseline to compare against the v8 RAG ensemble.

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
conda activate tfscope

set -eo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v7_5fold
mkdir -p $ROOT
cd /n/home13/leihuang/project/TFScope

# Launch all 5 folds in background — no --use-retrieval flag
PIDS=()
for FOLD in 0 1 2 3 4; do
    CKPT_DIR=$ROOT/fold${FOLD}
    mkdir -p $CKPT_DIR
    LOG=$CKPT_DIR/fold${FOLD}.log
    echo ">>> Launching v7 fold ${FOLD}, logging to $LOG"

    python scripts/train.py \
        --data          data/processed/tf_pwm_deeppbs_only.parquet \
        --split         data/processed/splits/deeppbs_5fold/fold${FOLD}.json \
        --out           $CKPT_DIR \
        --epochs        200 \
        --batch-size    32 \
        --lr            3e-4 \
        --lora-rank     16 \
        --lora-lr       1e-5 \
        --lora-alpha    32.0 \
        --lora-n-layers 6 \
        --warmup-steps  100 \
        --workers       2 \
        --save-every    10 \
        --early-stop-patience 50 \
        --no-wandb \
        > $LOG 2>&1 &
    PIDS+=($!)
done

echo "All 5 v7 folds launched in background: PIDs=${PIDS[@]}"

FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait $PID; then
        echo "PID $PID failed"
        FAIL=1
    fi
done

if [ $FAIL -ne 0 ]; then
    echo "At least one fold failed — check fold logs under $ROOT/foldN/foldN.log"
    exit 1
fi

echo "All 5 v7 folds complete at $(date)"
nvidia-smi
