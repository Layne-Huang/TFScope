#!/bin/bash
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=160G
#SBATCH -t 1-12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v8_5fold/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v8_5fold/slurm_%j.err
#SBATCH -J tfscope_v8_5fold

# Run all 5 v8 folds in PARALLEL on a single H200 (144GB VRAM, ~15GB per fold).
# Each fold runs as an independent Python process; CUDA multi-context handles the sharing.
# Wall-clock ~ slowest fold (parallel) instead of ~5× (sequential).

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"

set -eo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Keep CUDA context creation reasonable when 5 processes share the GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v8_5fold
mkdir -p $ROOT
cd /n/home13/leihuang/project/TFScope

# Launch all 5 folds in background
PIDS=()
for FOLD in 0 1 2 3 4; do
    CKPT_DIR=$ROOT/fold${FOLD}
    mkdir -p $CKPT_DIR
    LOG=$CKPT_DIR/fold${FOLD}.log
    echo ">>> Launching fold ${FOLD}, logging to $LOG"

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
        --use-retrieval \
        --retrieval-k   3 \
        --retrieval-dropout 0.15 \
        --retrieval-index-path data/processed/nn_index_5fold/fold${FOLD}.json \
        --no-wandb \
        > $LOG 2>&1 &
    PIDS+=($!)
done

echo "All 5 folds launched in background: PIDs=${PIDS[@]}"

# Wait for every background process; propagate the first non-zero exit code
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

echo "All 5 folds complete at $(date)"
nvidia-smi
