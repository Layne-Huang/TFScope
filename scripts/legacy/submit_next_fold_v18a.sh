#!/bin/bash
# Submit the next pending DeepPBS-5fold v18a job on gpu_test.
# v18a architecture + canonical registration + fixed-weight loss (no Kendall-Gal).
# gpu_test allows only 2 jobs total; run this after each fold finishes.
# Usage: bash scripts/submit_next_fold_v18a.sh

CKPT_ROOT="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_5fold_v18a"
DATA="data/processed/tf_pwm_deeppbs_only_canon.parquet"
PROJECT="/n/home13/leihuang/project/TFScope"

for FOLD in 0 1 2 3 4; do
    CKPT="$CKPT_ROOT/fold_${FOLD}/ckpt_best.pt"
    LOG_OUT="$CKPT_ROOT/fold_${FOLD}/logs"

    if [ -f "$CKPT" ]; then
        echo "fold $FOLD: done"
        continue
    fi

    RUNNING=$(squeue -u $USER -h 2>/dev/null | grep "v18a_5f_${FOLD}" | wc -l)
    if [ "$RUNNING" -gt 0 ]; then
        echo "fold $FOLD: already running/queued"
        continue
    fi

    echo "Submitting v18a fold $FOLD on gpu_test ..."
    mkdir -p $LOG_OUT
    sbatch \
        --job-name="v18a_5f_${FOLD}" \
        --partition=gpu_test --gres=gpu:1 -c 8 --mem=48G --time=12:00:00 \
        -o "$LOG_OUT/train_%j.out" \
        -e "$LOG_OUT/train_%j.err" \
        --wrap="
export PATH='/n/home13/leihuang/.conda/envs/tfscope/bin:\$PATH'
export TORCH_HOME='/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch'
export PYTHONUNBUFFERED=1
set -eo pipefail
echo '=== DeepPBS v18a fold ${FOLD} | Job \$SLURM_JOB_ID on \$SLURM_NODENAME ==='
nvidia-smi || true
cd $PROJECT
python scripts/train.py \\
    --data   $DATA \\
    --split  data/processed/splits/deeppbs_5fold/fold${FOLD}.json \\
    --out    $CKPT_ROOT/fold_${FOLD} \\
    --epochs 250 --batch-size 32 \\
    --lr 6e-4 --lora-lr 1e-5 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \\
    --warmup-steps 150 --workers 4 --save-every 25 \\
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \\
    --early-stop-patience 100 \\
    --pwm-head-v18 \\
    --no-wandb
echo 'v18a fold ${FOLD} done at \$(date)'"
    echo "fold $FOLD: submitted"
    exit 0
done
echo "All DeepPBS-5fold v18a folds done or queued."
