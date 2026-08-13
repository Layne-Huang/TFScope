#!/usr/bin/env bash
# Recommendation #1: tune ESM harder so the frozen encoder can adapt to OOD / de
# novo sequences instead of collapsing them onto the nearest natural homolog.
# Identical to run_v19_residue_moe.sh EXCEPT deeper+wider LoRA:
#   lora-n-layers 6->12, rank 16->32, alpha 32->64.
# Same combined split & recipe so it's comparable to supervision-only residue-MoE
# (0.703 gate / 0.680 panel). Batch 8 x accum 4 = 32 (extra LoRA activation mem).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_UUID="${GPU_UUID:-GPU-826b989f-a711-acc9-a350-9de857b0b2cf}"   # GPU 3
BATCH_SIZE="${BATCH_SIZE:-8}"; ACCUM="${ACCUM:-4}"; SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-250}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe_deeptune}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
RUN_NAME="${RUN_NAME:-deeptune_seed${SEED}}"; OUT_DIR="$OUT_ROOT/$RUN_NAME"
LOG_DIR="$OUT_ROOT/logs"; LOG="$LOG_DIR/$RUN_NAME.log"
mkdir -p "$OUT_DIR" "$LOG_DIR"; cd "$ROOT"
command=(
    "$PYTHON_BIN" scripts/train.py
    --data data/processed/tf_pwm_combined_fm_deeppbs.parquet
    --split data/processed/splits/combined_fm_deeppbs/split.json
    --out "$OUT_DIR" --seed "$SEED" --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE" --grad-accum-steps "$ACCUM"
    --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 32 --lora-alpha 64 --lora-n-layers 12
    --warmup-steps 150 --workers 2
    --save-epochs 150,175,200,225,250
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0
    --early-stop-patience 60
    --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 40
    --pwm-head-v18 --gene-balanced-sampling
    --v18-contact-supervision --v18-contact-weight 0.3 --v18-contact-bias-scale 0.0
    --recognition-prior-path data/contact_maps/recognition_residues_cluster40trainonly.json
    --family-embedding-path none
    --moe-granularity residue --num-experts 8 --n-shared-experts 2 --top-k 2
    --expert-hidden-dim 512 --balance-loss-weight 0.01 --diversity-loss-weight 0.0
    --route-supervision-weight 0.0
    --precision bf16 --tf32 --no-wandb
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_UUID"; printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
echo "Starting deep-tune residue-MoE (LoRA 12L/r32/a64) seed $SEED on GPU $GPU_UUID"; echo "Log: $LOG"
CUDA_VISIBLE_DEVICES="$GPU_UUID" TORCH_HOME="$TORCH_HOME" PYTHONUNBUFFERED=1 \
"${command[@]}" 2>&1 | tee "$LOG"
