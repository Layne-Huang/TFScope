#!/usr/bin/env bash
# Ablation: residue-MoE + integrated frozen-probe contact head → v18 contact BIAS
# (learnable scale). Same recipe/split as run_v19_residue_moe.sh (supervision-only
# baseline = 0.703 gate / 0.680 panel) so the ONLY change is the added predicted-
# contact bias. Contact SUPERVISION stays on (true broad contacts, train-only);
# the BIAS uses the frozen ESM→contact probe's per-residue P(contact) (bias_prior),
# kept separate from the supervision target. Learnable bias scale = the ablation
# readout (drifts to ~0 => bias adds nothing beyond supervision).
#
# Single-GPU pinned by UUID. 250 epochs; save ONLY 150/175/200/225/250 (+ ckpt_best).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_UUID="${GPU_UUID:-GPU-826b989f-a711-acc9-a350-9de857b0b2cf}"   # GPU 3
BATCH_SIZE="${BATCH_SIZE:-12}"; ACCUM="${ACCUM:-3}"; SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-250}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe_contactbias}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
PROBE="/data1/leihuang/TFScope/esm_contact_diagnostic/contact_probe_lr.joblib"
RUN_NAME="${RUN_NAME:-contactbias_seed${SEED}}"; OUT_DIR="$OUT_ROOT/$RUN_NAME"
LOG_DIR="$OUT_ROOT/logs"; LOG="$LOG_DIR/$RUN_NAME.log"
mkdir -p "$OUT_DIR" "$LOG_DIR"; cd "$ROOT"

command=(
    "$PYTHON_BIN" scripts/train.py
    --data data/processed/tf_pwm_combined_fm_deeppbs.parquet
    --split data/processed/splits/combined_fm_deeppbs/split.json
    --out "$OUT_DIR" --seed "$SEED" --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE" --grad-accum-steps "$ACCUM"
    --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6
    --warmup-steps 150 --workers 2
    --save-epochs 150,175,200,225,250
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0
    --early-stop-patience 60
    --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 40
    --pwm-head-v18 --gene-balanced-sampling
    --v18-contact-supervision --v18-contact-weight 0.3 --v18-contact-bias-scale 1.0
    --recognition-prior-path data/contact_maps/recognition_residues_cluster40trainonly.json
    --family-embedding-path none
    # ── per-residue MoE (same as baseline) ──
    --moe-granularity residue --num-experts 8 --n-shared-experts 2 --top-k 2
    --expert-hidden-dim 512 --balance-loss-weight 0.01 --diversity-loss-weight 0.0
    --route-supervision-weight 0.0
    # ── NEW: integrated frozen-probe contact head → learnable contact bias ──
    --contact-pred-head --contact-probe-path "$PROBE" --v18-contact-bias-learnable 1
    --precision bf16 --tf32 --no-wandb
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_UUID"; printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
echo "Starting contact-bias ablation seed $SEED on GPU $GPU_UUID"; echo "Log: $LOG"
CUDA_VISIBLE_DEVICES="$GPU_UUID" TORCH_HOME="$TORCH_HOME" PYTHONUNBUFFERED=1 \
"${command[@]}" 2>&1 | tee "$LOG"
