#!/usr/bin/env bash
# Per-residue fine-grained MoE (DeepSeekMoE-style) on the combined DeepPBS split.
#
# Rationale: the pooled MOEBlock makes ONE routing decision per protein (~881
# total) and every past run either collapsed to uniform routing or, when forced
# to specialize by CE supervision, LOST ~0.05 accuracy. This variant moves the
# MoE into a per-DBD-residue FFN (~50-70 decisions/protein) so specialization
# can *emerge* from the token-level task. DeepSeek recipe: 2 shared experts
# (absorb universal base-readout chemistry) + 8 fine-grained routed SwiGLU
# experts, top-2, token-level load balance only, family-diversity loss OFF,
# NO CE routing supervision (emergent). Refined residue reps feed both pooling
# and the cross-attention PWM-head keys, so the MoE is a real bottleneck.
#
# Single-GPU pinned by UUID (multi-GPU DDP stalls on this node; GPU 0 is broken
# for CUDA). Step-matched to combined: batch 12 x accum 3 = 36 global.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# GPU 3 (UUID 826b989f) — pin by UUID so the CVD index mapping can't misroute.
GPU_UUID="${GPU_UUID:-GPU-826b989f-a711-acc9-a350-9de857b0b2cf}"
BATCH_SIZE="${BATCH_SIZE:-12}"
ACCUM="${ACCUM:-3}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-225}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
RUN_NAME="${RUN_NAME:-residue_moe_seed${SEED}}"
OUT_DIR="$OUT_ROOT/$RUN_NAME"
LOG_DIR="$OUT_ROOT/logs"
LOG="$LOG_DIR/$RUN_NAME.log"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"

command=(
    "$PYTHON_BIN" scripts/train.py
    --data data/processed/tf_pwm_combined_fm_deeppbs.parquet
    --split data/processed/splits/combined_fm_deeppbs/split.json
    --out "$OUT_DIR"
    --seed "$SEED"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --grad-accum-steps "$ACCUM"
    --lr 4.5e-4
    --lora-lr 7.5e-6
    --lora-rank 16
    --lora-alpha 32
    --lora-n-layers 6
    --warmup-steps 150
    --workers 2
    --save-every 25
    --ic-pcc-weight 0.5
    --topbase-weight 0.1
    --topbase-margin 2.0
    --early-stop-patience 30
    --eval-oracle-r
    --oracle-r-every 5
    --oracle-r-n-tfs 40
    --pwm-head-v18
    --gene-balanced-sampling
    --v18-contact-supervision
    --v18-contact-weight 0.3
    --v18-contact-bias-scale 0.0
    --recognition-prior-path data/contact_maps/recognition_residues_cluster40trainonly.json
    --family-embedding-path none
    # ── per-residue fine-grained MoE ──
    --moe-granularity residue
    --num-experts 8
    --n-shared-experts 2
    --top-k 2
    --expert-hidden-dim 512
    --balance-loss-weight 0.01
    --diversity-loss-weight 0.0
    --route-supervision-weight 0.0
    --precision bf16
    --tf32
    --no-wandb
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_UUID"
    printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi

echo "Starting residue-MoE seed $SEED on GPU $GPU_UUID (batch ${BATCH_SIZE} x accum ${ACCUM} = $((BATCH_SIZE*ACCUM)) eff)"
echo "Log: $LOG"
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
TORCH_HOME="$TORCH_HOME" \
PYTHONUNBUFFERED=1 \
"${command[@]}" 2>&1 | tee "$LOG"
