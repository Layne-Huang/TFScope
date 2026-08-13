#!/usr/bin/env bash
# v20: residue-granularity MoE (8 routed + 2 shared, DeepSeek-style) on the
# REBUILT v2 data (current DB releases: CIS-BP v3.10 / HOCOMOCO v14 / JASPAR
# latest; DNA-contact-cropped structural rows; cluster-aware DBD crops).
#
# Architecture chosen from evidence, not habit:
#   * residue granularity: the ONLY factor that stopped routing collapse -- the
#     protein-granularity MoE routed to EXACTLY uniform 0.25 (dead router),
#     while this variant is diagnosed SPECIALIZED (entropy 1.75/2.08, NMI 0.19,
#     Forkhead->expert6 0.78, C2H2_medium->expert4 0.66).
#   * 8+2 experts, top-2: the config that produced that specialization.
#   * family embedding kept but expected minor (ESM already encodes family;
#     NMI only 0.19) -- learned 10-family, not semantic (semantic scored worse).
#
# New since the last residue-MoE run:
#   * data/split rebuilt on the FINAL post-QC table (train_v2), leakage-free
#     at 40% identity (component split); test = structure rows only, so DeepPBS
#     can be run on the same structures.
#   * --gate-length-weight: couples the gate to the eval protocol (a short gate
#     is scored on fewer, easier columns -> inflated r). Off (0.0) = old behaviour.
#   * --latent-registration: train-frame == eval-frame (shift+RC), removing the
#     strict-train / oracle-eval mismatch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

GPU_UUID="${GPU_UUID:-GPU-826b989f-a711-acc9-a350-9de857b0b2cf}"
BATCH_SIZE="${BATCH_SIZE:-12}"; ACCUM="${ACCUM:-3}"; SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-225}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v21_twochain_heterodimer}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
RUN_NAME="${RUN_NAME:-twochain_v2p_seed${SEED}}"
OUT_DIR="$OUT_ROOT/$RUN_NAME"; LOG_DIR="$OUT_ROOT/logs"; LOG="$LOG_DIR/$RUN_NAME.log"
GATE_LEN_W="${GATE_LEN_W:-0.05}"
LATENT_REG="${LATENT_REG:-1}"     # 1 => pass --latent-registration
mkdir -p "$OUT_DIR" "$LOG_DIR"

command=(
    "$PYTHON_BIN" scripts/train.py
    --data data/processed/tf_pwm_training_v2p.parquet
    --split data/processed/splits/train_v2/split.json
    --out "$OUT_DIR"
    --seed "$SEED" --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE" --grad-accum-steps "$ACCUM"
    --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6
    --warmup-steps 150 --workers 2 --save-every 25
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0
    --early-stop-patience 30
    --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 40
    --pwm-head-v18 --gene-balanced-sampling
    --v18-contact-supervision --v18-contact-weight 0.3 --v18-contact-bias-scale 0.0
    --recognition-prior-path data/contact_maps/recognition_residues_cluster40trainonly.json
    --family-embedding-path none
    --moe-granularity residue --num-experts 8 --n-shared-experts 2 --top-k 2
    --expert-hidden-dim 512
    --balance-loss-weight 0.01 --diversity-loss-weight 0.0 --route-supervision-weight 0.0
    --gate-length-weight "$GATE_LEN_W"
    --two-chain-input
    --precision bf16 --tf32 --no-wandb
)
[[ "$LATENT_REG" == "1" ]] && command+=(--latent-registration)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_UUID"; printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
echo "v20 residue-MoE seed $SEED on $GPU_UUID | gate_len_w=$GATE_LEN_W latent_reg=$LATENT_REG"
echo "Log: $LOG"
CUDA_VISIBLE_DEVICES="$GPU_UUID" TORCH_HOME="$TORCH_HOME" PYTHONUNBUFFERED=1 \
    "${command[@]}" 2>&1 | tee "$LOG"
