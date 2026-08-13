#!/usr/bin/env bash
# MoE-base (per-residue MoE, contact-SUPERVISION only) trained on the WHOLE dataset.
# This is the deployment model: after the contact-bias (parity, prior ignored),
# real-contact injection (inert) and deep-tune (overfits) ablations all came back
# null/negative, MoE-base is what we ship.
#
# Data: combined_fm_deeppbs_all -> train = ALL 4250 rows (val folded in).
#   NOTE: the 200-row "val" is a subset OF train (monitoring only). Its oracle-r /
#   val-loss are OPTIMISTIC and are NOT a held-out estimate. ckpt_best is therefore
#   meaningless here -> select from the milestone checkpoints instead.
# Saves ONLY epochs 150/175/200/225/250. Early stopping effectively disabled.
# 4-GPU torchrun DDP (4 x batch 8 = global 32).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-3,4,5,6}"                 # avoid broken GPU 0
BATCH_SIZE="${BATCH_SIZE:-8}"; SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-250}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_moe_base_alldata}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/torchrun}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
RUN_NAME="${RUN_NAME:-moe_base_alldata_seed${SEED}}"; OUT_DIR="$OUT_ROOT/$RUN_NAME"
LOG_DIR="$OUT_ROOT/logs"; LOG="$LOG_DIR/$RUN_NAME.log"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"; NPROC="${#GPUS[@]}"
mkdir -p "$OUT_DIR" "$LOG_DIR"; cd "$ROOT"
command=(
    "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC"
    scripts/train.py
    --data data/processed/tf_pwm_combined_fm_deeppbs.parquet
    --split data/processed/splits/combined_fm_deeppbs_all/split.json
    --out "$OUT_DIR" --seed "$SEED" --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE" --grad-accum-steps 1
    # baseline LoRA (deeper/wider overfits — see deep-tune negative)
    --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6
    --warmup-steps 150 --workers 2
    --save-epochs 150,175,200,225,250
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0
    --early-stop-patience 10000
    --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 40
    --pwm-head-v18 --gene-balanced-sampling
    --v18-contact-supervision --v18-contact-weight 0.3 --v18-contact-bias-scale 0.0
    --recognition-prior-path data/contact_maps/recognition_residues.json
    --family-embedding-path none
    # per-residue MoE (the "MoE-base" architecture)
    --moe-granularity residue --num-experts 8 --n-shared-experts 2 --top-k 2
    --expert-hidden-dim 512 --balance-loss-weight 0.01 --diversity-loss-weight 0.0
    --route-supervision-weight 0.0
    --precision bf16 --tf32 --no-wandb
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"; printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
echo "MoE-base on ALL 4250 rows, GPUs $GPU_IDS (${NPROC}x${BATCH_SIZE}=$((NPROC*BATCH_SIZE)) global)"; echo "Log: $LOG"
CUDA_VISIBLE_DEVICES="$GPU_IDS" TORCH_HOME="$TORCH_HOME" PYTHONUNBUFFERED=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1 \
"${command[@]}" 2>&1 | tee "$LOG"
