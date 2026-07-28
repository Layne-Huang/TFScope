#!/usr/bin/env bash
# Multi-GPU (torchrun DDP) version of the deep-tune residue-MoE run.
# 4 GPUs x per-gpu batch 8 x accum 1 = global batch 32 (matches single-GPU 8x4).
# Deep+wide LoRA (12 layers / rank32 / alpha64). Same combined split & recipe as
# the supervision-only residue-MoE baseline (0.703/0.680). DDP env from the
# working combined recipe (NCCL_P2P_DISABLE=1). find_unused_parameters=True in
# train.py handles MoE experts that get no tokens on a rank.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-3,4,5,6}"                       # avoid broken GPU 0
BATCH_SIZE="${BATCH_SIZE:-8}"; SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-250}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe_deeptune}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/torchrun}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
RUN_NAME="${RUN_NAME:-deeptune_ddp_seed${SEED}}"; OUT_DIR="$OUT_ROOT/$RUN_NAME"
LOG_DIR="$OUT_ROOT/logs"; LOG="$LOG_DIR/$RUN_NAME.log"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"; NPROC="${#GPUS[@]}"
mkdir -p "$OUT_DIR" "$LOG_DIR"; cd "$ROOT"
command=(
    "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC"
    scripts/train.py
    --data data/processed/tf_pwm_combined_fm_deeppbs.parquet
    --split data/processed/splits/combined_fm_deeppbs/split.json
    --out "$OUT_DIR" --seed "$SEED" --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE" --grad-accum-steps 1
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
if [[ "${DRY_RUN:-0}" == "1" ]]; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"; printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
echo "Starting DDP deep-tune on GPUs $GPU_IDS (${NPROC}x${BATCH_SIZE}=$((NPROC*BATCH_SIZE)) global)"; echo "Log: $LOG"
CUDA_VISIBLE_DEVICES="$GPU_IDS" TORCH_HOME="$TORCH_HOME" PYTHONUNBUFFERED=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1 \
"${command[@]}" 2>&1 | tee "$LOG"
