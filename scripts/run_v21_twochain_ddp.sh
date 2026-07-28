#!/usr/bin/env bash
# v21 two-chain (heterodimer) MoE, DDP on 6 GPUs.
# Global effective batch held at 36 to match single-chain v20 (6 GPU x batch 6
# x accum 1 = 36), so v21-vs-v20 stays a clean two-chain ablation, not a
# batch-size confound. Uses torchrun (env:// NCCL); find_unused_parameters=True
# in train.py. If NCCL stalls on this node, fall back to the single-GPU
# run_v21_twochain.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

# GPUS="all" -> leave CUDA_VISIBLE_DEVICES unset (ranks 0..NPROC-1 use physical
# GPU 0..NPROC-1). A subset string (e.g. "4,5,6,7,8,9") restricts to those, but
# a subset-count vs nproc race on this node intermittently trips a device_count
# assert, so "all" is the robust default.
GPUS="${GPUS:-all}"
NPROC="${NPROC:-6}"
BATCH_SIZE="${BATCH_SIZE:-6}"; ACCUM="${ACCUM:-1}"; SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-225}"
MASTER_PORT="${MASTER_PORT:-29517}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v21_twochain_heterodimer}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}"
RUN_NAME="${RUN_NAME:-twochain_v2p_ddp6_seed${SEED}}"
OUT_DIR="$OUT_ROOT/$RUN_NAME"; LOG_DIR="$OUT_ROOT/logs"; LOG="$LOG_DIR/$RUN_NAME.log"
GATE_LEN_W="${GATE_LEN_W:-0.05}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

args=(
    scripts/train.py
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
    --latent-registration
    --precision bf16 --tf32 --no-wandb
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT ${args[*]}"
    exit 0
fi
echo "v21 two-chain DDP: $NPROC GPUs [$GPUS] | global batch = $NPROC x $BATCH_SIZE x $ACCUM = $((NPROC*BATCH_SIZE*ACCUM))"
echo "Log: $LOG"
CVD_ENV=()
[[ "$GPUS" != "all" ]] && CVD_ENV=(CUDA_VISIBLE_DEVICES="$GPUS")
# NCCL_P2P_DISABLE / NCCL_IB_DISABLE: A6000 has no NVLink; NCCL's PCIe peer-to-
# peer path deadlocks on this node (all ranks spin at 100% on the first
# collective, no progress). Routing collectives through host memory is slower
# but reliable. CUDA_DEVICE_ORDER=PCI_BUS_ID keeps device ids stable.
env "${CVD_ENV[@]}" TORCH_HOME="$TORCH_HOME" PYTHONUNBUFFERED=1 PYTHONPATH=src \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" "${args[@]}" 2>&1 | tee "$LOG"
