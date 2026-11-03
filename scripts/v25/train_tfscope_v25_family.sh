#!/usr/bin/env bash
# The family_id ablation control: byte-identical to train_tfscope_v25.sh EXCEPT that
# `--no-family-id` is dropped, so the family label IS an input.
#
# It exists to answer two questions that are currently entangled:
#   1. what does removing family_id actually cost on the frozen v25 split?
#   2. how much of v24's 0.846-vs-v25's 0.615 on the identical 84 benchmark rows comes
#      from the family channel? That gap had been attributed to training leakage, but the
#      statistic behind that claim turned out to be chance-saturated (250 different-gene
#      PWMs already give a median best match of 0.995), so the gap is unexplained and
#      family_id is now the leading candidate.
#
# Same 225-epoch budget as the no-family run even though validation plateaus by ~epoch 75.
# Shortening it would confound the one variable this experiment is meant to isolate, and
# checkpoints are saved every 25 epochs so both runs can be reselected the same way.
#
# Checkpoint selection is now content_r (the project's single r), not covR — see
# `one-r-content-r`. The no-family run was selected on covR, so compare the two at their
# val-content_r-best checkpoints, not at their shipped ckpt_best.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"

# nvidia-smi 7,8 -> CUDA 6,7. Pinned by UUID: the dead GPU 0 is invisible to CUDA.
GPUS="${GPUS:-GPU-c9320d8a-9633-cf7a-f341-ac8096c6fc3a,GPU-2cf50fdc-4677-47ce-37fe-bb29d4932668}"
NPROC="${NPROC:-2}"; MASTER_PORT="${MASTER_PORT:-29613}"
BATCH_SIZE="${BATCH_SIZE:-6}"; ACCUM="${ACCUM:-3}"; SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-225}"
DATA="${DATA:-data/processed/tf_pwm_unified_v25.parquet}"
SPLIT="${SPLIT:-data/processed/splits/unified_v25/split.json}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/TFScope_store/checkpoints/v25}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
RUN_NAME="${RUN_NAME:-v25_family_seed${SEED}}"
OUT="$OUT_ROOT/$RUN_NAME"
mkdir -p "$OUT"

env CUDA_VISIBLE_DEVICES="$GPUS" WANT="$NPROC" "$PYTHON_BIN" -c "
import os, sys, torch
n = torch.cuda.device_count()
if n < int(os.environ['WANT']):
    sys.exit('ABORT: CUDA sees %d device(s) for CUDA_VISIBLE_DEVICES=%r -- pin by UUID'
             % (n, os.environ.get('CUDA_VISIBLE_DEVICES')))
for i in range(n):
    print('  gpu%d: %s (%.0f GiB free)' % (i, torch.cuda.get_device_name(i),
          torch.cuda.mem_get_info(i)[0] / 2**30))"

args=(
  scripts/train.py
  --data "$DATA" --split "$SPLIT" --out "$OUT"
  --seed "$SEED" --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE" --grad-accum-steps "$ACCUM"
  --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6
  --warmup-steps 150 --workers 2 --save-every 25
  --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0
  --pwm-head-v18 --group-balanced-sampling
  --v18-contact-supervision --v18-contact-weight 0.3
  --v18-contact-bias-scale 1.0 --v18-contact-bias-learnable 1
  --contact-distill-weight 0.2
  --contact-targets-path data/contact_maps/contact_targets_unified_v25.json
  --recognition-prior-path data/contact_maps/recognition_residues_unified_v25.json
  --family-embedding-path none
  --moe-granularity residue --num-experts 8 --n-shared-experts 2 --top-k 2
  --expert-hidden-dim 512 --balance-loss-weight 0.01 --diversity-loss-weight 0.0
  --gate-length-weight 0.05 --latent-registration
  --gate-mode span --max-motif-length 42 --motif-overflow-policy error
  --pwm-cov-r-weight 0.25 --pwm-core-ic-thresh 0.25
  --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 0
  --oracle-aggregation gene --early-stop-patience 30
  --two-chain-input --chain-id-embedding --max-chains 4
  --precision bf16 --tf32 --no-wandb
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC ${args[*]}"; exit 0
fi

echo "v25 WITH family-id DDP: $NPROC GPUs | global batch=$((NPROC*BATCH_SIZE*ACCUM)) | out=$OUT"
exec env CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 PYTHONPATH=src \
  CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" "${args[@]}"
