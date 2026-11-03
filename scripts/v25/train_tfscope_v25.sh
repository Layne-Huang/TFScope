#!/usr/bin/env bash
# TFScope on the frozen unified-v25 split, WITHOUT the family-id input. 2-GPU DDP.
#
# Identical to the v24 recipe (scripts/run_v24_contact_ddp.sh) except for four things,
# all of them deliberate:
#   --data/--split        the frozen unified_v25 table + 40%-identity split, which puts
#                         all 84 DeepPBS benchmark rows in test
#   --no-family-id        the family label is not an input; family embeddings collapse
#                         to one learned constant and family_id is zeroed at the model
#                         entry, so the prediction is a function of the sequence alone
#   contact json          *_unified_v25.json, re-framed onto these crops. The v23 files
#                         indexed a DIFFERENT crop frame, which put 181/309 rows' residue
#                         indices past the end of their own sequence, silently clipped
#   global batch 36       2 GPUs x batch 6 x accum 3 (v24 was 6 x 6 x 1, LOFO 1 x 12 x 3).
#                         The GLOBAL batch is held at 36 so the optimisation is the same
#                         recipe; only the micro-batch/accum division differs, which is a
#                         pure throughput knob. 18x1 was measured and is NOT faster
#                         (177s vs 165s/epoch): sequences span 15-1128 aa, so a larger
#                         micro-batch just pads more.
#
# The dead GPU 0 is invisible to CUDA, so GPUs are pinned by UUID: an index-based
# CUDA_VISIBLE_DEVICES lands one card off. NCCL_P2P_DISABLE is required on this node
# (A6000s without NVLink hang in the P2P path).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"

# nvidia-smi 1,2 -> CUDA 0,1
GPUS="${GPUS:-GPU-808f6dc0-74c2-c81b-b926-f15a367f6a4d,GPU-a0253811-eb51-fb1d-9951-80e883714728}"
NPROC="${NPROC:-2}"; MASTER_PORT="${MASTER_PORT:-29611}"
BATCH_SIZE="${BATCH_SIZE:-6}"; ACCUM="${ACCUM:-3}"; SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-225}"
DATA="${DATA:-data/processed/tf_pwm_unified_v25.parquet}"
SPLIT="${SPLIT:-data/processed/splits/unified_v25/split.json}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/TFScope_store/checkpoints/v25}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
RUN_NAME="${RUN_NAME:-v25_nofam_seed${SEED}}"
OUT="$OUT_ROOT/$RUN_NAME"
mkdir -p "$OUT"

# Fail in seconds, not after ESM loads, if the cards were mis-pinned: an out-of-range
# CUDA_VISIBLE_DEVICES yields a silent CPU fallback that only trips much later on
# "--precision bf16 requires CUDA".
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
  --family-embedding-path none --no-family-id
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

echo "v25 no-family-id DDP: $NPROC GPUs | global batch=$((NPROC*BATCH_SIZE*ACCUM)) | out=$OUT"
exec env CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 PYTHONPATH=src \
  CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" "${args[@]}"
