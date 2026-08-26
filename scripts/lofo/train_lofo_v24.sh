#!/usr/bin/env bash
# Train ONE leave-one-family-out v24 model. Config is byte-for-byte the v24 recipe
# (scripts/run_v24_contact_ddp.sh), re-expressed for a single GPU: micro-batch 12 x
# accum 3 = the same global batch of 36 the 6-GPU DDP run and the 4 ensemble seeds used.
# Only --split and --out differ between families.
#
#   scripts/lofo/train_lofo_v24.sh <FAMILY_TAG>
#
# Not meant to be run directly -- use scripts/lofo/launch_lofo_wave.sh, which pins a
# free GPU and detaches the job so it survives the SSH/agent session.
set -euo pipefail

TAG="${1:?usage: $0 <FAMILY_TAG>   (e.g. C2H2, HMG-SOX)}"
SPLIT="data/processed/splits/lofo_v24/${TAG}.json"
OUT="checkpoints/lofo_v24/${TAG}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"

[[ -f "$SPLIT" ]] || { echo "no such split: $SPLIT" >&2; exit 2; }
mkdir -p "$OUT"

echo "=== LOFO v24 | held-out family = ${TAG} ==="
echo "split: $SPLIT"
echo "out:   $OUT"
"$PYTHON_BIN" - "$SPLIT" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["metadata"]
print(f"  train={m['n_train_rows']:>5} rows / {m['n_train_genes']} genes")
print(f"  val  ={m['n_val_rows']:>5} rows / {m['n_val_genes']} genes")
print(f"  ctrl ={m['n_ctrl_rows']:>5} rows / {m['n_ctrl_genes']} genes "
      f"({m['ctrl_families']} retained families)")
print(f"  test ={m['n_test_rows']:>5} rows / {m['n_test_genes']} genes  "
      f"(held-out family, {m['n_test_rows_from_excluded']} from the excluded buffer)")
print(f"  own ctrl genes for the paired delta: {len(m['own_ctrl_genes'])}")
PY
# Fail in seconds, not after model construction, if the card was mis-pinned. The dead
# GPU 0 is invisible to CUDA, so an index-based CUDA_VISIBLE_DEVICES lands one card off
# and an out-of-range one yields a silent CPU fallback that only trips much later on
# "--precision bf16 requires CUDA".
"$PYTHON_BIN" -c "
import os, sys, torch
if not torch.cuda.is_available():
    sys.exit('ABORT: no CUDA device for CUDA_VISIBLE_DEVICES=%r -- pin the GPU by UUID'
             % os.environ.get('CUDA_VISIBLE_DEVICES'))
print('  gpu: %s (%.0f GiB free)' % (torch.cuda.get_device_name(0),
      torch.cuda.mem_get_info(0)[0] / 2**30))"

exec "$PYTHON_BIN" scripts/train.py \
  --data data/processed/tf_pwm_training_v23.parquet \
  --split "$SPLIT" --out "$OUT" \
  --seed 42 --epochs 225 \
  --batch-size 12 --grad-accum-steps 3 \
  --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
  --warmup-steps 150 --workers 2 --save-every 25 \
  --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
  --pwm-head-v18 --group-balanced-sampling \
  --v18-contact-supervision --v18-contact-weight 0.3 \
  --v18-contact-bias-scale 1.0 --v18-contact-bias-learnable 1 \
  --contact-distill-weight 0.2 \
  --contact-targets-path data/contact_maps/contact_targets_v23.json \
  --recognition-prior-path data/contact_maps/recognition_residues_v23.json \
  --family-embedding-path none \
  --moe-granularity residue --num-experts 8 --n-shared-experts 2 --top-k 2 \
  --expert-hidden-dim 512 --balance-loss-weight 0.01 --diversity-loss-weight 0.0 \
  --gate-length-weight 0.05 --latent-registration \
  --gate-mode span --max-motif-length 42 --motif-overflow-policy error \
  --pwm-cov-r-weight 0.25 --pwm-core-ic-thresh 0.25 \
  --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 0 \
  --oracle-aggregation gene --early-stop-patience 30 \
  --two-chain-input --chain-id-embedding --max-chains 4 \
  --precision bf16 --tf32 --no-wandb
