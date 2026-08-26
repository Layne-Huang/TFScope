#!/usr/bin/env bash
# Train the V24 ARCHITECTURE on V26 DATA (flank20 = flanks + partner chains, v26 clean split).
#
# Mirror of the v23compat experiment. Hyperparameters copied verbatim from
# scripts/run_v24_contact_ddp.sh so the only changes are --data and --split.
#
# Contact supervision is OFF, matching what v26 actually had on this comparison
# (v23compat had 0 contact-labelled rows). Turning it on for v24 only would reintroduce a
# confound the whole point of this run is to remove.
set -uo pipefail
: "${PWD:?}"
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python
OUT=${OUT:-/data1/leihuang/TFScope_store/checkpoints/v26/v24arch_on_v26data/seed42}
mkdir -p "$OUT"

"$PY" scripts/train.py \
  --data data/processed/v26/v24compat_flank20.parquet \
  --split data/processed/splits/v26/split_v24compat_flank20.json --out "$OUT" \
  --seed 42 --epochs 225 --batch-size 8 --grad-accum-steps 2 \
  --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
  --warmup-steps 150 --workers 2 --save-every 25 \
  --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
  --pwm-head-v18 --group-balanced-sampling \
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
rc=$?
echo "[$(date +%T)] v24-arch-on-v26-data exit=$rc"
exit $rc
