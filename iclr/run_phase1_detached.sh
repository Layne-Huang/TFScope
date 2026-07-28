#!/usr/bin/env bash
# Detached Phase-I campaign runner (survives session close via setsid).
#
# Runs training-free baselines (B0,B1), then trains B2–B7 across seeds — one job
# per GPU, each GPU processing a serial queue — and evaluates every checkpoint on
# the immutable 291-row test with the shared gene_covR protocol. B8 (frozen v24)
# is evaluated only if a checkpoint+config.json is available.
#
# Launch (detached):
#   setsid bash iclr/run_phase1_detached.sh >/data1/leihuang/TFScope_store/iclr_phase1_logs/driver.log 2>&1 </dev/null &
#
# Monitor:
#   tail -f /data1/leihuang/TFScope_store/iclr_phase1_logs/driver.log
#   ls /data1/leihuang/TFScope_store/iclr_phase1_logs/
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

source /data1/leihuang/miniconda3/etc/profile.d/conda.sh
conda activate tfscope
export TORCH_HOME=/data1/leihuang/.cache/torch
export PYTHONPATH=.:src
export CUDA_DEVICE_ORDER=PCI_BUS_ID

DATA="data/processed/tf_pwm_training_v23.parquet"
SPLIT="data/processed/splits/train_v22/split.json"
OUT_ROOT="${OUT_ROOT:-checkpoints/iclr_phase1}"
LOG_DIR="${LOG_DIR:-/data1/leihuang/TFScope_store/iclr_phase1_logs}"
RESULTS="${RESULTS:-results/iclr_phase1}"
SEEDS=(${SEEDS:-42 1 7})
GPUS=(${GPUS:-0 1 2 3 4 5 6 7 8})        # GPU9 is CUDA-dead on this node (verified 2026-07-28)
VARIANTS=(${VARIANTS:-B2 B3 B4 B5 B6 B7})
mkdir -p "$LOG_DIR" "$RESULTS"

echo "[driver] START $(date)  variants=${VARIANTS[*]}  seeds=${SEEDS[*]}  gpus=${GPUS[*]}"

# ── training-free baselines (CPU) ──────────────────────────────────────────────
for V in B0 B1; do
  echo "[driver] baseline $V $(date)"
  python -m iclr.baselines --variant "$V" --train-data "$DATA" --split "$SPLIT" \
    --test-data "$DATA" --test-split "$SPLIT" --out "$OUT_ROOT/$V" \
    >"$LOG_DIR/${V}.log" 2>&1
done

# ── build the job list (VID:SEED) and round-robin onto GPU queues ──────────────
JOBS=()
for V in "${VARIANTS[@]}"; do for S in "${SEEDS[@]}"; do JOBS+=("$V:$S"); done; done
NG=${#GPUS[@]}

run_job() {  # $1=VID:SEED  $2=gpu
  local vid="${1%%:*}" seed="${1##*:}" gpu="$2"
  local tag="${vid}_seed${seed}" outdir="$OUT_ROOT/${vid}/seed${seed}"
  local log="$LOG_DIR/${tag}.log"
  echo "[gpu$gpu] TRAIN $tag $(date)"
  # single source of truth: variants.py emits the exact train command.
  local cmd; cmd="$(python -m iclr.variants "$vid" --seed "$seed" --out-root "$OUT_ROOT" | grep -v '^#')"
  CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" >"$log" 2>&1
  if [[ -f "$outdir/ckpt_best.pt" ]]; then
    echo "[gpu$gpu] EVAL  $tag $(date)"
    CUDA_VISIBLE_DEVICES="$gpu" python -m iclr.evaluate --ckpt "$outdir/ckpt_best.pt" \
      --test-data "$DATA" --test-split "$SPLIT" --tag "$tag" --out "$RESULTS/$vid" \
      >>"$log" 2>&1
  else
    echo "[gpu$gpu] WARN  $tag produced no ckpt_best.pt (see $log)"
  fi
}

# worker per GPU: process every job whose index mod NG == worker index
worker() {
  local wi="$1" gpu="${GPUS[$1]}"
  local i
  for ((i=wi; i<${#JOBS[@]}; i+=NG)); do
    run_job "${JOBS[$i]}" "$gpu"
  done
  echo "[gpu$gpu] queue done $(date)"
}

for ((w=0; w<NG; w++)); do worker "$w" & done
wait
echo "[driver] all training/eval jobs finished $(date)"

# ── B8: frozen v24 reference (eval only, if available) ─────────────────────────
V24_CKPT="${V24_CKPT:-checkpoints/v24_e1_paired/pwmhead_ft.pt}"
if [[ -f "$(dirname "$V24_CKPT")/config.json" ]]; then
  echo "[driver] B8 eval frozen v24 $(date)"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -m iclr.evaluate --ckpt "$V24_CKPT" \
    --test-data "$DATA" --test-split "$SPLIT" --tag B8_v24 --out "$RESULTS/B8" \
    >"$LOG_DIR/B8.log" 2>&1 || echo "[driver] B8 eval failed (see B8.log)"
else
  echo "[driver] B8 SKIPPED: no config.json next to $V24_CKPT (canonical v24 base ckpt not on this node)"
fi

echo "[driver] DONE $(date). Results in $RESULTS/, logs in $LOG_DIR/"
