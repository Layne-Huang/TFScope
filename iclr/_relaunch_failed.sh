#!/usr/bin/env bash
# Relaunch the two jobs lost to the dead GPU9 (B4:7, B7:7) serially on GPU0.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
source /data1/leihuang/miniconda3/etc/profile.d/conda.sh
conda activate tfscope
export TORCH_HOME=/data1/leihuang/.cache/torch PYTHONPATH=.:src CUDA_DEVICE_ORDER=PCI_BUS_ID
DATA="data/processed/tf_pwm_training_v23.parquet"
SPLIT="data/processed/splits/train_v22/split.json"
OUT_ROOT="checkpoints/iclr_phase1"; RESULTS="results/iclr_phase1"
LOG_DIR="/data1/leihuang/TFScope_store/iclr_phase1_logs"
GPU=0
for spec in B4:7 B7:7; do
  vid="${spec%%:*}"; seed="${spec##*:}"; tag="${vid}_seed${seed}"
  outdir="$OUT_ROOT/${vid}/seed${seed}"; log="$LOG_DIR/${tag}.log"
  echo "[relaunch gpu$GPU] TRAIN $tag $(date)"
  cmd="$(python -m iclr.variants "$vid" --seed "$seed" --out-root "$OUT_ROOT" | grep -v '^#')"
  CUDA_VISIBLE_DEVICES="$GPU" bash -c "$cmd" >"$log" 2>&1
  if [[ -f "$outdir/ckpt_best.pt" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python -m iclr.evaluate --ckpt "$outdir/ckpt_best.pt" \
      --test-data "$DATA" --test-split "$SPLIT" --tag "$tag" --out "$RESULTS/$vid" >>"$log" 2>&1
    echo "[relaunch gpu$GPU] EVAL done $tag $(date)"
  fi
done
echo "[relaunch] done $(date)"
