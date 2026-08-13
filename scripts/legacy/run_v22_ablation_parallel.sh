#!/usr/bin/env bash
# Run the 4-stage x 3-seed v22 matrix on independent GPUs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7 8}"
SEEDS=(42 43 44)
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v22_ablation}"
LOG_ROOT="$OUT_ROOT/launcher_logs"
mkdir -p "$LOG_ROOT"

run_job() {
  local stage="$1" seed="$2" gpu="$3"
  echo "launch stage=$stage seed=$seed gpu=$gpu"
  STAGES="$stage" SEEDS="$seed" CUDA_VISIBLE_DEVICES="$gpu" \
    OUT_ROOT="$OUT_ROOT" scripts/run_v22_ablation.sh \
    >"$LOG_ROOT/${stage}_seed${seed}.log" 2>&1
}

pids=()
job=0
for stage in data span loss; do
  for seed in "${SEEDS[@]}"; do
    if (( job >= ${#GPUS[@]} )); then
      echo "At least 9 GPUs are required for the first ablation round" >&2
      exit 2
    fi
    run_job "$stage" "$seed" "${GPUS[$job]}" &
    pids+=("$!")
    ((job+=1))
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
for index in "${!SEEDS[@]}"; do
  run_job multichain "${SEEDS[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
"$PYTHON_BIN" scripts/summarize_v22_ablation.py
echo "all v22 ablations completed"
