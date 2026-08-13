#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v19_e5_structural_register_bf16_ddp3}"
EPOCHS="${EPOCHS:-200}"
GPU_GROUPS=("${GPU_GROUP_0:-0,1,2}" "${GPU_GROUP_1:-3,4,5}" "${GPU_GROUP_2:-6,7,8}")
SEEDS=(42 43 44)
pids=()
names=()

cd "$ROOT"
mkdir -p "$OUT_ROOT/launcher_logs"

for index in 0 1 2; do
    seed="${SEEDS[$index]}"
    gpu_ids="${GPU_GROUPS[$index]}"
    name="e5_structural_rag_seed${seed}"
    echo "Launching $name on GPUs $gpu_ids"
    GPU_IDS="$gpu_ids" \
    SEED="$seed" \
    EPOCHS="$EPOCHS" \
    OUT_ROOT="$OUT_ROOT" \
    bash scripts/run_v19_e5_rag_ddp3.sh \
        >"$OUT_ROOT/launcher_logs/${name}.log" 2>&1 &
    pids+=("$!")
    names+=("$name")
done

status=0
for index in 0 1 2; do
    if wait "${pids[$index]}"; then
        echo "COMPLETED ${names[$index]}"
    else
        echo "FAILED ${names[$index]} (see $OUT_ROOT/launcher_logs/${names[$index]}.log)" >&2
        status=1
    fi
done
exit "$status"
