#!/usr/bin/env bash
# One GPU worker: runs its assigned (config,seed) jobs SEQUENTIALLY on a single GPU.
# Launched once per GPU by run_sweep.sh, so 5 workers = 5 concurrent runs and no races.
#
#   run_sweep_worker.sh <gpu_index> <job1> [job2 ...]      job = "<config>:<seed>"
#
# Deliberately NOT `set -e`: a single failed run must not kill the rest of the queue.
# Already-finished runs are skipped via their DONE.json, so a worker can be relaunched safely.
set -uo pipefail
: "${PWD:?}"
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python
CKROOT=/data1/leihuang/TFScope_store/checkpoints/v26
EPOCHS=${EPOCHS:-225}
PATIENCE=${PATIENCE:-30}

GPU="$1"; shift                      # a GPU UUID (GPU-xxxxxxxx...), not an index
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[$(date +%T)] worker pinned to $GPU, queue: $*"

# Guard: exactly one device must be visible, and it must be the requested UUID. Index-based
# pinning silently placed runs on other users' GPUs (3/6/7); UUIDs cannot be misresolved.
seen=$("$PY" - <<'PYCHK'
import torch
n = torch.cuda.device_count()
print(n, torch.cuda.get_device_properties(0).uuid if n == 1 else "NA")
PYCHK
)
echo "[$(date +%T)] visible devices: $seen"
case "$seen" in
  "1 "*) : ;;
  *) echo "[$(date +%T)] ABORT: expected exactly 1 visible device, got: $seen"; exit 30 ;;
esac
used=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader \
        | grep -F "${GPU}" | awk -F', ' '{s+=$2} END {print s+0}')
if [ "${used:-0}" -gt 2000 ]; then
  echo "[$(date +%T)] ABORT: $GPU already has ${used} MiB in use by another job"; exit 31
fi

for job in "$@"; do
  cfg="${job%%:*}"; seed="${job##*:}"
  name="$(basename "$cfg" .yaml)"
  out="$CKROOT/${name}/seed${seed}"
  if [ -f "$out/DONE.json" ]; then
    echo "[$(date +%T)] SKIP $name seed$seed (already DONE)"; continue
  fi
  mkdir -p "$out"
  echo "[$(date +%T)] START $name seed$seed -> $out"
  "$PY" scripts/v26/train_v26.py --config "$cfg" --seed "$seed" --out "$out" \
        --epochs "$EPOCHS" --patience "$PATIENCE" > "$out/train.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date +%T)] OK   $name seed$seed"
    tail -1 "$out/train.log" | sed 's/^/      /'
  else
    echo "[$(date +%T)] FAIL $name seed$seed rc=$rc (see $out/train.log)"
    tail -5 "$out/train.log" | sed 's/^/      /'
  fi
done
echo "[$(date +%T)] worker on GPU $GPU FINISHED"
