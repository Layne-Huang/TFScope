#!/usr/bin/env bash
# Score finished B2/B3/B4 checkpoints (all seeds) through the FROZEN unified
# evaluator on idle GPUs, in parallel (per-tag output to avoid JSON races), then
# merge into unified_models.json. No test-best re-selection (ckpt_best.pt = the
# validation-selected checkpoint).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
source /data1/leihuang/miniconda3/etc/profile.d/conda.sh; conda activate tfscope
export TORCH_HOME=/data1/leihuang/.cache/torch PYTHONPATH=.:src CUDA_DEVICE_ORDER=PCI_BUS_ID
RES=results/iclr_phase1_apples_to_apples; PART=$RES/_parts; mkdir -p "$PART"
GPUS=(0 1 2 3 4 5 6 7 8)   # device 9 CUDA-dead
i=0
for v in B2 B3 B4; do for s in 42 1 7; do
  ck=checkpoints/iclr_phase1/$v/seed$s/ckpt_best.pt
  [ -f "$ck" ] || { echo "skip $v/$s (no ckpt)"; continue; }
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$gpu python -m iclr.score_checkpoint_unified \
      --ckpt "$ck" --tag ${v}_seed${s} --device cuda \
      --out "$PART/${v}_seed${s}.json" >"$PART/${v}_seed${s}.log" 2>&1 ) &
done; done
wait
python - <<'PY'
import json, glob, os
RES="results/iclr_phase1_apples_to_apples"
merged=json.load(open(f"{RES}/unified_models.json")) if os.path.exists(f"{RES}/unified_models.json") else {}
for f in glob.glob(f"{RES}/_parts/*.json"):
    merged.update(json.load(open(f)))
json.dump(merged, open(f"{RES}/unified_models.json","w"), indent=2)
print("models in unified_models.json:", sorted(merged.keys()))
PY