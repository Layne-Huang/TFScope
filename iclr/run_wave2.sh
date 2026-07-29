#!/usr/bin/env bash
# Recovery launcher for wave-2 (B5/B6/B7 x 3 seeds) after the driver died on a
# transient AFS glitch. Hardened: the exact train commands are captured ONCE at
# launch (single iclr.variants import while AFS is healthy) into an array, then
# each GPU worker runs its pre-captured command string — no repeated runtime
# imports. One job per GPU, serial queue per GPU. Does NOT touch B0-B4 outputs.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
source /data1/leihuang/miniconda3/etc/profile.d/conda.sh; conda activate tfscope
export TORCH_HOME=/data1/leihuang/.cache/torch PYTHONPATH=.:src CUDA_DEVICE_ORDER=PCI_BUS_ID
LOG_DIR=/data1/leihuang/TFScope_store/iclr_phase1_logs; mkdir -p "$LOG_DIR"
OUT_ROOT=checkpoints/iclr_phase1
GPUS=(0 1 2 3 4 5 6 7 8)          # device 9 is CUDA-dead; 0 is good
SEEDS=(42 1 7); VARIANTS=(B5 B6 B7)

# --- capture commands once (fail loudly here if AFS/import is broken) ---
TAGS=(); CMDS=()
for v in "${VARIANTS[@]}"; do for s in "${SEEDS[@]}"; do
  c=$(python -m iclr.variants "$v" --seed "$s" --out-root "$OUT_ROOT" | grep -v '^#')
  if [[ -z "$c" ]]; then echo "[FATAL] empty command for $v seed$s — aborting"; exit 1; fi
  TAGS+=("${v}_seed${s}"); CMDS+=("$c")
done; done
echo "[wave2] captured ${#CMDS[@]} commands $(date)"

NG=${#GPUS[@]}
worker() {
  local wi="$1" gpu="${GPUS[$1]}" i
  for ((i=wi; i<${#CMDS[@]}; i+=NG)); do
    local tag="${TAGS[$i]}" cmd="${CMDS[$i]}" out="$OUT_ROOT/${TAGS[$i]/_seed//seed}"
    echo "[gpu$gpu] TRAIN $tag $(date)"
    CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" >"$LOG_DIR/${tag}.log" 2>&1 \
      && echo "[gpu$gpu] DONE $tag $(date)" || echo "[gpu$gpu] FAIL $tag $(date)"
  done
  echo "[gpu$gpu] queue done $(date)"
}
for ((w=0; w<NG && w<${#CMDS[@]}; w++)); do worker "$w" & done
wait
echo "[wave2] all B5/B6/B7 training finished $(date). Score via iclr.score_checkpoint_unified."
