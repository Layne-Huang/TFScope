#!/usr/bin/env bash
# Detached Boltz-2 foldability campaign: v24 vs DeepPBS predicted-consensus DNA.
# 82 folds (41 TFs x {v24, deeppbs}). MSA via colabfold server; identical protein
# sequences are cached server-side, so each unique protein's MSA is effectively
# computed once and reused across its two DNA conditions (and repeats). Folds are
# distributed across GPUs; ipTM/pLDDT collected from summary_confidences.json.
#
# Launch:
#   setsid bash iclr/run_boltz_foldability.sh >/data1/leihuang/TFScope_store/boltz_v24/driver.log 2>&1 </dev/null &
set -uo pipefail
BOLTZ=/data1/leihuang/miniconda3/envs/boltz/bin/boltz
ROOT=/data1/leihuang/TFScope_store/boltz_v24
IN=$ROOT/inputs; OUT=$ROOT/out; CACHE=/data1/leihuang/.cache/boltz
GPUS=(0 1 2 3)                      # spread folds over 4 GPUs (device 9 is dead; 0-8 ok)
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

yamls=( "$IN"/*.yaml )
echo "[boltz] START $(date)  ${#yamls[@]} folds over ${#GPUS[@]} GPUs"

fold_one() {  # $1 = yaml, $2 = gpu
  local y="$1" gpu="$2" name; name=$(basename "$y" .yaml)
  if find "$OUT" -path "*${name}*summary_confidences.json" 2>/dev/null | grep -q .; then
    echo "[gpu$gpu] SKIP $name (done)"; return; fi
  echo "[gpu$gpu] FOLD $name $(date)"
  CUDA_VISIBLE_DEVICES="$gpu" "$BOLTZ" predict "$y" \
    --out_dir "$OUT" --cache "$CACHE" \
    --use_msa_server --msa_server_url https://api.colabfold.com \
    --output_format pdb --num_workers 2 --override \
    >"$ROOT/logs_${name}.log" 2>&1 \
    && echo "[gpu$gpu] DONE $name" || echo "[gpu$gpu] FAIL $name"
}

NG=${#GPUS[@]}
worker() {  # $1 = worker index -> processes yamls[i] for i % NG == wi
  local wi="$1" gpu="${GPUS[$1]}" i
  for ((i=wi; i<${#yamls[@]}; i+=NG)); do fold_one "${yamls[$i]}" "$gpu"; done
  echo "[gpu$gpu] queue done $(date)"
}
mkdir -p "$ROOT"/logs 2>/dev/null || true
for ((w=0; w<NG; w++)); do worker "$w" & done
wait
echo "[boltz] all folds finished $(date). Collect: python -m iclr.collect_boltz_foldability"
