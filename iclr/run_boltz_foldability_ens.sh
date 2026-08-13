#!/usr/bin/env bash
# Detached Boltz-2 foldability: v24-ENSEMBLE consensus DNA (differing genes only).
# DeepPBS + identical-gene TFScope folds are reused from the boltz_v24 campaign.
# Launch:
#   setsid bash iclr/run_boltz_foldability_ens.sh >/data1/leihuang/TFScope_store/boltz_v24_ens/driver.log 2>&1 </dev/null &
set -uo pipefail
BOLTZ=/data1/leihuang/miniconda3/envs/boltz/bin/boltz
ROOT=/data1/leihuang/TFScope_store/boltz_v24_ens
IN=$ROOT/inputs; OUT=$ROOT/out; CACHE=/data1/leihuang/.cache/boltz
GPUS=(3 5 2 4)
mkdir -p "$OUT"; export CUDA_DEVICE_ORDER=PCI_BUS_ID
yamls=( "$IN"/*.yaml )
echo "[boltz-ens] START $(date)  ${#yamls[@]} folds over ${#GPUS[@]} GPUs"

fold_one() {
  local y="$1" gpu="$2" name; name=$(basename "$y" .yaml)
  if [ -f "$OUT/boltz_results_${name}/predictions/${name}/confidence_${name}_model_0.json" ]; then
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
worker() { local wi="$1" gpu="${GPUS[$1]}" i; for ((i=wi; i<${#yamls[@]}; i+=NG)); do fold_one "${yamls[$i]}" "$gpu"; done; echo "[gpu$gpu] queue done $(date)"; }
for ((w=0; w<NG; w++)); do worker "$w" & done
wait
echo "[boltz-ens] all folds finished $(date). Collect: python -m iclr.collect_boltz_foldability_ens"
