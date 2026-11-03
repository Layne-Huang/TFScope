#!/usr/bin/env bash
# Retrain the DeepPBS 5-model ensemble on the frozen unified-v25 split, one GPU.
#
# Folds come from scripts/build_deeppbs_folds_v25.py, which projects TFScope's split
# onto DeepPBS's structure files, so the two models train and test on exactly the same
# partition. Neither the released checkpoints (which saw all 20 benchmark genes) nor
# the earlier `pdbdisjoint` retrain (cut on a different criterion) could claim that.
#
# Runs in the dedicated `deeppbs` env: its torch 2.3.0+cu121 / PyG 2.5.0 matches the
# compiled C++ extensions, which segfault under the multiflow env.
#
# DeepPBS is batch_size=1 over small surface graphs, so all five folds fit on one
# A6000 concurrently and finish in roughly the wall-clock of a single fold. Drop
# CONCURRENCY if memory is tight.
set -uo pipefail

DP=/data1/leihuang/miniconda3/envs/deeppbs/bin/python
RUNDIR=/data1/leihuang/DeepPBS/deeppbsmar24/run
FOLDS="${FOLDS:-v25_folds}"
OUT="${OUT:-/data1/leihuang/DeepPBS/v25_retrain}"
# nvidia-smi 3. Pinned by UUID: the dead GPU 0 is invisible to CUDA, so an index here
# would silently select a different card.
GPU="${GPU:-GPU-826b989f-a711-acc9-a350-9de857b0b2cf}"
CONCURRENCY="${CONCURRENCY:-5}"
EVAL_EVERY="${EVAL_EVERY:-2}"

cd "$RUNDIR" || exit 90
mkdir -p "$OUT"
for k in 0 1 2 3 4; do
  for f in "train$k.txt" "valid$k.txt"; do
    [[ -s "$FOLDS/$f" ]] || { echo "MISSING $RUNDIR/$FOLDS/$f" >&2; exit 2; }
  done
done
cp "$FOLDS"/{manifest.json,test149.txt,test84_benchmark.txt} "$OUT/" 2>/dev/null || true

echo "[$(date +%T)] DeepPBS retrain on $FOLDS -> $OUT (concurrency $CONCURRENCY)"
pids=()
for k in 0 1 2 3 4; do
  while (( $(jobs -rp | wc -l) >= CONCURRENCY )); do sleep 20; done
  CUDA_VISIBLE_DEVICES="$GPU" nohup "$DP" -W ignore driver.py \
    "$FOLDS/train${k}.txt" "$FOLDS/valid${k}.txt" -c "$FOLDS/config.json" \
    --single_gpu --output_path "$OUT" --run_name "m${k}" --eval_every "$EVAL_EVERY" \
    > "$OUT/m${k}.log" 2>&1 &
  pids+=($!)
  echo "[$(date +%T)]  m${k} pid ${pids[-1]}  ($(wc -l < "$FOLDS/train${k}.txt") train / $(wc -l < "$FOLDS/valid${k}.txt") valid)"
  sleep 10
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
printf "m0\nm1\nm2\nm3\nm4\n" > "$OUT/model_list.txt"
if (( fail )); then
  echo "[$(date +%T)] AT LEAST ONE FOLD FAILED -- check $OUT/m*.log" | tee "$OUT/RETRAIN_FAILED"
  exit 1
fi
echo "[$(date +%T)] ALL DONE" | tee "$OUT/RETRAIN_DONE"
