#!/usr/bin/env bash
set -uo pipefail
cd /afs/csail.mit.edu/u/l/leihuang/project/TFScope
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python
export TORCH_HOME=/data1/leihuang/.cache/torch PYTHONPATH=src
GPU=GPU-26df3b25-f077-10ed-57eb-47e5a71c0cef
FT=/data1/leihuang/project/TFScope/checkpoints/v24ft_mutation/v24ft_seed42
LOG=/data1/leihuang/project/TFScope/checkpoints/v24ft_mutation/logs/v24ft_seed42.log
RES=results/mutation_benchmark
while true; do
  if grep -qE "Done\. Best|Early stopping at epoch" "$LOG" 2>/dev/null && [ -f "$FT/ckpt_best.pt" ]; then break; fi
  sleep 240
done
sleep 60; aklog 2>/dev/null || true
# eval FT model on all 55 pairs (saves barrera_pairs.json = FT)
CUDA_VISIBLE_DEVICES=$GPU BENCH_CK=$FT $PY scripts/barrera_mutation_benchmark.py > "$RES/v24ft_eval.log" 2>&1
# compare HELD-OUT genes: FT vs v24 baseline
$PY - <<'PY2' >> "$RES/v24ft_eval.log" 2>&1
import json,numpy as np
ho=set(json.load(open("results/mutation_benchmark/heldout_genes.json")))
base=json.load(open("results/mutation_benchmark/barrera_pairs_v24base.json"))
ft=json.load(open("results/mutation_benchmark/barrera_pairs.json"))
def held(d):
    idx=[i for i,p in enumerate(d["pairs"]) if p["gene"] in ho]
    return idx
def report(tag,d):
    idx=held(d); dp=np.array(d["dpred"])[idx]; dt=np.array(d["dtrue"])[idx]
    print(f"{tag} HELD-OUT (n={len(idx)}): mean pred change={dp.mean():.3f} (measured {dt.mean():.3f}) corr={np.corrcoef(dp,dt)[0,1]:.3f}")
print("=== held-out genes:",sorted(ho))
report("v24 base", base); report("v24-FT ", ft)
PY2
echo "V24FT AUTOTEST DONE" > "$RES/DONE.flag"
