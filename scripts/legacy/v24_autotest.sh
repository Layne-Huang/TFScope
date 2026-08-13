#!/usr/bin/env bash
# Babysitter: wait for v24 training to finish, then auto-run all case-study tests.
# Launch detached: setsid nohup bash scripts/v24_autotest.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python
export TORCH_HOME=/data1/leihuang/.cache/torch PYTHONPATH=src
GPU=GPU-26df3b25-f077-10ed-57eb-47e5a71c0cef   # index 4 (v24 frees it on finish)
RUN=/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42
LOG=/data1/leihuang/project/TFScope/checkpoints/v24_contact/logs/contact_v24_seed42.log
DATA=data/processed/tf_pwm_training_v23.parquet
SPLIT=data/processed/splits/train_v22/split.json
RES=results/v24_autotest; mkdir -p "$RES"

echo "[autotest] waiting for v24 to finish ..."
while true; do
  if grep -qE "Done\. Best|Early stopping at epoch" "$LOG" 2>/dev/null && [ -f "$RUN/ckpt_best.pt" ]; then break; fi
  sleep 300
done
sleep 90    # let the final checkpoint flush
aklog 2>/dev/null || true
echo "[autotest] v24 finished. best: $(grep -oE 'Best oracle r[: ]+[0-9.]+' "$LOG" | tail -1)"

# --- 1. TEST covR diagnostic (v24) ---
echo "[autotest] (1/3) test diagnostic"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_v22_diagnostics.py \
  --checkpoint "$RUN/ckpt_best.pt" --data "$DATA" --split "$SPLIT" --split-name test \
  --out results/v22_ablation/contact_v24_seed42.json > "$RES/diag.log" 2>&1

# --- add v24 to the multi-model eval lists (idempotent) ---
$PY - <<'PY'
for f in ("scripts/eval_mutations_all_models.py","scripts/eval_designs_all_models.py"):
    s=open(f).read()
    if "v24_contact" not in s:
        anchor=' ("v23_fulldata",     f"{CK}/v23_fulldata/nchain_v23_full_seed42",         "tf_pwm_training_v23"),\n'
        s=s.replace(anchor, anchor+' ("v24_contact",      f"{CK}/v24_contact/contact_v24_seed42",               "tf_pwm_training_v23"),\n')
        open(f,"w").write(s); print("added v24 to",f)
PY

# --- 2. mutation case ---
echo "[autotest] (2/3) mutation case"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_mutations_all_models.py > "$RES/mutation.log" 2>&1

# --- 3. design case (corrected direction) ---
echo "[autotest] (3/3) design case"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_designs_all_models.py > "$RES/design.log" 2>&1

# --- summary ---
{
  echo "=== V24 AUTOTEST SUMMARY ==="; date
  echo; echo "--- test covR (v24 vs v23) ---"
  $PY - <<'PY'
import json
for name,f in [("v24",'results/v22_ablation/contact_v24_seed42.json'),("v23",'results/v22_ablation/nchain_v23_seed42.json')]:
    m=json.load(open(f))["metrics"]["predicted_gate"]
    print(f"  {name}: row={m['row_mean']:.3f} gene-bal={m['gene_balanced_mean']:.3f}")
PY
  echo; echo "--- mutation (MyoD1 row) ---"; grep -E "v24_contact|v23_nchain|combined " "$RES/mutation.log" | grep -iE "myod|switch|GCTG|CACG" | head
  grep -A20 "MyoD1" "$RES/mutation.log" | grep -E "v24_contact|v23" | head
  echo; echo "--- design leaderboard ---"; sed -n '/leaderboard/,$p' "$RES/design.log"
} > "$RES/SUMMARY.txt" 2>&1
echo "AUTOTEST DONE" > "$RES/DONE.flag"
echo "[autotest] done -> $RES/SUMMARY.txt"
