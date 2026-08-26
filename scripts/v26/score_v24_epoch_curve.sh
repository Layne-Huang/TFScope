#!/usr/bin/env bash
# Score EVERY saved v24 checkpoint on v24's own benchmark, to answer:
# how much better is the early-stopped "best" (epoch 60) than training to epoch 200?
#
# Uses the LEGACY v24 benchmark on purpose (tf_pwm_training_v23 + train_v22 split). That split
# has known leakage (docs/v26_audit.md Findings A/B), but the question here is about v24's own
# training dynamics measured the way v24 measured them -- so the legacy protocol is the correct
# one, and the leakage caveat applies equally to every point on the curve.
#
# Writes to results/v26/ ONLY. It must not touch
# results/iclr_phase1_apples_to_apples/unified_models.json (a legacy artifact).
set -uo pipefail          # not -e: one bad checkpoint must not abort the curve
: "${PWD:?}"
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python
CK=/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42
OUT=results/v26/v24_epoch_curve.json

n_ok=0; n_fail=0
echo "[$(date +%T)] scoring v24 checkpoint curve -> $OUT"
for f in ckpt_best ckpt_epoch025 ckpt_epoch050 ckpt_epoch075 ckpt_epoch100 \
         ckpt_epoch125 ckpt_epoch150 ckpt_epoch175 ckpt_epoch200; do
  p="$CK/$f.pt"
  [ -f "$p" ] || { echo "  skip $f (absent)"; continue; }
  echo "[$(date +%T)] --- $f"
  # Capture to a temp file rather than piping into grep: a pipe discards the payload's exit
  # status, which is how an earlier version reported COMPLETE after 9/9 failures.
  tmp=$(mktemp)
  "$PY" -m iclr.score_checkpoint_unified --ckpt "$p" --tag "v24_$f" \
        --device cuda --out "$OUT" > "$tmp" 2>&1
  rc=$?
  grep -E "PanelA|content_r" "$tmp" | head -3
  if [ $rc -ne 0 ]; then
    n_fail=$((n_fail+1)); echo "  FAILED rc=$rc"; tail -3 "$tmp" | sed 's/^/    /'
  else
    n_ok=$((n_ok+1))
  fi
  rm -f "$tmp"
done
echo "[$(date +%T)] curve done: $n_ok ok, $n_fail failed"
if [ "$n_ok" -eq 0 ]; then echo "ABORT: no checkpoint scored"; exit 40; fi
if [ "$n_fail" -gt 0 ]; then exit 41; fi
echo "[$(date +%T)] V24 EPOCH CURVE COMPLETE"
