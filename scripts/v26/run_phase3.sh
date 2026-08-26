#!/usr/bin/env bash
# Phase 3: column-alignment diagnostic, then the leakage-controlled split.
set -euo pipefail
: "${PWD:?}"
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python

echo "[$(date +%T)] === diagnostic: PWM column alignment sanity ==="
"$PY" scripts/v26/diagnose_pwm_column_alignment.py

echo "[$(date +%T)] === Phase 3: leakage-controlled split ==="
"$PY" scripts/v26/build_v26_split.py --dataset core

echo "[$(date +%T)] === tests ==="
"$PY" tests/v26/test_legacy_untouched.py
echo "[$(date +%T)] PHASE 3 COMPLETE"
