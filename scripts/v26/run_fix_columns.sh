#!/usr/bin/env bash
# Rebuild 2-D PWM columns with IC-weighted scoring + runner-up margin, then sweep the margin
# threshold to choose it from data. Gates whether Phase-5 stage C (2-D distillation) is usable.
set -euo pipefail
: "${PWD:?}"
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python
echo "[$(date +%T)] === rebuild 2-D columns (IC-weighted + margin) ==="
"$PY" scripts/v26/build_contact_2d_columns.py --datasets core
echo "[$(date +%T)] === margin sweep diagnostic ==="
"$PY" scripts/v26/diagnose_pwm_column_alignment.py
echo "[$(date +%T)] COLUMN ALIGNMENT FIX COMPLETE"
