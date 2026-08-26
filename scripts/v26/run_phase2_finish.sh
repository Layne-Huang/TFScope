#!/usr/bin/env bash
# Phase-2 remaining steps: 2-D PWM columns, recognition prior, v24-compatible rebuild, tests.
set -euo pipefail
: "${PWD:?}"
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python

echo "[$(date +%T)] === step 3b: 2-D PWM-column assignment ==="
"$PY" scripts/v26/build_contact_2d_columns.py --datasets core flank20 flank32

echo "[$(date +%T)] === step 4: rule-based recognition prior (UniProt coords) ==="
"$PY" scripts/v26/build_recognition_prior_v26.py

echo "[$(date +%T)] === step 5: v24-compatible contact targets (fair ablation) ==="
"$PY" scripts/v26/rebuild_v24_compatible_contacts.py --dataset core

echo "[$(date +%T)] === tests ==="
"$PY" tests/v26/test_contact_projection.py
"$PY" tests/v26/test_dbd_spans.py
"$PY" tests/v26/test_legacy_untouched.py

echo "[$(date +%T)] PHASE 2 COMPLETE"
