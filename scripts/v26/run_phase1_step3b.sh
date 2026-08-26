#!/usr/bin/env bash
# Phase-1 step 3b: apply triaged whitelist additions, rebuild DBD spans, re-triage to confirm
# the residual shrank. Fails loudly on any error (see docs/v26_audit.md §2.3 for why).
set -euo pipefail
# NOTE: run_detached.sh already cd'd us into the correct root (AFS repo, or the
# /data1 mirror when launched with --mirror). Do NOT hardcode the AFS path here --
# AFS tokens drop every ~10 min on this host and would kill the job mid-run.
: "${PWD:?}"
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python

echo "[$(date +%T)] === backup whitelist before mutation (first time only) ==="
# The pre-triage snapshot is the reference for the zero-drift invariant in
# tests/v26/test_dbd_spans.py. Overwriting it with an already-augmented whitelist would
# silently make that test vacuous, so create it once and never again.
if [ -f data/annotations_v26/dbd_pfam_whitelist.pre_triage.json ]; then
  echo "  pre_triage snapshot already exists -- preserving it as the drift reference"
else
  cp -v data/annotations_v26/dbd_pfam_whitelist.json \
        data/annotations_v26/dbd_pfam_whitelist.pre_triage.json
fi

echo "[$(date +%T)] === apply whitelist additions ==="
"$PY" scripts/v26/triage_missing_dbd.py --apply

echo "[$(date +%T)] === rebuild DBD candidate spans ==="
"$PY" scripts/v26/build_dbd_spans.py

echo "[$(date +%T)] === re-triage (residual must be small and explicit) ==="
"$PY" scripts/v26/triage_missing_dbd.py

echo "[$(date +%T)] === verify ==="
for f in data/processed/v26/dbd_candidates.parquet \
         results/v26/missing_dbd_triage.csv \
         data/annotations_v26/dbd_pfam_whitelist.json; do
  [[ -s "$f" ]] || { echo "ABORT: missing $f"; exit 15; }
  echo "  ok  $f"
done
echo "[$(date +%T)] PHASE 1 STEP 3b COMPLETE"
