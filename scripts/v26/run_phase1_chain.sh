#!/usr/bin/env bash
# Phase-1 chain: wait for the annotation snapshot, then build canonical targets + DBD spans.
#
# Failure handling is deliberate. The v25 eval scripts had no `set -e` and piped into
# `grep`, so a conda PermissionError still printed "done" and the run was silently lost
# (docs/v26_audit.md §2.3). This script:
#   * set -euo pipefail
#   * checks the upstream STATUS marker, not just file existence
#   * aborts with a distinct exit code per failure mode
#
# Launch detached:
#   scripts/v26/run_detached.sh v26_p1_chain scripts/v26/run_phase1_chain.sh

set -euo pipefail

REPO="/afs/csail.mit.edu/u/l/leihuang/project/TFScope"
LOGROOT="/data1/leihuang/TFScope_store/v26_logs"
PY="/data1/leihuang/miniconda3/envs/tfscope/bin/python"
UPSTREAM="$LOGROOT/v26_p1_annotation_snapshot"
MAX_WAIT_SEC=7200          # 2 h ceiling on the snapshot fetch

cd "$REPO"

echo "[$(date +%T)] waiting for annotation snapshot ..."
waited=0
while true; do
  st="$(cat "$UPSTREAM/STATUS" 2>/dev/null || echo MISSING)"
  case "$st" in
    DONE)     echo "[$(date +%T)] snapshot DONE"; break ;;
    FAILED:*) echo "[$(date +%T)] ABORT: snapshot $st"; exit 11 ;;
    MISSING)  echo "[$(date +%T)] ABORT: no upstream job at $UPSTREAM"; exit 12 ;;
  esac
  if (( waited >= MAX_WAIT_SEC )); then
    echo "[$(date +%T)] ABORT: snapshot still RUNNING after ${MAX_WAIT_SEC}s"; exit 13
  fi
  sleep 60; waited=$((waited+60))
done

# Sanity: the snapshot must actually contain data, not just have exited 0.
for f in uniprot.jsonl.gz interpro.jsonl.gz sifts.jsonl.gz; do
  p="data/annotations_v26/$f"
  [[ -s "$p" ]] || { echo "ABORT: missing/empty $p"; exit 14; }
  echo "  $p  $(du -h "$p" | cut -f1)"
done

echo "[$(date +%T)] === step 2/3: build_canonical_targets.py ==="
"$PY" scripts/v26/build_canonical_targets.py

echo "[$(date +%T)] === step 3/3: build_dbd_spans.py ==="
"$PY" scripts/v26/build_dbd_spans.py

echo "[$(date +%T)] === verifying outputs ==="
for f in data/processed/v26/accessions.parquet \
         data/processed/v26/domains.parquet \
         data/processed/v26/sifts_mappings.parquet \
         data/processed/v26/row_resolution.parquet \
         data/processed/v26/dbd_candidates.parquet; do
  [[ -s "$f" ]] || { echo "ABORT: expected output missing: $f"; exit 15; }
  echo "  ok  $f"
done

echo "[$(date +%T)] PHASE 1 (steps 1-3) COMPLETE"
