#!/usr/bin/env bash
# Self-healing detached Boltz-2 foldability campaign (v24 vs DeepPBS).
# Ensures the CCD mols cache is complete (re-download if truncated), then folds
# all 82 complexes across GPUs. Survives session close (launch via setsid).
set -uo pipefail
CACHE=/data1/leihuang/.cache/boltz
MOL_URL=https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar
ROOT=/data1/leihuang/TFScope_store/boltz_v24

echo "[prep] $(date) verifying CCD mols cache"
if ! tar tf "$CACHE/mols.tar" >/dev/null 2>&1; then
  echo "[prep] mols.tar truncated -> clean re-download"
  curl -fL --retry 8 --retry-all-errors -C - -o "$CACHE/mols.tar" "$MOL_URL" || \
  curl -fL --retry 8 -o "$CACHE/mols.tar" "$MOL_URL"
fi
if [ ! -f "$CACHE/mols/ALA.pkl" ] || ! tar tf "$CACHE/mols.tar" >/dev/null 2>&1; then
  echo "[prep] (re)extracting mols.tar"
  rm -rf "$CACHE/mols"; mkdir -p "$CACHE/mols"
  # if tar still truncated, re-download once more fresh then extract
  tar tf "$CACHE/mols.tar" >/dev/null 2>&1 || curl -fL --retry 8 -o "$CACHE/mols.tar" "$MOL_URL"
  tar xf "$CACHE/mols.tar" -C "$CACHE"
fi
if [ ! -f "$CACHE/mols/ALA.pkl" ]; then echo "[prep] FATAL: mols still incomplete"; exit 1; fi
echo "[prep] mols OK ($(ls "$CACHE/mols" | wc -l) components) $(date)"

bash /afs/csail.mit.edu/u/l/leihuang/project/TFScope/iclr/run_boltz_foldability.sh
