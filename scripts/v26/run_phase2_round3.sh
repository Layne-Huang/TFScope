#!/usr/bin/env bash
# Phase-2 round 3: fetch the partner-chain accessions SIFTS knows about but we never pulled,
# then re-map contacts to UniProt. Closes the 766-chain / 10,644-contact partner gap.
set -euo pipefail
: "${PWD:?}"
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python

echo "[$(date +%T)] === fetch partner-chain accessions (uniprot + interpro) ==="
"$PY" scripts/v26/fetch_annotation_snapshot.py --round2 --from-sifts --only uniprot
"$PY" scripts/v26/fetch_annotation_snapshot.py --round2 --from-sifts --only interpro

echo "[$(date +%T)] === rebuild canonical targets (picks up new accessions) ==="
"$PY" scripts/v26/build_canonical_targets.py
echo "[$(date +%T)] === rebuild DBD spans ==="
"$PY" scripts/v26/build_dbd_spans.py
echo "[$(date +%T)] === rebuild datasets ==="
"$PY" scripts/v26/build_v26_datasets.py
echo "[$(date +%T)] === re-map contacts to UniProt ==="
"$PY" scripts/v26/map_contacts_to_uniprot.py

echo "[$(date +%T)] === tests ==="
"$PY" tests/v26/test_dbd_spans.py
"$PY" tests/v26/test_legacy_untouched.py
echo "[$(date +%T)] PHASE 2 ROUND 3 COMPLETE"
