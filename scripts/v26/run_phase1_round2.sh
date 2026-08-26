#!/usr/bin/env bash
# Phase-1 round 2: fetch the 163 accessions the round-1 seed missed (SIFTS-discovered
# orthologs/isoforms), then rebuild targets + spans + triage and re-run the invariant tests.
set -euo pipefail
# NOTE: run_detached.sh already cd'd us into the correct root (AFS repo, or the
# /data1 mirror when launched with --mirror). Do NOT hardcode the AFS path here --
# AFS tokens drop every ~10 min on this host and would kill the job mid-run.
: "${PWD:?}"
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python

echo "[$(date +%T)] === round-2 fetch (uniprot + interpro for newly discovered accessions) ==="
"$PY" scripts/v26/fetch_annotation_snapshot.py --round2 --only uniprot
"$PY" scripts/v26/fetch_annotation_snapshot.py --round2 --only interpro

echo "[$(date +%T)] === rebuild canonical targets ==="
"$PY" scripts/v26/build_canonical_targets.py

echo "[$(date +%T)] === rebuild DBD spans (tier1 + tier2 rescue) ==="
"$PY" scripts/v26/build_dbd_spans.py

echo "[$(date +%T)] === re-triage residual ==="
"$PY" scripts/v26/triage_missing_dbd.py

echo "[$(date +%T)] === coverage after round 2 ==="
"$PY" - <<'PYEOF'
import pandas as pd
c=pd.read_parquet('data/processed/v26/dbd_candidates.parquet')
r=pd.read_parquet('data/processed/v26/row_resolution.parquet')
a=pd.read_parquet('data/processed/v26/accessions.parquet')
have=set(c.accession)
n=int(r.primary_accession.isin(have).sum())
print(f"accessions fetched      : {a.accession.nunique()}")
print(f"accessions with a DBD   : {c.accession.nunique()}  ({c.tier.value_counts().to_dict()})")
print(f"v23 rows with a DBD     : {n}/{len(r)}  ({100*n/len(r):.1f}%)")
miss=r[~r.primary_accession.isin(have)]
print(f"rows still without DBD  : {len(miss)}  by method {miss.resolution_method.value_counts().to_dict()}")
nf=miss[~miss.primary_accession.isin(set(a.accession))]
print(f"  still-unfetched rows  : {len(nf)} ({nf.primary_accession.nunique()} accessions)")
PYEOF

echo "[$(date +%T)] === invariant tests ==="
"$PY" tests/v26/test_dbd_spans.py
"$PY" tests/v26/test_legacy_untouched.py

echo "[$(date +%T)] PHASE 1 ROUND 2 COMPLETE"
