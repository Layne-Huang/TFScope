#!/usr/bin/env python
"""Reverse-compat: express the v26 dataset in v23 schema so v24's trainer can consume it.

The mirror image of build_v23compat_dataset.py. Together the two isolate architecture from data
in both directions:

    v26 model on v24 data  ->  already done: 0.3641 vs v24's 0.4290 cov_r  (v26 architecture loses)
    v24 model on v26 data  ->  THIS SCRIPT                                  (does v24 still win?)

If v24 also collapses on v26 inputs, the problem is the v26 DATA (flanks, partner packing, the
harder clean split). If v24 holds up, the problem is confirmed to be the v26 ARCHITECTURE.

Inputs are v26's, exactly as v26 sees them -- flank20 sequences with an interior DBD
(dbd_start > 0, which is what v24's dbd_mask was always designed for but never actually got,
since every v23 row had dbd_start = 0) plus partner chains.

  python scripts/v26/build_v24compat_dataset.py --dataset flank20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

OUTD = "data/processed/v26"
SPD = "data/processed/splits/v26"

# v24's canonical 10-family scheme (build_training_table.py FID)
FID = {"C2H2_short": 0, "C2H2_medium": 1, "C2H2_long": 2, "bHLH": 3, "Homeodomain": 4,
       "bZIP": 5, "Nuclear_Receptor": 6, "Forkhead": 7, "ETS": 8, "Other": 9}


def map_family(s):
    for f in str(s).split(";"):
        if f in FID:
            return FID[f], f
        if f.startswith("C2H2"):
            return FID["C2H2_long"], "C2H2_long"
        if f == "Homeo_prospero":
            return FID["Homeodomain"], "Homeodomain"
    return FID["Other"], "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="flank20",
                    help="v26 dataset to convert (flank20 = flanks+pairs, as v26 sees it)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"v24compat_{a.dataset}"
    os.makedirs(OUTD, exist_ok=True); os.makedirs(SPD, exist_ok=True)

    ex = pd.read_parquet(f"{OUTD}/v26_{a.dataset}.parquet")
    man = pd.read_parquet(f"{SPD}/manifest.parquet")[
        ["target_unit_id", "split", "application_holdout"]].drop_duplicates("target_unit_id")
    ex = ex.merge(man, on="target_unit_id", how="inner")
    ex = ex[~ex.application_holdout].reset_index(drop=True)   # locked sets stay locked

    rows, split_of = [], {}
    for i, r in enumerate(ex.itertuples()):
        fid, fname = map_family(r.dbd_families_for_analysis_only)
        partners = [str(p["sequence"]) for p in json.loads(r.partner_entities or "[]")]
        fn = f"v26_{i}"
        split_of[fn] = r.split
        rows.append({
            "filename": fn,
            "gene_symbol": str(r.primary_gene_symbol_legacy),
            "sequence": str(r.sequence),
            "pwm": r.pwm,
            "motif_length": int(r.motif_length),
            "seq_length": len(str(r.sequence)),
            # THE POINT: dbd_start > 0 here. Every v23 row had dbd_start=0, so v24's dbd_mask was
            # degenerate in training (audit Finding, docs/v26_audit.md §3). Here it is informative.
            "dbd_start": int(r.dbd_start),
            "dbd_end": int(r.dbd_end),
            "family_id": fid,
            "family_name": fname,
            "family_source": "v26_analysis_only",
            "motif_source": str(r.motif_source),
            "partner_sequence": partners[0] if partners else "",
            "partner_gene": "",
            "is_dimer": bool(len(partners) > 0),
            "_set": "str" if r.structure_id else "seq",
            "gene_key": str(r.primary_gene_symbol_legacy),
            "group_id": f"{r.primary_gene_symbol_legacy}|{r.primary_sequence_hash}|{r.motif_source}",
            "multichain_eligible": bool(len(partners) > 0),
            "partner_seqs": np.array(partners, dtype=object) if partners else None,
            "n_chains": 1 + len(partners),
        })
    df = pd.DataFrame(rows)
    p = f"{OUTD}/{tag}.parquet"
    df.to_parquet(p, index=False)

    sp = {k: [] for k in ["train", "val", "test", "excluded"]}
    for fn, s in split_of.items():
        sp.setdefault(s, []).append(fn)
    sj = f"{SPD}/split_{tag}.json"
    json.dump(sp, open(sj, "w"))

    L = df.sequence.str.len()
    span = df.dbd_end - df.dbd_start
    print(f"rows: {len(df)}   genes: {df.gene_symbol.nunique()}")
    print(f"split: { {k: len(v) for k, v in sp.items() if v} }")
    print(f"seq len: median {int(L.median())}  DBD span median {int(span.median())}")
    print(f"dbd_start > 0: {100*(df.dbd_start > 0).mean():.1f}%   "
          f"(v23 was 0% -- v24's dbd_mask was degenerate there)")
    print(f"rows with partners: {int(df.is_dimer.sum())}  "
          f"n_chains: {df.n_chains.value_counts().to_dict()}")
    print(f"family_id spread: {df.family_id.value_counts().sort_index().to_dict()}")
    print(f"\nwrote {p}\nwrote {sj}")


if __name__ == "__main__":
    main()
