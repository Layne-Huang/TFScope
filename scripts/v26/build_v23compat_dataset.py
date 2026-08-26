#!/usr/bin/env python
"""Build a v26-format dataset from the v23 rows + the v24 train_v22 split.

PURPOSE: the cleanest possible v24-vs-v26 ablation. Same rows, same split, same 291 test rows,
same evaluation -- the ONLY variable left is the model and its loss.

Right now v24 scores 0.5828 cov_r and v26 reg_strong 0.3757, but on different data AND different
splits AND (possibly) different evaluation protocols, so the 0.207 gap is not attributable.
Training v26 on v23+train_v22 removes two of those three confounds at once.

DELIBERATE CAVEAT: train_v22 carries the leakage the whole v26 rebuild exists to remove
(291/291 test rows have structure-defined input boundaries; all 20 Barrera genes are in train).
That is ACCEPTABLE HERE because this is a diagnostic, not a benchmark: the leakage applies
equally to v24 and to v26, so it cancels in the comparison. Numbers produced from this dataset
must never be reported as v26's headline result.

  python scripts/v26/build_v23compat_dataset.py
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

V23 = "data/processed/tf_pwm_training_v23.parquet"
STRUCT = "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
OUTD = "data/processed/v26"
SPD = "data/processed/splits/v26"


def _h(*p):
    return hashlib.sha256("|".join(str(x) for x in p).encode()).hexdigest()[:16]


def main():
    os.makedirs(OUTD, exist_ok=True)
    os.makedirs(SPD, exist_ok=True)
    v = pd.read_parquet(V23).rename(columns={"_set": "set_label"})
    split = json.load(open(SPLIT))
    split_of = {fn: sp for sp, fns in split.items() for fn in fns}
    st = pd.read_parquet(STRUCT).reset_index(drop=True)
    pdb_of = {f"str_{i}": (str(r.pdb_id).upper(), str(r.chain_id))
              for i, r in st.iterrows()}

    rows = []
    for r in v.itertuples():
        fn = str(r.filename)
        sp = split_of.get(fn)
        if sp is None or sp == "excluded":       # 'excluded' rows are unusable by design
            continue
        seq = str(r.sequence)
        # v23 rows are tight DBD crops with dbd_start=0, dbd_end=len -- verified in the audit.
        d0, d1 = int(r.dbd_start), int(r.dbd_end)
        partners = []
        ps = r.partner_seqs
        if ps is not None:
            for j, p in enumerate(list(ps)):
                p = str(p)
                if len(p) >= 10:
                    partners.append({"index": j, "sequence": p, "sequence_hash": _h(p),
                                     "length": len(p), "gene_hint": str(r.partner_gene or "")})
        gene = str(r.gene_symbol).upper()
        # target_unit_id groups by gene+DBD sequence, mirroring the v26 definition as closely as
        # v23 allows (v23 has no UniProt accession, so gene symbol is the available proxy).
        tuid = _h(gene, seq)
        moid = _h(fn, str(r.motif_source))
        pdb, chain = pdb_of.get(fn, (None, None))
        rows.append({
            "example_id": _h(moid, gene, 0, len(seq)),
            "target_unit_id": tuid,
            "motif_observation_id": moid,
            "legacy_filename": fn,
            "legacy_set": str(r.set_label),
            "motif_source": str(r.motif_source),
            "primary_accession": None,                 # v23 has none; unused downstream
            "primary_gene": gene,
            "primary_gene_symbol_legacy": gene,
            "primary_sequence_hash": _h(seq),
            "primary_protein_len": len(seq),
            "organism_taxid": None,
            "dbd_unp_start": d0 + 1, "dbd_unp_end": d1,
            "dbd_families_for_analysis_only": str(r.family_name),   # analysis only, NOT an input
            "dbd_tier": "v23_legacy",
            "dbd_selection_mode": "v23_legacy_crop",
            "n_dbd_candidates": 1,
            "is_composite_dimer_name": "::" in gene,
            "partner_entities": json.dumps(partners),
            "n_partners": len(partners),
            "pwm": r.pwm, "motif_length": int(r.motif_length),
            "structure_id": pdb, "structure_chain": chain,
            "sequence": seq,
            "crop_unp_start": 1, "crop_unp_end": len(seq),
            "dbd_start": d0, "dbd_end": d1,
            "flank_width": 0,
            "split": sp,
        })
    df = pd.DataFrame(rows)
    ds = df.drop(columns=["split"])
    ds.to_parquet(f"{OUTD}/v26_v23compat.parquet", index=False)

    # manifest keyed on target_unit_id, matching what V26Data joins on
    man = df[["target_unit_id", "split"]].copy()
    # a target unit must land in exactly one split; v23's split is row-level, so resolve
    # conflicts by majority (they are rare -- reported below)
    conflict = man.groupby("target_unit_id").split.nunique()
    n_conf = int((conflict > 1).sum())
    man = (man.groupby("target_unit_id").split
           .agg(lambda s: s.value_counts().idxmax()).reset_index())
    man["application_holdout"] = False
    man = man.drop_duplicates("target_unit_id")
    man.to_parquet(f"{SPD}/manifest_v23compat.parquet", index=False)

    print(f"rows: {len(ds)}  target_units: {ds.target_unit_id.nunique()}  "
          f"genes: {ds.primary_gene.nunique()}")
    print("split (rows):", df.split.value_counts().to_dict())
    print("split (target units):",
          df.groupby("split").target_unit_id.nunique().to_dict())
    print(f"target units spanning >1 split (resolved by majority): {n_conf}")
    print(f"test rows: {int((df.split=='test').sum())} (v24 benchmark is 291)")
    print(f"\nwrote {OUTD}/v26_v23compat.parquet")
    print(f"wrote {SPD}/manifest_v23compat.parquet")


if __name__ == "__main__":
    main()
