#!/usr/bin/env python
"""Build the N-chain (order-aware) training table v23 from v22.

For every structure row, recover ALL protomers on the same primary DNA duplex
(find_dimer_partners' shared-duplex logic -- captures trimers/tetramers, not
just the single best dimer partner), store them as an ordered `partner_seqs`
list (cap 3 partners => tetramer). p53/HSF/NF-Y/IRF thus get their full
multimer, not a truncated dimer. Sequence-set partners (from v22's single
partner_sequence) are carried over as a 1-element list.

Also fixes taxonomy: POU -> Homeodomain family_id (POU contains a homeodomain
and Homeodomain has 1235 train rows, vs the inert Other=9). p53 name kept.

Outputs data/processed/tf_pwm_training_v23.parquet
"""
import os, sys, importlib.util, time
import numpy as np, pandas as pd

sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("bd", "scripts/build_deeppbs_structural_v2.py")
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

V22 = "data/processed/tf_pwm_training_v22.parquet"
STR = "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"
CIF = "data/raw/pdb_cif_cache"
OUT = "data/processed/tf_pwm_training_v23.parquet"
MAX_PARTNERS = 3          # + self = 4 chains (tetramer cap)
MAX_PLEN = 150            # per-partner DBD crop length cap
HOMEODOMAIN_ID = 4
POU = {"POU1F1","POU2F1","POU2F2","POU2F3","POU3F1","POU3F2","POU5F1","POU6F1","POU5F1::SOX2"}


def main():
    v = pd.read_parquet(V22).reset_index(drop=True)
    st = pd.read_parquet(STR); st["G"] = st.gene.str.upper()
    # (gene, sequence) -> list of (pdb, chain)
    key2 = {}
    for _, r in st.iterrows():
        key2.setdefault((r.G, r.sequence), []).append((r.pdb_id, r.chain_id))

    partner_seqs = [[] for _ in range(len(v))]
    order = [1] * len(v)
    vs = v[v.filename.str.startswith("str_")]
    # group work by pdb (load + find_dimer_partners once per structure)
    by_pdb = {}
    for i, r in vs.iterrows():
        hits = key2.get((str(r.gene_symbol).upper(), r.sequence), [])
        if hits:
            pdb, chain = hits[0]
            by_pdb.setdefault(pdb, []).append((i, chain))
    print(f"[v23] {len(vs)} structure rows across {len(by_pdb)} pdbs", flush=True)

    ok = 0; t0 = time.time()
    for n, (pdb, rows) in enumerate(by_pdb.items(), 1):
        if n % 100 == 0 or n == len(by_pdb):
            print(f"  [{n}/{len(by_pdb)}] ok={ok} {time.time()-t0:.0f}s", flush=True)
        path = os.path.join(CIF, f"{str(pdb).lower()}.cif")
        if not os.path.exists(path):
            continue
        try:
            prot, dna, dna_by_chain = bd.load_chains(path)
            crops, cdc, _ = bd.find_dimer_partners(prot, dna, dna_by_chain)
        except Exception:
            continue
        for i, chain in rows:
            my_duplex = cdc.get(chain, set())
            if not my_duplex:
                continue
            # all OTHER chains sharing this chain's primary DNA duplex = protomers
            prot_chains = [c for c in crops
                           if c != chain and (cdc.get(c, set()) & my_duplex)]
            # order by crop length (strongest/most-complete first), cap + length-cap
            seqs = sorted((crops[c][0] for c in prot_chains), key=len, reverse=True)
            seqs = [s for s in seqs if 0 < len(s) <= MAX_PLEN][:MAX_PARTNERS]
            if seqs:
                partner_seqs[i] = seqs
                order[i] = 1 + len(seqs)
                ok += 1

    # carry over seq-set single partners (from v22 partner_sequence) as 1-lists
    seq_mask = v.filename.str.startswith("seq_") & (v["partner_sequence"].str.len() > 0)
    for i in v.index[seq_mask]:
        ps = str(v.at[i, "partner_sequence"])
        if 0 < len(ps) <= MAX_PLEN:
            partner_seqs[i] = [ps]; order[i] = 2

    v["partner_seqs"] = partner_seqs
    v["n_chains"] = order
    # keep the legacy single partner_sequence = first protomer (back-compat)
    v["partner_sequence"] = [s[0] if s else "" for s in partner_seqs]

    # taxonomy: POU -> Homeodomain id
    pou = v.gene_symbol.str.upper().isin(POU)
    v.loc[pou, "family_id"] = HOMEODOMAIN_ID
    if "family_source" in v.columns:
        v.loc[pou, "family_source"] = "curated_POU->Homeodomain"

    nmulti = (v["n_chains"] >= 3).sum()
    print(f"\n[v23] partner_seqs set on {(v['partner_seqs'].map(len)>0).sum()} rows "
          f"| >=3 chains (trimer+): {nmulti} rows")
    print("  n_chains distribution:", v["n_chains"].value_counts().sort_index().to_dict())
    print("  POU rows remapped to Homeodomain:", int(pou.sum()))
    v.to_parquet(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
