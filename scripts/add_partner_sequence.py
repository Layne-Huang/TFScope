#!/usr/bin/env python
"""Phase 1 of the trained two-chain (heterodimer) path.

For every structure row with a GENUINE heterodimer partner (a partner chain
whose gene differs from the primary gene), extract the partner chain's
DNA-contacting DBD crop from the cached mmCIF (same contiguous_dbd_crop used to
build the primary crops) and store it as a new `partner_sequence` column.

Rows without a real heterodimer partner get partner_sequence="" (single-chain,
unchanged behaviour). Writes a new structure parquet; does NOT touch the merged
training table (build_training_table.py handles that next).

Scope decision (audit_dimer_partners.py): heterodimers only -- the clean,
~100%-extractable set. p53 (homo-tetramer, partial chain records) excluded.

Outputs data/processed/tf_pwm_deeppbs_v2_partner.parquet
"""
import os, sys, importlib.util, time
import numpy as np, pandas as pd

sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("bd", "scripts/build_deeppbs_structural_v2.py")
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

STR = "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"
OUT = "data/processed/tf_pwm_deeppbs_v2_partner.parquet"
CIF = "data/raw/pdb_cif_cache"


def main():
    st = pd.read_parquet(STR)
    st["G"] = st["gene"].str.upper()
    partner_seq = [""] * len(st)
    partner_gene_used = [""] * len(st)
    partner_chain_used = [""] * len(st)

    # rows with a real heterodimer partner (distinct gene + a chain listed)
    cand = []
    for i, r in st.iterrows():
        genes = list(r["partner_genes"]) if r["partner_genes"] is not None else []
        chains = list(r["partner_chains"]) if r["partner_chains"] is not None else []
        pairs = [(c, g) for c, g in zip(chains, genes)
                 if g and str(g).upper() != r["G"]]
        if pairs:
            cand.append((i, r["pdb_id"], pairs))
    print(f"{len(st)} structure rows | {len(cand)} heterodimer rows to process", flush=True)

    cache = {}
    ok = fail_chain = fail_crop = 0
    t0 = time.time()
    for n, (i, pdb, pairs) in enumerate(cand, 1):
        if n % 25 == 0 or n == len(cand):
            el = time.time() - t0
            print(f"  [{n}/{len(cand)}] ok={ok} fail_chain={fail_chain} "
                  f"fail_crop={fail_crop} | {el:.0f}s "
                  f"(~{el/n*len(cand):.0f}s total)", flush=True)
        path = os.path.join(CIF, f"{str(pdb).lower()}.cif")
        if not os.path.exists(path):
            fail_chain += 1; continue
        try:
            if pdb not in cache:
                cache[pdb] = bd.load_chains(path)
            prot, dna, dna_by_chain = cache[pdb]
            crops, _, real_partners = bd.find_dimer_partners(
                prot, dna, dna_by_chain
            )
        except Exception:
            fail_chain += 1; continue
        primary_chain = str(st.loc[i, "chain_id"])
        allowed = set(real_partners.get(primary_chain, []))
        # pick the partner chain that yields the LONGEST DBD crop (most contact)
        best_seq, best_gene, best_chain, best_len = "", "", "", 0
        for pc, pg in pairs:
            if pc not in allowed or pc not in crops:
                continue
            crop = crops[pc]
            if crop is not None and len(crop[0]) > best_len:
                best_seq, best_gene, best_chain, best_len = (
                    crop[0], str(pg).upper(), str(pc), len(crop[0])
                )
        if best_seq:
            partner_seq[i] = best_seq
            partner_gene_used[i] = best_gene
            partner_chain_used[i] = best_chain
            ok += 1
        else:
            fail_crop += 1

    st["partner_sequence"] = partner_seq
    st["partner_gene_used"] = partner_gene_used
    st["partner_chain_used"] = partner_chain_used
    st["partner_crop_method"] = np.where(
        np.asarray(partner_seq, dtype=object) != "",
        "shared_primary_duplex",
        "",
    )
    st = st.drop(columns=["G"])
    has = (st["partner_sequence"].str.len() > 0)
    print(f"\npartner_sequence set on {has.sum()} rows "
          f"({st.loc[has, 'gene'].str.upper().nunique()} primary genes)")
    print("mean partner DBD len:", round(st.loc[has, "partner_sequence"].str.len().mean(), 1))

    # spot-check the key test genes
    print("\n--- spot check (test heterodimer genes) ---")
    for g in ["THRB", "NFE2L2", "ELF1", "POU5F1::SOX2"]:
        sub = st[(st.gene.str.upper() == g) & has]
        if len(sub) == 0:
            print(f"  {g}: (no partner extracted)"); continue
        r = sub.iloc[0]
        print(f"  {g}: partner={r['partner_gene_used']} "
              f"primary_len={len(r['sequence'])} partner_len={len(r['partner_sequence'])}")
        print(f"     primary: {r['sequence'][:70]}")
        print(f"     partner: {r['partner_sequence'][:70]}")

    st.to_parquet(OUT)
    print(f"\nsaved {OUT}  ({len(st)} rows)")


if __name__ == "__main__":
    main()
