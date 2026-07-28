#!/usr/bin/env python
"""Audit: which structure-set genes are GENUINE two-chain-repeat binders, and
can we extract their partner chain's DBD sequence from the cached mmCIF?

Motivation: v20 predicts a ~constant ~10 bp footprint, so it truncates long
motifs. For true multimeric-repeat binders (NR direct/inverted repeats, p53,
bZIP/bHLH dimers) the fix is a trained TWO-CHAIN input -- but only if (a) the
gene really needs two chains (the `is_dimer` flag is over-inclusive: ETS/FOX
are flagged True but bind short single-domain sites) and (b) the partner chain
is actually present in the structure so we can feed it.

This audits both before committing to a retrain. No model is loaded.

Outputs results/audit/dimer_partner_audit.csv and prints a summary.
"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd

sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("bd", "scripts/build_deeppbs_structural_v2.py")
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

STR = "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"
AUG = "data/processed/tf_pwm_aug_dbd_canon_trim_v2.parquet"
SPLIT = "data/processed/splits/train_v2/split.json"
TRAIN = "data/processed/tf_pwm_training_v2.parquet"
CIF = "data/raw/pdb_cif_cache"

# families whose biology is a genuine two-half-site / multimer readout
TWO_CHAIN_FAMS = {"Nuclear_Receptor", "bZIP", "bHLH"}
P53_GENES = {"TP53", "TP63", "TP73", "P53"}


def main():
    st = pd.read_parquet(STR)
    aug = pd.read_parquet(AUG)
    # real family per gene (from the sequence-only table, which carries family_name)
    gene_fam = {g: sub["family_name"].mode().iat[0]
                for g, sub in aug.groupby(aug["gene_symbol"].str.upper())}

    # which genes are in the held-out test split
    sp = json.load(open(SPLIT)); tr = pd.read_parquet(TRAIN)
    fn2gene = dict(zip(tr["filename"], tr["gene_symbol"].str.upper()))
    test_genes = {fn2gene[f] for f in sp["test"] if f in fn2gene}

    st["G"] = st["gene"].str.upper()
    rows = []
    for _, r in st.iterrows():
        g = r["G"]
        fam = gene_fam.get(g, "Other")
        partners = list(r["partner_genes"]) if r["partner_genes"] is not None else []
        real_partners = [p for p in partners if p and str(p).upper() != g]
        het = len(real_partners) > 0
        # genuine two-chain need?
        genuine = (fam in TWO_CHAIN_FAMS) or (g in P53_GENES) or het
        rows.append(dict(gene=g, fam=fam, pdb=r["pdb_id"], chain=r["chain_id"],
                         is_dimer=bool(r["is_dimer"]),
                         partner_chains=list(r["partner_chains"]) if r["partner_chains"] is not None else [],
                         partner_genes=partners, heterodimer=het,
                         motif_len=int(r["motif_length"]), genuine=genuine,
                         in_test=g in test_genes))
    A = pd.DataFrame(rows)

    print("=" * 70)
    print(f"structure set: {len(A)} rows, {A.gene.nunique()} genes")
    print(f"  is_dimer=True flag                 : {A.is_dimer.sum():4d} rows")
    print(f"  GENUINE two-chain-repeat binder    : {A.genuine.sum():4d} rows, "
          f"{A[A.genuine].gene.nunique()} genes")
    print(f"    of which heterodimer (real partner): {A.heterodimer.sum():4d} rows")
    print(f"  is_dimer=True but NOT genuine (noise): {(A.is_dimer & ~A.genuine).sum():4d} rows "
          f"({A[A.is_dimer & ~A.genuine].fam.value_counts().to_dict()})")

    print("\n--- genuine two-chain binders by family ---")
    print(A[A.genuine].groupby("fam").agg(
        rows=("gene", "size"), genes=("gene", "nunique"),
        mean_motif=("motif_len", "mean"), long_ge13=("motif_len", lambda s: (s >= 13).sum())
    ).to_string())

    # PRIORITY = genuine AND long motif (>=13): where truncation actually hurts
    prio = A[A.genuine & (A.motif_len >= 13)].copy()
    print(f"\n--- PRIORITY (genuine & motif>=13): {len(prio)} rows, {prio.gene.nunique()} genes "
          f"| in test: {prio[prio.in_test].gene.nunique()} genes ---")

    # can we extract the partner chain DBD from the cached cif?
    print("\n--- partner-chain extractability (priority rows) ---")
    ok = miss_cif = no_partner_chain = crop_fail = 0
    ext_rows = []
    cache = {}
    for _, r in prio.iterrows():
        pdb = str(r["pdb"]).lower()
        path = os.path.join(CIF, f"{pdb}.cif")
        if not os.path.exists(path):
            miss_cif += 1; ext_rows.append((r["gene"], r["pdb"], "no_cif", None)); continue
        try:
            if pdb not in cache:
                cache[pdb] = bd.load_chains(path)
            prot, dna, dna_by_chain = cache[pdb]
        except Exception as e:
            miss_cif += 1; ext_rows.append((r["gene"], r["pdb"], f"parse_err", None)); continue
        pchains = r["partner_chains"]
        if not pchains:
            no_partner_chain += 1; ext_rows.append((r["gene"], r["pdb"], "no_partner_chain", None)); continue
        got = None
        for pc in pchains:
            if pc in prot:
                crop = bd.contiguous_dbd_crop(prot[pc], dna)
                if crop is not None:
                    got = crop[0]; break
        if got:
            ok += 1; ext_rows.append((r["gene"], r["pdb"], "OK", len(got)))
        else:
            crop_fail += 1; ext_rows.append((r["gene"], r["pdb"], "crop_fail", None))
    tot = len(prio)
    print(f"  partner DBD extracted OK : {ok}/{tot}  ({100*ok/max(tot,1):.0f}%)")
    print(f"  cif missing / parse err  : {miss_cif}")
    print(f"  no partner chain listed  : {no_partner_chain}")
    print(f"  partner chain not croppable to DBD : {crop_fail}")

    # per-gene priority summary with test membership
    print("\n--- priority genes (partner-extract status shown) ---")
    ext_df = pd.DataFrame(ext_rows, columns=["gene", "pdb", "status", "partner_len"])
    g_ok = ext_df[ext_df.status == "OK"].groupby("gene").size()
    for g, sub in prio.groupby("gene"):
        n_ok = int(g_ok.get(g, 0))
        tag = "TEST" if sub.in_test.any() else "train"
        print(f"  {g:14s} [{sub.fam.iloc[0]:16s}] {tag:5s} rows={len(sub):2d} "
              f"motif~{sub.motif_len.mean():.0f} het={sub.heterodimer.any()} "
              f"partner_extractable={n_ok}/{len(sub)}")

    os.makedirs("results/audit", exist_ok=True)
    A.to_csv("results/audit/dimer_partner_audit.csv", index=False)
    ext_df.to_csv("results/audit/priority_partner_extract.csv", index=False)
    print("\nsaved results/audit/dimer_partner_audit.csv + priority_partner_extract.csv")


if __name__ == "__main__":
    main()
