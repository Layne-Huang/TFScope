#!/usr/bin/env python
"""Batch-build the contact-distillation teacher set from the local crystal_pdbs/ co-crystals.

For every plain (full-duplex) co-crystal PDB: compute the residue->base contact map + per-base
soft targets, map the TF to its rebin family, and flag TFs that leak into TFScope's TEST split.
Emits one contacts npz per structure + a manifest. (Alignment of these contacts to the model's
attention index space is the downstream step; this produces the raw, validated teacher targets.)
"""
import os, sys, re, glob, json
import numpy as np, pandas as pd
sys.path.insert(0, "scripts/contact_teacher")
from extract_pdb_contacts import parse_pdb, contact_map, base_residue_targets

PDB_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/crystal_pdbs"
OUT = "results/contact_teacher/teacher_set"
REBIN = "data/processed/tf_pwm_aug_dbd_canon_trim_rebin34.parquet"
SPLIT = "data/processed/splits/cluster40/split.json"

def parse_name(b):
    """1a1g_A_Egr1.MA0162.1.pdb -> (prot_chain 'A', TF 'EGR1')."""
    m = re.match(r'^[0-9a-z]{4}_([A-Za-z0-9]+)_([A-Za-z0-9:]+)\.', b)
    if not m:
        return None, None
    return m.group(1), m.group(2).upper()

def min_chain_dist(prot_res, dna_res):
    pa = np.concatenate([p["xyz"] for p in prot_res], 0)
    da = np.concatenate([d["xyz"] for d in dna_res], 0)
    # block to avoid a huge (Np*Nd) allocation
    best = np.inf
    for i in range(0, len(pa), 2000):
        chunk = pa[i:i+2000]
        d = np.sqrt(((chunk[:, None, :] - da[None, :, :]) ** 2).sum(-1)).min()
        best = min(best, float(d))
    return best

def select_chains(prot, dna, prot_chain):
    """Keep only the named protein chain + the 2 DNA strands of its bound duplex
    (the 2 DNA chains closest to that protein chain)."""
    p = [r for r in prot if r["chain"] == prot_chain]
    if not p:                                   # fallback: chain id absent -> use all protein
        p = prot
    # group DNA by chain, rank chains by proximity to the protein chain, keep top 2
    dna_by_ch = {}
    for r in dna:
        dna_by_ch.setdefault(r["chain"], []).append(r)
    if len(dna_by_ch) <= 2:
        d = dna
    else:
        ranked = sorted(dna_by_ch.items(), key=lambda kv: min_chain_dist(p, kv[1]))
        keep = {ranked[0][0], ranked[1][0]}
        d = [r for r in dna if r["chain"] in keep]
    return p, d

def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(REBIN)
    gene2fam = dict(zip(df['gene_symbol'].astype(str).str.upper(), df['family_name_rebin']))
    sp = json.load(open(SPLIT))
    test_genes = set(df[df['filename'].isin(set(sp.get('test', [])))]['gene_symbol'].astype(str).str.upper())

    files = sorted(f for f in glob.glob(f"{PDB_DIR}/*.pdb") if "renamed_pwm_hybrid" not in f)
    manifest = []
    ok = fail = 0
    for f in files:
        b = os.path.basename(f)
        prot_chain, tf = parse_name(b)
        prot, dna = parse_pdb(f)
        if not prot or not dna:
            fail += 1; continue
        prot, dna = select_chains(prot, dna, prot_chain)   # named protein chain + its 2 DNA strands
        if not prot or not dna:
            fail += 1; continue
        D, pl, dl = contact_map(prot, dna)
        T, contacts = base_residue_targets(D)
        if not contacts:
            fail += 1; continue
        gene = tf.split("::")[0] if tf else None          # heterodimer -> first protomer
        fam = gene2fam.get(gene)
        leak = gene in test_genes
        np.savez(os.path.join(OUT, b.replace(".pdb", "_contacts.npz")),
                 D=D, T=T, prot_labels=np.array(pl), dna_labels=np.array(dl))
        manifest.append(dict(pdb=b, tf=tf, gene=gene, family=fam, test_leakage=bool(leak),
                             n_prot=len(pl), n_dna=len(dl), n_contacting_bases=len(contacts)))
        ok += 1
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)

    mdf = pd.DataFrame(manifest)
    print(f"extracted contacts: {ok} structures ok, {fail} skipped")
    print(f"unique TFs: {mdf['tf'].nunique()}  | mapped to a rebin family: {mdf['family'].notna().sum()}")
    print(f"TEST-LEAKAGE structures (exclude from training teacher): {mdf['test_leakage'].sum()}")
    print("\n=== teacher coverage per rebin family ===")
    cov = mdf.dropna(subset=['family']).groupby('family')['gene'].nunique().sort_values(ascending=False)
    for fam, n in cov.items():
        print(f"  {fam:18s} {n} genes")
    unmapped = sorted(mdf[mdf['family'].isna()]['tf'].dropna().unique())
    if unmapped:
        print("\nunmapped TFs (not in taxonomy):", ", ".join(unmapped))
    print(f"\nsaved -> {OUT}/  (contacts npz + manifest.json)")

if __name__ == "__main__":
    main()
