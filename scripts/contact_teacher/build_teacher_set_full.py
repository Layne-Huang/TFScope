#!/usr/bin/env python
"""Build the FULL contact-distillation teacher set from the entire DeepPBS co-crystal set.

Manifest of (PDB, protein-chain, TF, motif) comes from DeepPBS's assembly2024 npz filenames
(566 unique PDB IDs). Coordinates come from the cached mmCIF in .cache/pdb (gemmi parses both
cif and pdb); missing ones are reported for optional download. For EACH protein chain we pair it
with its 2 nearest DNA strands (the bound duplex) and compute the residue->base contact map +
per-base soft targets. TF -> rebin family; TFs in TFScope's TEST split are flagged as leakage.

Run from repo root in the `tfscope` env (needs gemmi).
"""
import os, sys, re, glob, json, urllib.request
import numpy as np, pandas as pd
import gemmi

CACHE = "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/pdb"
ASM = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/deeppbsmar24/data/assembly2024"
CRYSTAL = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/crystal_pdbs"
OUT = "results/contact_teacher/teacher_set_full"
REBIN = "data/processed/tf_pwm_aug_dbd_canon_trim_rebin34.parquet"
SPLIT = "data/processed/splits/cluster40/split.json"
THRESH, TAU = 4.5, 2.0
AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
       "MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
DNA = {"DA","DC","DG","DT"}

_SPECIES = ("HUMAN", "MOUSE", "RAT", "BOVIN", "DROME", "YEAST", "CHICK", "XENLA", "PIG")

def pdb_to_tfs():
    """{pdb_id: {chain: GENE}} and {pdb_id: set(GENE)} — gene parsed from the HOCOMOCO-named
    npz (`<PDB>_<chain>_<GENE>_<SPECIES>.H1xMO...`). JASPAR-only names give only a matrix id and
    are skipped for gene identity. crystal_pdbs filenames supplement (clean gene names)."""
    chain_tf, pdb_tf = {}, {}
    def add(pid, ch, gene):
        chain_tf.setdefault(pid, {}).setdefault(ch, gene)   # keep first (HOCOMOCO) seen
        pdb_tf.setdefault(pid, set()).add(gene)
    for f in glob.glob(f"{ASM}/*.npz"):
        b = os.path.basename(f)
        m = re.match(r'^([0-9a-z]{4})_([A-Za-z0-9]+)_(.+?)\.(H1\dMO|H13)', b)   # HOCOMOCO only
        if not m:
            continue
        pid, ch, tf = m.group(1).lower(), m.group(2), m.group(3).upper()
        gene = re.sub(r'_(' + "|".join(_SPECIES) + r')$', '', tf)
        add(pid, ch, gene)
    # supplement with crystal_pdbs clean names
    for f in glob.glob(f"{CRYSTAL}/*.pdb"):
        if "renamed_pwm_hybrid" in f:
            continue
        m = re.match(r'^([0-9a-z]{4})_([A-Za-z0-9]+)_([A-Za-z0-9:]+)\.', os.path.basename(f))
        if m:
            add(m.group(1).lower(), m.group(2), m.group(3).upper().split("::")[0])
    return chain_tf, pdb_tf

def load_chains(path):
    """Return {'prot': {chain: [(resname, Natoms x3)]}, 'dna': {chain: [...]}} heavy atoms only."""
    st = gemmi.read_structure(path)
    st.setup_entities()
    model = st[0]
    prot, dna = {}, {}
    for chain in model:
        for res in chain:
            rn = res.name.strip()
            tgt = prot if rn in AA3 else (dna if rn in DNA else None)
            if tgt is None:
                continue
            xyz = [[a.pos.x, a.pos.y, a.pos.z] for a in res
                   if a.element.name != "H" and a.altloc in ("\x00", "", " ", "A")]
            if xyz:
                tgt.setdefault(chain.name, []).append((rn, np.asarray(xyz)))
    return prot, dna

def chain_coords(res_list):
    return np.concatenate([r[1] for r in res_list], 0)

def min_dist_chains(a_res, b_res):
    a, b = chain_coords(a_res), chain_coords(b_res)
    best = np.inf
    for i in range(0, len(a), 3000):
        c = a[i:i+3000]
        best = min(best, float(np.sqrt(((c[:,None]-b[None])**2).sum(-1)).min()))
    return best

def contacts_for(prot_res, dna_res):
    """residue x base min-heavy-atom distance -> (D, targets T, n_contacting_bases)."""
    Lp, Ld = len(prot_res), len(dna_res)
    D = np.zeros((Lp, Ld))
    for i, (_, pc) in enumerate(prot_res):
        for j, (_, bc) in enumerate(dna_res):
            D[i, j] = np.sqrt(((pc[:,None]-bc[None])**2).sum(-1)).min()
    T = np.zeros((Ld, Lp)); ncb = 0
    for j in range(Ld):
        near = np.where(D[:, j] <= THRESH)[0]
        if len(near):
            w = np.exp(-D[near, j]/TAU); w /= w.sum(); T[j, near] = w; ncb += 1
    return D, T, ncb

def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(REBIN)
    gene2fam = dict(zip(df['gene_symbol'].astype(str).str.upper(), df['family_name_rebin']))
    sp = json.load(open(SPLIT))
    test_genes = set(df[df['filename'].isin(set(sp.get('test', [])))]['gene_symbol'].astype(str).str.upper())

    chain_tf, pdb_tf = pdb_to_tfs()
    cached = {os.path.basename(f)[:4].lower(): f for f in glob.glob(f"{CACHE}/*.cif")}
    pids = sorted(pdb_tf)
    print(f"DeepPBS PDB ids: {len(pids)} | cached cif: {len(cached)} | "
          f"missing (skip for now): {len([p for p in pids if p not in cached])}")

    manifest = []; ok = skip = 0
    for pid in pids:
        path = cached.get(pid)
        if not path:
            skip += 1; continue
        try:
            prot, dna = load_chains(path)
        except Exception as e:
            skip += 1; continue
        if not prot or not dna:
            skip += 1; continue
        dna_items = list(dna.items())
        for pch, pres in prot.items():
            # pair this protein chain with its 2 nearest DNA strands (the bound duplex)
            ranked = sorted(dna_items, key=lambda kv: min_dist_chains(pres, kv[1]))
            keep = ranked[:2]
            if min_dist_chains(pres, keep[0][1]) > 8.0:     # this chain doesn't bind DNA
                continue
            dres = [r for _, rl in keep for r in rl]
            D, T, ncb = contacts_for(pres, dres)
            if ncb == 0:
                continue
            # TF: prefer the assembly2024 chain->TF map; else the structure's single TF
            tf = chain_tf.get(pid, {}).get(pch)
            if tf is None:
                tfs = pdb_tf.get(pid, set())
                tf = next(iter(tfs)) if len(tfs) == 1 else None
            gene = tf.split("::")[0] if tf else None
            fam = gene2fam.get(gene)
            leak = gene in test_genes if gene else False
            name = f"{pid}_{pch}"
            np.savez(os.path.join(OUT, f"{name}_contacts.npz"),
                     D=D, T=T, prot_resnames=np.array([r[0] for r in pres]),
                     dna_resnames=np.array([r[0] for r in dres]))
            manifest.append(dict(name=name, pdb=pid, prot_chain=pch, tf=tf, gene=gene,
                                 family=fam, test_leakage=bool(leak),
                                 n_prot=len(pres), n_dna=len(dres), n_contacting_bases=ncb))
            ok += 1
        if (len(manifest)) and ok % 100 == 0:
            print(f"  ...{ok} chain-teachers so far", flush=True)
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)

    mdf = pd.DataFrame(manifest)
    print(f"\nteachers: {ok} protein-chain contact maps from {mdf['pdb'].nunique()} PDB ids "
          f"({skip} PDBs skipped/missing)")
    print(f"unique TFs: {mdf['tf'].nunique()} | mapped to rebin family: {mdf['family'].notna().sum()} chains")
    print(f"TEST-LEAKAGE chains (exclude from training): {mdf['test_leakage'].sum()} "
          f"({mdf[mdf['test_leakage']]['gene'].nunique()} genes)")
    print("\n=== teacher coverage per rebin family (unique genes) ===")
    cov = mdf.dropna(subset=['family']).drop_duplicates('gene').groupby('family')['gene'].nunique().sort_values(ascending=False)
    for fam, n in cov.items():
        print(f"  {fam:18s} {n}")
    print(f"\nsaved -> {OUT}/ (per-chain contacts npz + manifest.json)")

if __name__ == "__main__":
    main()
