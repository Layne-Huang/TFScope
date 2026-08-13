"""Fig 2b ground truth via DSSR (as in TFlow/DeepPBS): all nt:aa base H-bonds.

For each cluster40 test structure, run DSSR (--json --non-pair, X3DNA env set) using the
TFlow dssr_hbonds pipeline, keep nt:aa hydrogen bonds whose DNA partner atom is a BASE
atom (major + minor groove; sugar-phosphate backbone excluded), and map the protein
residues onto the DBD sequence index used by the alanine scan.

Output: data/contact_maps/dssr_recognition_residues.json  { base_key: [dbd_idx,...] }
        data/contact_maps/dssr_majorgroove_residues.json   (subset, for reference)
"""
import os, re, sys, json, glob
import pandas as pd
sys.path.insert(0, "/afs/csail.mit.edu/u/l/leihuang/project/TFlow_1/multiflow/evaluation")
import dssr_hbonds as H

PARQ = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
PDBDIR = "/data1/leihuang/TFlow/data/TF_split_index"
AA3 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL MSE".split())

def pdb_order(pdbf, chain):
    seen = []; s = set()
    for line in open(pdbf):
        if line[:6] not in ("ATOM  ", "HETATM") or line[21] != chain: continue
        if line[17:20].strip() not in AA3: continue
        rs = int(line[22:26])
        if rs not in s: s.add(rs); seen.append(rs)
    return {r: i for i, r in enumerate(seen)}

def aa_resnum(reskey):           # 'A.ARG15' -> ('A', 15)
    ch, rr = reskey.split(".")
    return ch, int(re.match(r"[A-Za-z]+(\d+)", rr).group(1))

test = set(json.load(open(SPLIT))["test"])
df = pd.read_parquet(PARQ); df = df[df.filename.astype(str).isin(test)].reset_index(drop=True)
base_key = lambda fn: re.sub(r"\.txt(__os\d+)?$", "", str(fn))

out_all, out_mg, miss = {}, {}, []
for r in df.itertuples():
    m = re.match(r"([0-9a-zA-Z]{4})_([A-Za-z0-9])_", str(r.filename))
    if not m: miss.append(str(r.filename)); continue
    pdbid, chain = m.group(1), m.group(2)
    g = glob.glob(f"{PDBDIR}/{pdbid.upper()}_*_{chain}_WITH_*.pdb")
    if not g: miss.append(f"{pdbid}_{chain}"); continue
    pdbf = g[0]
    try:
        res = H.classify_hbonds(H.run_dssr(pdbf))
    except Exception as e:
        miss.append(f"{pdbid}_{chain} (dssr {e})"); continue
    r2i = pdb_order(pdbf, chain)
    # all nt:aa BASE H-bonds = total nt:aa minus those to DNA backbone atoms
    def idxset(hblist):
        s = set()
        for hb in hblist:
            ch, rn = aa_resnum(hb["aa_residue"])
            if ch == chain and rn in r2i: s.add(r2i[rn])
        return sorted(s)
    base_hb = [hb for hb in res["total_hbonds"] if hb["dna_atom"] not in H.DNA_BACKBONE_ATOMS]
    out_all[base_key(r.filename)] = idxset(base_hb)
    out_mg[base_key(r.filename)] = idxset(res["major_groove_hbonds"])
    print(f"{r.gene_symbol:12s} {pdbid}_{chain}  base-Hbond={len(out_all[base_key(r.filename)]):2d} "
          f"major-groove={len(out_mg[base_key(r.filename)]):2d}", flush=True)

json.dump(out_all, open("data/contact_maps/dssr_recognition_residues.json", "w"))
json.dump(out_mg, open("data/contact_maps/dssr_majorgroove_residues.json", "w"))
print(f"\nmapped {len(out_all)}/{len(df)} ; missing {len(miss)}: {miss[:8]}")
print("saved data/contact_maps/dssr_recognition_residues.json (+majorgroove)")
