"""Extract TRUE base-contacting (recognition) residues from the test-set crystal
structures, per DBD-sequence index — the structural ground truth for Fig 2b.

A protein residue is a base-readout residue if any of its side-chain (+CA) heavy
atoms lies within CUTOFF of a DNA *base* atom (ring/edge atoms only; sugar-phosphate
backbone excluded). Residues are mapped onto the DBD-canon-trim sequence used by the
alanine scan by aligning the PDB chain sequence to the parquet sequence.

Output: data/contact_maps/true_recognition_residues.json  { base_key: [dbd_idx,...] }
PDBs: /data1/leihuang/TFlow/data/TF_split_index/<PDBID>_0_<chain>_WITH_*.pdb
"""
import os, re, json, glob
import numpy as np
import pandas as pd
import difflib

PDBDIR = "/data1/leihuang/TFlow/data/TF_split_index"
PARQ = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
OUT = "data/contact_maps/true_recognition_residues.json"
CUTOFF = 4.5

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
       "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
       "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
DNA_RES = {"DA","DC","DG","DT","A","C","G","T"}
BACKBONE = {"N","C","O"}                              # protein main-chain to drop (keep CA)
BASE_ATOMS = {"N1","C2","N3","C4","C5","C6","N7","C8","N9",
              "O2","N2","O4","N4","O6","N6","C7","C5M","C5A"}

def parse_pdb(path, prot_chain):
    """Return (prot_seq, prot_res_atoms[list of (reskey, [coords])], base_coords ndarray)."""
    prot = {}          # reskey -> list of side-chain/CA coords
    order = []         # reskey order
    seq = []
    base = []
    seen = set()
    for line in open(path):
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        atom = line[12:16].strip()
        if atom.startswith("H"):
            continue
        resn = line[17:20].strip()
        chain = line[21]
        resseq = line[22:27]                          # resseq + icode
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        if chain == prot_chain and resn in AA3:
            key = (chain, resseq)
            if key not in seen:
                seen.add(key); order.append(key); seq.append(AA3[resn])
                prot[key] = []
            if atom not in BACKBONE:                  # keep CA + side chain
                prot[key].append(xyz)
        elif resn in DNA_RES and atom in BASE_ATOMS:
            base.append(xyz)
    base = np.array(base) if base else np.zeros((0, 3))
    return "".join(seq), order, prot, base

def contacts(order, prot, base):
    """Indices (in chain order) of protein residues contacting a DNA base atom."""
    hit = []
    if len(base) == 0:
        return hit
    for i, key in enumerate(order):
        ats = prot[key]
        if not ats:
            continue
        d = np.sqrt(((np.array(ats)[:, None, :] - base[None]) ** 2).sum(-1))
        if d.min() < CUTOFF:
            hit.append(i)
    return hit

def map_to_dbd(pdb_seq, pdb_hits, dbd_seq):
    """Map PDB-chain hit indices onto dbd_seq indices via best alignment."""
    sm = difflib.SequenceMatcher(None, pdb_seq, dbd_seq, autojunk=False)
    p2d = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            p2d[a + k] = b + k
    return sorted({p2d[h] for h in pdb_hits if h in p2d and 0 <= p2d[h] < len(dbd_seq)})

def find_pdb(pdbid, chain):
    g = glob.glob(f"{PDBDIR}/{pdbid.upper()}_*_{chain.upper()}_WITH_*.pdb")
    return g[0] if g else None

test = set(json.load(open(SPLIT))["test"])
df = pd.read_parquet(PARQ)
df = df[df.filename.astype(str).isin(test)].reset_index(drop=True)
base_key = lambda fn: re.sub(r"\.txt(__os\d+)?$", "", str(fn))

out, ok, miss = {}, 0, []
for r in df.itertuples():
    m = re.match(r"([0-9a-zA-Z]{4})_([A-Za-z0-9])_", str(r.filename))
    if not m:
        miss.append(str(r.filename)); continue
    pdbid, chain = m.group(1), m.group(2)
    path = find_pdb(pdbid, chain)
    if not path:
        miss.append(f"{pdbid}_{chain} (no pdb)"); continue
    pseq, order, prot, base = parse_pdb(path, chain)
    if not pseq or len(base) == 0:
        miss.append(f"{pdbid}_{chain} (no prot/dna)"); continue
    hits = contacts(order, prot, base)
    dbd_hits = map_to_dbd(pseq, hits, r.sequence)
    out[base_key(r.filename)] = dbd_hits
    ok += 1
    print(f"{r.gene_symbol:12s} {pdbid}_{chain}  pdbLen={len(pseq):3d} dbdLen={len(r.sequence):3d} "
          f"contacts={len(hits):2d} mapped={len(dbd_hits):2d}", flush=True)

json.dump(out, open(OUT, "w"))
print(f"\nmapped {ok}/{len(df)} test entries ; missing {len(miss)}")
for m in miss[:20]: print("  miss:", m)
print("saved", OUT)
