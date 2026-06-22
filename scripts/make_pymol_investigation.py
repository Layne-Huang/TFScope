"""PyMOL investigation kit for TFScope's important residues on a real complex.
For a TF: (1) write TFScope per-residue importance into the B-factor column of the
PDB (color by `spectrum b` in PyMOL), (2) a residue annotation table (resnum, AA,
importance rank, nearest DNA base + distance + likely interaction type, crystal
contact?), (3) a ready-to-run .pml that colors by importance and labels the top hits.

Usage: python scripts/make_pymol_investigation.py [GENE PDBID CHAIN]
Default: ZBTB7A 7N5V A
"""
import os, sys, re, json, glob
import numpy as np

GENE  = sys.argv[1] if len(sys.argv) > 1 else "ZBTB7A"
PDBID = sys.argv[2] if len(sys.argv) > 2 else "7N5V"
CHAIN = sys.argv[3] if len(sys.argv) > 3 else "A"
TOPN  = 20
PDBDIR = "/data1/leihuang/TFlow/data/TF_split_index"
OUT = f"results/pymol_investigation/{GENE}"
os.makedirs(OUT, exist_ok=True)

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
       "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
       "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
DNA = {"DA","DC","DG","DT"}
BACK = {"N","C","O"}
BASE_ATOMS = {"N1","C2","N3","C4","C5","C6","N7","C8","N9","O2","N2","O4","N4","O6","N6","C7","C5M"}
SUGARP = {"P","OP1","OP2","OP3","O5'","C5'","C4'","O4'","C3'","O3'","C2'","C1'"}
AROM = {"F","Y","W","H"}; POS = {"R","K"}; NEG = {"D","E"}

pdbf = glob.glob(f"{PDBDIR}/{PDBID}_*_{CHAIN}_WITH_*.pdb")[0]

# ── parse: protein residues (ordered) + their atoms; DNA atoms tagged base/backbone ──
prot = {}; order = []
dna_base = []; dna_all = []   # (chain,resn,resnum,atom,xyz)
for line in open(pdbf):
    if line[:6] not in ("ATOM  ", "HETATM"): continue
    atom = line[12:16].strip(); resn = line[17:20].strip(); ch = line[21]
    rs = line[22:26].strip()
    try: xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    except ValueError: continue
    if ch == CHAIN and resn in AA3:
        key = int(rs)
        if key not in prot: prot[key] = {"aa": AA3[resn], "side": [], "all": []}; order.append(key)
        prot[key]["all"].append(xyz)
        if atom not in BACK and not atom.startswith("H"): prot[key]["side"].append(xyz)
    elif resn in DNA:
        if atom in BASE_ATOMS: dna_base.append((ch, resn, rs, atom, xyz))
        if not atom.startswith("H"): dna_all.append((ch, resn, rs, atom, xyz))

# ── TFScope importance, mapped DBD-index -> resnum (file order == DBD index) ──
row = next(r for r in json.load(open("results/per_family/alascan_population.json"))["rows"]
           if r["filename"].lower().startswith(f"{PDBID.lower()}_{CHAIN.lower()}"))
imp = np.nan_to_num(np.array(row["imp"], float)); L = row["L"]; rec = set(row["recog"])
n = min(L, len(order))
res2imp = {order[i]: float(imp[i]) for i in range(n)}
res2idx = {order[i]: i for i in range(n)}
ranks = {order[i]: int(rk) for rk, i in enumerate(np.argsort(-imp[:n]))}   # 0 = most important
top_resnums = [order[i] for i in np.argsort(-imp[:n])[:TOPN]]

# ── per-residue nearest DNA contact + interaction guess ──
def nearest_contact(side):
    best = (1e9, None)
    for ch, resn, rs, at, xyz in dna_base:
        for s in side:
            d = float(np.linalg.norm(s - xyz))
            if d < best[0]: best = (d, (ch, resn, rs, at))
    return best
def nearest_backbone(side):
    best = 1e9
    for ch, resn, rs, at, xyz in dna_all:
        if at in SUGARP:
            for s in side:
                best = min(best, float(np.linalg.norm(s - xyz)))
    return best

rows_out = []
for resnum in order[:n]:
    side = prot[resnum]["side"] or prot[resnum]["all"]
    d_base, who = nearest_contact(side); d_bb = nearest_backbone(side)
    aa = prot[resnum]["aa"]
    inter = "-"
    if d_base < 4.5:
        if d_base < 3.5: inter = "H-bond/polar"
        elif aa in AROM: inter = "pi-stack/vdW"
        else: inter = "vdW/hydrophobic"
    elif d_bb < 4.0 and aa in POS: inter = "salt-bridge (backbone P)"
    rows_out.append(dict(resnum=resnum, aa=aa, idx=res2idx[resnum],
                         importance=round(res2imp[resnum], 3), rank=ranks[resnum] + 1,
                         top20=resnum in top_resnums, crystal_contact=res2idx[resnum] in rec,
                         nearest_base=(f"{who[0]}/{who[1]}{who[2]}@{who[3]}" if who else "-"),
                         d_base=round(d_base, 2), nearest_bb_dist=round(d_bb, 2), interaction=inter))

# ── (1) importance-in-bfactor PDB ──
imax = imp[:n].max() if imp[:n].max() > 0 else 1
outpdb = f"{OUT}/{GENE}_importance.pdb"
with open(outpdb, "w") as o:
    for line in open(pdbf):
        if line[:6] in ("ATOM  ", "HETATM") and line[21] == CHAIN and line[17:20].strip() in AA3:
            rs = int(line[22:26]); b = 100.0 * res2imp.get(rs, 0.0) / imax
            o.write(line[:60] + f"{b:6.2f}" + line[66:])
        elif line[:6] in ("ATOM  ", "HETATM"):
            o.write(line[:60] + f"{0.0:6.2f}" + line[66:])
        else:
            o.write(line)

# ── (2) annotation table ──
rows_out.sort(key=lambda r: r["rank"])
with open(f"{OUT}/{GENE}_residues.md", "w") as o:
    o.write(f"# TFScope important residues — {GENE} ({PDBID}:{CHAIN})\n\n")
    o.write("rank | resnum | aa | importance | top20 | crystal_contact | nearest_base | d(Å) | interaction\n")
    o.write("---|---|---|---|---|---|---|---|---\n")
    for r in rows_out[:TOPN]:
        o.write(f"{r['rank']} | {r['resnum']} | {r['aa']} | {r['importance']} | "
                f"{'✓' if r['top20'] else ''} | {'✓' if r['crystal_contact'] else ''} | "
                f"{r['nearest_base']} | {r['d_base']} | {r['interaction']}\n")
import csv
with open(f"{OUT}/{GENE}_residues.csv", "w", newline="") as o:
    w = csv.DictWriter(o, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)

# ── (3) PyMOL script ──
sel = "+".join(str(r) for r in sorted(top_resnums))
with open(f"{OUT}/{GENE}_pymol.pml", "w") as o:
    o.write(f"""# TFScope importance investigation — {GENE} ({PDBID}:{CHAIN})
load {os.path.abspath(outpdb)}, {GENE}
bg_color white
hide everything
# protein coloured by TFScope importance (B-factor)
show cartoon, polymer.protein
spectrum b, white_red, polymer.protein and chain {CHAIN}
set cartoon_transparency, 0.1
# DNA
show cartoon, polymer.nucleic
set cartoon_ring_mode, 3
color grey70, polymer.nucleic
# TFScope top-{TOPN} important residues as sticks, labelled
select tfscope_top, chain {CHAIN} and resi {sel}
show sticks, tfscope_top and not name N+C+O
color orange, tfscope_top and name C*
label tfscope_top and name CA, "%s%s" % (one_letter[resn], resi)
set label_size, 14
deselect
orient
set ray_opaque_background, 0
# color scale: white (low importance) -> red (high). sticks = top-{TOPN} hits.
""")

print(f"saved kit for {GENE} ({PDBID}:{CHAIN}) -> {OUT}/")
print(f"  {GENE}_importance.pdb   (B-factor = TFScope importance; spectrum b)")
print(f"  {GENE}_residues.md/.csv (per-residue table)")
print(f"  {GENE}_pymol.pml        (pymol {OUT}/{GENE}_pymol.pml)")
print("\nTop residues:")
print(f"{'rank':>4} {'res':>6} {'imp':>6} {'top20':>6} {'xtal':>5} {'nearest_base':<16} {'d':>5} interaction")
for r in rows_out[:TOPN]:
    print(f"{r['rank']:>4} {r['aa']}{r['resnum']:<5} {r['importance']:>6} "
          f"{'Y' if r['top20'] else '':>6} {'Y' if r['crystal_contact'] else '':>5} "
          f"{r['nearest_base']:<16} {r['d_base']:>5} {r['interaction']}")
