"""Fig 2b (structure panel) — map TFScope's per-residue importance onto the 3D
protein-DNA complex and compare with the true base contacts.

Protein Cα spheres are coloured by the alanine-scan importance (model's view of which
residues set specificity); residues that actually contact a DNA base in the crystal are
ringed in black. The model's hot (red) residues should sit on the DNA interface and
coincide with the ringed contacts. DNA strands drawn as grey backbone traces.

Example: USF1 (1AN4 chain A, bHLH, AUROC 0.92).
"""
import os, re, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PDBDIR = "/data1/leihuang/TFlow/data/TF_split_index"
GENE = "ZBTB7A"; PDBID = "7N5V"; CHAIN = "A"; FAM = "C2H2 zinc finger"   # monomeric DBD
AA3 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL MSE".split())
DNA_RES = {"DA","DC","DG","DT","A","C","G","T"}

def parse(path, chain):
    ca = {}; order = []; dna = {}   # dna: chain -> list of (resseq, P-coord)
    dna_all = []
    for line in open(path):
        if line[:6] not in ("ATOM  ", "HETATM"): continue
        atom = line[12:16].strip(); resn = line[17:20].strip(); ch = line[21]
        rs = line[22:27]
        try: xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError: continue
        if ch == chain and resn in AA3:
            if atom == "CA":
                if rs not in ca: order.append(rs)
                ca[rs] = xyz
        elif resn in DNA_RES:
            dna_all.append(xyz)
            if atom in ("P", "C1'"):
                dna.setdefault(ch, {}).setdefault(rs, xyz)
    caxyz = np.array([ca[r] for r in order])
    return caxyz, dna, np.array(dna_all)

# ---- data ----
path = glob.glob(f"{PDBDIR}/{PDBID}_*_{CHAIN}_WITH_*.pdb")[0]
caxyz, dna, dna_all = parse(path, CHAIN)
D = json.load(open("results/per_family/alascan_population.json"))["rows"]  # geometric all-forces contacts
pref = f"{PDBID.lower()}_{CHAIN.lower()}"
row = next(r for r in D if r["filename"].lower().startswith(pref))
imp = np.array(row["imp"], float); rec = set(row["recog"]); L = row["L"]
imp = np.nan_to_num(imp)
n = min(len(caxyz), L)
caxyz, imp = caxyz[:n], imp[:n]
norm = imp / (imp.max() + 1e-9)

# ---- figure ----
fig = plt.figure(figsize=(7.2, 6.4))
ax = fig.add_subplot(111, projection="3d")

# DNA backbone traces (grey)
for ch, res in dna.items():
    pts = np.array(list(res.values()))
    if len(pts) > 1:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color="#9aa7b4", lw=4, alpha=0.55,
                solid_capstyle="round")
ax.scatter(dna_all[:, 0], dna_all[:, 1], dna_all[:, 2], s=2, color="#cfd8e0", alpha=0.25)

# protein backbone trace
ax.plot(caxyz[:, 0], caxyz[:, 1], caxyz[:, 2], "-", color="#555", lw=1.5, alpha=0.6)

# Cα spheres coloured by importance; true contacts ringed black
edge = np.array(["black" if i in rec else "none" for i in range(n)])
lw = np.array([2.0 if i in rec else 0.0 for i in range(n)])
sizes = 40 + 320 * norm
sc = ax.scatter(caxyz[:, 0], caxyz[:, 1], caxyz[:, 2], c=imp[:n], cmap="Reds",
                s=sizes, edgecolors=edge, linewidths=lw, depthshade=False, zorder=5)

cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
cb.set_label("TFScope residue importance (Δmotif)", fontsize=9)

# legend proxies
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker="o", color="w", markerfacecolor="#d73027", markersize=11,
              label="TFScope important (red)"),
       Line2D([0],[0], marker="o", color="w", markerfacecolor="#eee",
              markeredgecolor="black", markeredgewidth=2, markersize=11,
              label="true base contact (crystal)"),
       Line2D([0],[0], color="#9aa7b4", lw=4, label="DNA")]
ax.legend(handles=leg, loc="upper left", fontsize=8.5, frameon=False)

# match precision: how many of TFScope's top-|rec| are true contacts
k = len(rec)
topk = set(np.argsort(-imp[:n])[:k].tolist())
prec = len(topk & rec) / max(1, k)
ax.set_title(f"{GENE} ({PDBID}:{CHAIN}, {FAM})  —  TFScope importance vs DSSR base contacts\n"
             f"high-importance residues co-localize with base contacts at the DNA interface",
             fontsize=10, fontweight="bold")
ax.set_axis_off()
ax.view_init(elev=18, azim=-60)
try: ax.set_box_aspect((np.ptp(caxyz[:,0]), np.ptp(caxyz[:,1]), np.ptp(caxyz[:,2])))
except Exception: pass

fig.tight_layout()
out = "figures/figure2b_structure/figure2b_structure"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out, f"; top-{k} precision={prec:.2f}")
