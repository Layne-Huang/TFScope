"""Fig 2b — in-silico mutagenesis identifies specificity-determining residues.

Panels:
 (a) example residue-importance track for a clean bHLH (recognition residues red).
 (b) example track for a C2H2 zinc-finger (confounded by Zn-structural residues).
 (c) population: per-TF AUROC(importance -> recognition residue) distribution.
 (d) pooled importance at recognition vs background positions (all test TFs).

Reads results/per_family/alascan_population.json (from alascan_population.py).
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
})
RED = "#D55E00"; GREY = "#B0B0B0"; BLUE = "#0072B2"; GREEN = "#009E73"

from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

D = json.load(open("results/per_family/alascan_population.json"))  # geometric base contacts
rows = D["rows"]                                                   # (all interaction types: H-bond,
                                                                  #  hydrophobic, salt-bridge, pi-stacking)
W = 15   # DBD window margin: test recognition vs OTHER DBD residues (hard negatives),
         # not vs the non-DBD protein tail (which would trivially inflate AUROC).

def windowed(r):
    """Recompute importance/labels restricted to a window bracketing recognition residues."""
    imp = np.array(r["imp"], float); L = r["L"]; rec = set(r["recog"])
    if not r["recog"]:
        return np.nan, np.array([]), np.array([])
    lo = max(0, min(r["recog"]) - W); hi = min(L, max(r["recog"]) + W + 1)
    idx = [i for i in range(lo, hi) if not np.isnan(imp[i])]
    y = np.array([1 if i in rec else 0 for i in idx])
    s = imp[idx]
    auc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
    return auc, s[y == 1], s[y == 0]

for r in rows:
    r["wauc"], _, _ = windowed(r)
by = {r["gene"]: r for r in rows}

def pick(family, want_high_auc=True):
    cand = [r for r in rows if r["family"] == family and r.get("wauc") == r.get("wauc")]
    if not cand: return None
    return sorted(cand, key=lambda r: -r["wauc"] if want_high_auc else r["wauc"])[0]

# Use MONOMERIC DBDs only: bHLH/bZIP read as dimers (each monomer reads a half-site),
# but the model is fed a single monomer sequence, so those families are the wrong illustration.
ex_a = next((r for r in rows if r["filename"].lower().startswith("5zfy_a")), None)   # DUX4, double homeodomain
ex_b = next((r for r in rows if r["filename"].lower().startswith("7n5v_a")), None)   # ZBTB7A, C2H2

fig = plt.figure(figsize=(11.5, 5.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.7, 1.7, 1.2], height_ratios=[1, 1],
                      hspace=0.55, wspace=0.38)
axA = fig.add_subplot(gs[0, :2])
axB = fig.add_subplot(gs[1, :2])
axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[1, 2])

BLUEM = "#0072B2"   # TFScope's own top-k important residues

TOPN = 20
def track(ax, r, title):
    imp = np.nan_to_num(np.array(r["imp"], float)); L = r["L"]
    rec = set(r["recog"])
    topn = set(np.argsort(-imp)[:TOPN].tolist())               # TFScope's top-20 important
    hit = len(topn & rec)                                       # contacts captured in top-20 (recall)
    cols = [RED if i in rec else GREY for i in range(L)]        # fill = DNA contact (all forces)
    x = np.arange(L)
    ax.bar(x, imp, color=cols, width=0.85, zorder=2)
    # mark TFScope's top-20 important residues with a blue caret above the bar
    ymax = imp.max() if imp.max() > 0 else 1
    for i in sorted(topn):
        ax.plot(i, imp[i] + 0.05 * ymax, marker="v", color=BLUEM, ms=5,
                zorder=4, clip_on=False)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_xlabel("DBD residue position", fontsize=9)
    ax.set_ylabel(r"$\Delta$motif (L1)", fontsize=9)
    ttl = f"{title}: {r['gene']} ({r['family']})"
    if r["wauc"] == r["wauc"]:
        ttl += f"  AUROC={r['wauc']:.2f}, top-{TOPN} recovers {hit}/{len(rec)} contacts"
    ax.set_title(ttl, fontsize=10, fontweight="bold")

track(axA, ex_a, "a  homeodomain")
track(axB, ex_b, "b  C2H2 zinc finger")
# legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
axA.legend(handles=[Patch(color=RED, label="DNA contact (all forces, ≤4.5Å)"),
                    Patch(color=GREY, label="other DBD residue"),
                    Line2D([0],[0], marker="v", color="w", markerfacecolor=BLUEM,
                           markersize=8, label="TFScope top-20 important")],
           fontsize=7.2, loc="upper right", frameon=False, ncol=1)

# (c) windowed AUROC distribution
aucs = np.array([r["wauc"] for r in rows if r["wauc"] == r["wauc"]])
axC.hist(aucs, bins=np.arange(0, 1.01, 0.1), color=BLUE, alpha=0.85,
         edgecolor="white")
axC.axvline(0.5, ls="--", color="black", lw=1)
axC.axvline(np.median(aucs), ls="-", color=RED, lw=1.8)
axC.text(np.median(aucs), axC.get_ylim()[1]*0.92, f" med {np.median(aucs):.2f}",
         color=RED, fontsize=8.5, ha="left")
axC.set_xlabel("per-TF AUROC (within DBD)", fontsize=9)
axC.set_ylabel("# TFs", fontsize=9)
axC.set_title(f"c  {(aucs>0.5).mean()*100:.0f}% of {len(aucs)} TFs > chance",
              fontsize=9.5, fontweight="bold", loc="left")

# (d) pooled importance recog vs background (within DBD window)
prec, pbg = [], []
for r in rows:
    _, sr, sb = windowed(r)
    prec += list(sr); pbg += list(sb)
U, pmw = mannwhitneyu(prec, pbg, alternative="greater")
bp = axD.boxplot([pbg, prec], labels=["other\nDBD", "recog."], showfliers=False,
                 patch_artist=True, widths=0.6)
for patch, c in zip(bp["boxes"], [GREY, RED]): patch.set_facecolor(c); patch.set_alpha(0.8)
for med in bp["medians"]: med.set_color("black")
axD.set_ylabel(r"$\Delta$motif (L1)", fontsize=9)
axD.set_title(f"d  recog. > other DBD\n(Mann-Whitney p={pmw:.0e})",
              fontsize=9.5, fontweight="bold", loc="left")

fig.suptitle("In-silico mutagenesis recovers crystal-structure DNA contacts (all forces)",
             fontsize=12.5, fontweight="bold", y=0.99)
out = "figures/figure2b_mutagenesis/figure2b_mutagenesis"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out + ".png /.pdf")
print(f"examples: a={ex_a['gene']}  b={ex_b['gene']}")
