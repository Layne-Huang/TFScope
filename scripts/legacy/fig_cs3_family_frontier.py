#!/usr/bin/env python
"""Case study 3 — Family conditioning: the zinc-finger transfer frontier.

Demonstrates the family-conditioned TFScope prior under leave-family-out (LFO):
per-family oracle-r distributions over 4,241 held-out TFs. The point is that the
family prior transfers cleanly for families with one shared recognition grammar,
but long C2H2 zinc-finger arrays (per-finger codes, no family consensus) show the
widest within-family spread and sit at the transfer frontier.

Data: results/lofo/per_tf_oracle_r.json  (file-backed, no model needed)
Out:  figures/cs3_family_frontier.pdf / .png
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SRC = "results/lofo/per_tf_oracle_r.json"
OUT = "figures/cs3_family_frontier"

# Colorblind-safe (Okabe-Ito): C2H2 subfamilies vermillion, others muted blue/grey.
C2H2 = {"C2H2_long", "C2H2_medium", "C2H2_short"}
COL_C2H2 = "#D55E00"
COL_OTHER = "#0072B2"
COL_LONG = "#9E2A00"   # darker for the long-array frontier

d = json.load(open(SRC))
rows = []
for fam, lst in d.items():
    rs = np.array([x["oracle_r"] for x in lst], float)
    rows.append((fam, rs))
# order by median ascending (worst transfer at bottom)
rows.sort(key=lambda x: np.median(x[1]))

fams = [r[0] for r in rows]
data = [r[1] for r in rows]
meds = [np.median(r) for r in data]
ns = [len(r) for r in data]
overall_mean = np.concatenate(data).mean()

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ypos = np.arange(len(fams))

# horizontal box + jittered strip
for i, rs in enumerate(data):
    is_long = fams[i] == "C2H2_long"
    is_c2h2 = fams[i] in C2H2
    face = COL_LONG if is_long else (COL_C2H2 if is_c2h2 else COL_OTHER)
    bp = ax.boxplot(rs, positions=[i], vert=False, widths=0.62, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=1.6),
                    whiskerprops=dict(color=face, lw=1.1),
                    capprops=dict(color=face, lw=1.1),
                    boxprops=dict(facecolor=face, edgecolor=face, alpha=0.30))
    jit = (np.random.RandomState(0).rand(len(rs)) - 0.5) * 0.42
    ax.scatter(rs, i + jit, s=4, color=face, alpha=0.35, linewidths=0, zorder=3)

ax.axvline(overall_mean, color="0.35", ls="--", lw=1.2, zorder=1)
ax.text(overall_mean, len(fams) - 0.3, f"  LFO transfer floor\n  (mean r = {overall_mean:.2f})",
        fontsize=8, color="0.30", va="top")

ax.set_yticks(ypos)
ax.set_yticklabels([f"{f}  (n={n})" for f, n in zip(fams, ns)], fontsize=9)
ax.set_xlabel("Per-TF oracle Pearson r (leave-family-out)", fontsize=10)
ax.set_xlim(-0.4, 1.02)
ax.axvline(0, color="0.8", lw=0.8, zorder=0)
ax.set_title("Family-conditioned transfer to an unseen family\n"
             "Long C2H2 zinc-finger arrays span the widest range — the frontier",
             fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

leg = [Patch(facecolor=COL_LONG, alpha=0.5, label="C2H2 long array (frontier)"),
       Patch(facecolor=COL_C2H2, alpha=0.5, label="C2H2 short / medium"),
       Patch(facecolor=COL_OTHER, alpha=0.5, label="single-consensus families")]
ax.legend(handles=leg, fontsize=8, loc="lower right", frameon=False)

# annotate the C2H2_long min/max example TFs (same family, opposite outcomes)
long_lst = d["C2H2_long"]
best = max(long_lst, key=lambda x: x["oracle_r"])
worst = min(long_lst, key=lambda x: x["oracle_r"])
yl = fams.index("C2H2_long")


def gene(fn):
    # filenames look like pdb_chain_GENE.SOURCE.txt
    return fn.split("_", 2)[-1].split(".")[0]


ax.annotate(f"{gene(best['fn'])}\nr={best['oracle_r']:.2f}",
            xy=(best["oracle_r"], yl), xytext=(0.62, yl - 1.15),
            fontsize=7.5, color=COL_LONG, ha="center",
            arrowprops=dict(arrowstyle="->", color=COL_LONG, lw=0.9))
ax.annotate(f"{gene(worst['fn'])}\nr={worst['oracle_r']:.2f}",
            xy=(worst["oracle_r"], yl), xytext=(-0.18, yl - 1.15),
            fontsize=7.5, color=COL_LONG, ha="center",
            arrowprops=dict(arrowstyle="->", color=COL_LONG, lw=0.9))

fig.tight_layout()
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
print("saved", OUT + ".pdf")
print(f"overall LFO mean r = {overall_mean:.3f}, n = {sum(ns)}")
print(f"C2H2_long best  {best['fn']}  r={best['oracle_r']:.3f}")
print(f"C2H2_long worst {worst['fn']}  r={worst['oracle_r']:.3f}")
