"""Composite Figure 3d — model-guided DNA design with TFScope.
  a  in-silico SELEX: optimisation converges from random DNA to each factor's consensus
  b  good examples: TFScope-guided designs are target-selective on independent experimental PWMs
  c  the law: experimental design transfer is governed by target self-prediction fidelity
Reads existing results: fig3d_evolution.json, good_designs.tsv, scan_table.tsv. No recompute.
Out: figures/figure3d_composite/figure3d_composite.{png,pdf,svg}
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUTD = "figures/figure3d_composite"; os.makedirs(OUTD, exist_ok=True)
evo = json.load(open("results/fig3d_evolution/fig3d_evolution.json"))
good = pd.read_csv("results/specificity_design/good_designs.tsv", sep="\t")
T = pd.read_csv("results/specificity_design/scan_table.tsv", sep="\t")
FAM_COL = {"Homeodomain": "#7B6BB1", "bHLH": "#55A868", "bZIP": "#D95F4C", "ETS": "#E69F00",
           "Forkhead": "#3B9AB2", "C2H2_short": "#CC6677", "C2H2_medium": "#882255",
           "Nuclear_Receptor": "#117733"}

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.7})
fig = plt.figure(figsize=(12, 3.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0], wspace=0.34, left=0.06, right=0.985, top=0.84, bottom=0.17)

# (a) SELEX convergence
axa = fig.add_subplot(gs[0, 0])
for fam in evo:
    a = np.array(evo[fam]["mean_aff"]); rand = a[0]; best = evo[fam]["best_aff"]
    axa.plot((a - rand) / (best - rand + 1e-9), lw=1.8, color=FAM_COL.get(fam, "#444"),
             label=f"{evo[fam]['gene']} ({fam})")
axa.axhline(1.0, color="#999", ls=":", lw=1)
axa.set_xlabel("evolution generation", fontsize=8); axa.set_ylim(-0.03, 1.08)
axa.set_ylabel("predicted affinity\n(0 random → 1 optimal)", fontsize=8)
axa.set_title("a  In-silico SELEX recovers consensus", fontsize=8.8, fontweight="bold", loc="left")
axa.legend(fontsize=6, frameon=False, loc="lower right")
for s in ["top", "right"]: axa.spines[s].set_visible(False)

# (b) good examples: target vs max off-target experimental Z
axb = fig.add_subplot(gs[0, 1])
gg = good.sort_values("margin_exp", ascending=False)
y = np.arange(len(gg))[::-1]; w = 0.4
axb.barh(y + w / 2, gg.target_z_exp, w, color=[FAM_COL.get(f, "#444") for f in gg.family], edgecolor="k", lw=0.3, label="target")
axb.barh(y - w / 2, gg.max_offtarget_z_exp, w, color="#cfd4da", edgecolor="k", lw=0.3, label="worst off-target")
axb.axvline(0, color="k", lw=0.6)
axb.set_yticks(y); axb.set_yticklabels([f"{t}\n({f.split('_')[0]})" for t, f in zip(gg.target, gg.family)], fontsize=6)
axb.set_xlabel("experimental-PWM Z-score", fontsize=8)
axb.set_title("b  Designs are target-selective (held-out exp PWMs)", fontsize=8.8, fontweight="bold", loc="left")
axb.legend(fontsize=6.3, frameon=False, loc="lower right")
for s in ["top", "right"]: axb.spines[s].set_visible(False)

# (c) the law: transfer vs self-prediction
axc = fig.add_subplot(gs[0, 2])
rho, p = spearmanr(T.target_self_corr, T.tfscope_transfer_margin)
axc.scatter(T.target_self_corr, T.tfscope_transfer_margin, s=9, c="#4575b4", alpha=0.45, lw=0)
edges = np.linspace(0.3, 1.0, 8); bx, by = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (T.target_self_corr >= lo) & (T.target_self_corr < hi)
    if m.sum() >= 4: bx.append((lo + hi) / 2); by.append(T.tfscope_transfer_margin[m].median())
axc.plot(bx, by, "-o", color="#d73027", lw=2, ms=4)
axc.axhline(0, color="k", lw=0.6, ls=":")
axc.set_xlabel("target self-prediction accuracy\n(pred vs experimental PWM r)", fontsize=8)
axc.set_ylabel("experimental transfer margin", fontsize=8)
axc.set_title(f"c  Transfer ∝ self-prediction\nρ={rho:.2f} (p={p:.0e}, n={len(T)})", fontsize=8.8, fontweight="bold", loc="left")
for s in ["top", "right"]: axc.spines[s].set_visible(False)

fig.suptitle("Model-guided DNA design: TFScope recovers consensus and designs target-selective sequences where it predicts accurately",
             fontsize=9.8, fontweight="bold", y=0.99)
out = f"{OUTD}/figure3d_composite"
for ext in ["pdf", "svg"]: fig.savefig(f"{out}.{ext}", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
print(f"saved {out}.{{png,pdf,svg}}")
