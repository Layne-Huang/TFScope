"""Specificity-design scan: the continuous law governing experimental design transfer.
Across 166 targets (full TF set, no hand-picking):
  a  experimental transfer margin vs TARGET self-prediction accuracy -> strong monotonic law.
  b  vs off-target separability (the originally-hypothesised driver) -> weak/confounded.
  c  experimental-oracle upper bound is positive almost everywhere (task is feasible);
     TFScope transfer tracks self-prediction, not feasibility.
Also writes representative case selection (sanity / main / stress).
Out: figures/figure_specificity_scan/specificity_scan.{png,pdf}; results/specificity_design/case_selection.tsv
"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
SRC = "results/specificity_design"; OUTD = "figures/figure_specificity_scan"; os.makedirs(OUTD, exist_ok=True)
T = pd.read_csv(f"{SRC}/scan_table.tsv", sep="\t")
T["sep"] = 1 - T["pred_off_corr_max"]                       # predicted separability (high = distinct)
rho_self, p_self = spearmanr(T.target_self_corr, T.tfscope_transfer_margin)
rho_sep, p_sep = spearmanr(T.sep, T.tfscope_transfer_margin)

# representative cases
works = T[T.target_self_corr > 0.96].nlargest(2, "tfscope_transfer_margin")
moderate = T[(T.target_self_corr.between(0.88, 0.96))].nlargest(6, "tfscope_transfer_margin")
hard = T[T.target.isin(["LHX5", "MYOG", "CREB3L2", "ELK1"])]
sel = pd.concat([works.assign(role="sanity"), moderate.assign(role="main"), hard.assign(role="stress")])
sel[["role", "target", "family", "target_self_corr", "pred_off_corr_max", "exp_oracle_margin",
     "tfscope_transfer_margin"]].to_csv(f"{SRC}/case_selection.tsv", sep="\t", index=False)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.7})
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))

def binned(x, y, edges):
    bx, by, bl, bh = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 4:
            bx.append((a + b) / 2); by.append(np.median(y[m]))
            bl.append(np.percentile(y[m], 25)); bh.append(np.percentile(y[m], 75))
    return np.array(bx), np.array(by), np.array(bl), np.array(bh)

# (a) transfer vs self-prediction accuracy — the law
ax[0].scatter(T.target_self_corr, T.tfscope_transfer_margin, s=12, c="#4575b4", alpha=0.5, lw=0)
bx, by, bl, bh = binned(T.target_self_corr.values, T.tfscope_transfer_margin.values, np.linspace(0.3, 1.0, 8))
ax[0].plot(bx, by, "-o", color="#d73027", lw=2, ms=4, zorder=5)
ax[0].fill_between(bx, bl, bh, color="#d73027", alpha=0.12)
ax[0].axhline(0, color="k", lw=0.7, ls=":")
ax[0].set_xlabel("target self-prediction accuracy\n(predicted vs experimental PWM r)", fontsize=8)
ax[0].set_ylabel("experimental transfer margin", fontsize=8)
ax[0].set_title(f"a  Transfer ∝ self-prediction\nSpearman ρ={rho_self:.2f} (p={p_self:.0e})", fontsize=8.6, fontweight="bold", loc="left")
for s in ["top", "right"]: ax[0].spines[s].set_visible(False)

# (b) transfer vs separability — the hypothesis that does NOT hold
ax[1].scatter(T.sep, T.tfscope_transfer_margin, s=12, c="#999", alpha=0.5, lw=0)
bx, by, bl, bh = binned(T.sep.values, T.tfscope_transfer_margin.values, np.linspace(0, 0.5, 7))
ax[1].plot(bx, by, "-o", color="#777", lw=2, ms=4, zorder=5)
ax[1].axhline(0, color="k", lw=0.7, ls=":")
ax[1].set_xlabel("predicted off-target separability\n(1 − max predicted corr)", fontsize=8)
ax[1].set_ylabel("experimental transfer margin", fontsize=8)
ax[1].set_title(f"b  Separability is NOT the driver\nSpearman ρ={rho_sep:.2f} (p={p_sep:.1g})", fontsize=8.6, fontweight="bold", loc="left")
for s in ["top", "right"]: ax[1].spines[s].set_visible(False)

# (c) feasibility (exp-oracle) vs TFScope transfer, colored by self-pred
sc = ax[2].scatter(T.exp_oracle_margin, T.tfscope_transfer_margin, s=14, c=T.target_self_corr,
                   cmap="viridis", alpha=0.8, lw=0)
ax[2].axhline(0, color="k", lw=0.7, ls=":")
ax[2].plot([0, T.exp_oracle_margin.max()], [0, T.exp_oracle_margin.max()], "--", color="#bbb", lw=0.8)
cb = fig.colorbar(sc, ax=ax[2], shrink=0.8, pad=0.02); cb.set_label("self-pred r", fontsize=7)
ax[2].set_xlabel("experimental-oracle margin\n(task feasibility, upper bound)", fontsize=8)
ax[2].set_ylabel("TFScope transfer margin", fontsize=8)
ax[2].set_title("c  Task feasible everywhere;\n     TFScope limited by self-pred", fontsize=8.6, fontweight="bold", loc="left")
for s in ["top", "right"]: ax[2].spines[s].set_visible(False)

fig.suptitle("Experimental design transfer is governed by target self-prediction fidelity, not off-target separability "
             f"(n={len(T)} TFs)", fontsize=9.6, fontweight="bold", y=1.03)
fig.tight_layout()
out = f"{OUTD}/specificity_scan"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
print(f"n={len(T)}  rho(self,transfer)={rho_self:.2f}  rho(sep,transfer)={rho_sep:.2f}")
print("case selection:"); print(sel[["role", "target", "family", "target_self_corr", "tfscope_transfer_margin"]].to_string(index=False))
print(f"saved {out}.png/.pdf + case_selection.tsv")
