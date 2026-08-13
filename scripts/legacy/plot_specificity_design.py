"""Specificity-aware design diagnostic figure (Supplementary / limitation).
Shows the experimental-oracle UPPER BOUND vs TFScope-guided design:
  a  experimental specificity margin per target for consensus / target-only / TFScope-proposed,
     with the experimental-oracle upper bound (▲) — the gap = TFScope's resolution limit.
  b  root cause: TFScope predicts the hardest off-targets as near-identical (predicted-PWM corr ~1)
     even though the experimental PWMs are separable (oracle margin > 0) → Case B for all targets.
Out: figures/figure_specificity_design/specificity_design.{png,pdf}
"""
import os, json
import numpy as np, pandas as pd
SRC = "results/specificity_design"; OUTD = "figures/figure_specificity_design"; os.makedirs(OUTD, exist_ok=True)
df = pd.read_csv(f"{SRC}/final_designs.tsv", sep="\t")
summ = json.load(open(f"{SRC}/summary.json"))
off = pd.read_csv(f"{SRC}/off_target_selection.tsv", sep="\t")
TARGETS = ["LHX5", "MYOG", "CREB3L2", "ELK1"]
COL = {"consensus": "#9aa7b4", "target_only": "#4575b4", "proposed": "#d73027", "exp_oracle": "#1a9850"}

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.7})
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.7, 1]})

# (a) experimental margin per target: 3 design methods as bars + oracle upper bound as marker
meths = ["consensus", "target_only", "proposed"]
x = np.arange(len(TARGETS)); w = 0.26
for k, mth in enumerate(meths):
    med = [df[(df.target_tf == t) & (df.method == mth)].margin_exp.median() for t in TARGETS]
    ax[0].bar(x + (k - 1) * w, med, w, color=COL[mth], edgecolor="k", lw=0.4,
              label={"consensus": "consensus", "target_only": "target-only GA", "proposed": "TFScope-proposed"}[mth])
orac = [summ[t]["exp_oracle_upper_bound"] for t in TARGETS]
ax[0].scatter(x, orac, marker="v", s=70, color=COL["exp_oracle"], edgecolor="k", lw=0.5, zorder=5,
              label="experimental-oracle (upper bound)")
for xi, o in zip(x, orac):
    ax[0].annotate("", xy=(xi, o), xytext=(xi, df[(df.target_tf == TARGETS[xi]) & (df.method == "proposed")].margin_exp.median()),
                   arrowprops=dict(arrowstyle="<->", color="#777", lw=0.7))
ax[0].axhline(0, color="k", lw=0.7, ls=":")
ax[0].text(len(TARGETS) - 0.5, 0.06, "margin 0 = no selectivity", fontsize=6.5, color="#555", ha="right")
ax[0].set_xticks(x); ax[0].set_xticklabels(TARGETS, fontsize=8)
ax[0].set_ylabel("experimental specificity margin\n$Z_t^{exp}-\\max_o Z_o^{exp}$", fontsize=8)
ax[0].set_title("a  Task is feasible (oracle) but beyond TFScope's resolution", fontsize=8.8, fontweight="bold", loc="left")
ax[0].legend(fontsize=6.3, frameon=False, loc="upper left", ncol=1)
for s in ["top", "right"]: ax[0].spines[s].set_visible(False)

# (b) root cause: predicted off-target similarity (TFScope sees them as ~identical)
mean_corr = [off[off.target == t].pred_pwm_corr.mean() for t in TARGETS]
ax[1].bar(x, mean_corr, 0.6, color="#d73027", edgecolor="k", lw=0.4)
ax[1].axhline(1.0, color="#999", ls=":", lw=0.8)
for xi, c in zip(x, mean_corr): ax[1].text(xi, c + 0.005, f"{c:.2f}", ha="center", fontsize=6.5)
ax[1].set_xticks(x); ax[1].set_xticklabels(TARGETS, fontsize=8); ax[1].set_ylim(0.8, 1.02)
ax[1].set_ylabel("mean predicted-PWM corr\n(target vs off-targets)", fontsize=8)
ax[1].set_title("b  TFScope predicts off-targets\n     as near-identical", fontsize=8.8, fontweight="bold", loc="left")
for s in ["top", "right"]: ax[1].spines[s].set_visible(False)

fig.suptitle("Specificity-aware design: feasible in PWM space (experimental oracle), not captured by TFScope (Case B)",
             fontsize=9.5, fontweight="bold", y=1.02)
fig.tight_layout()
out = f"{OUTD}/specificity_design"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
print("medians (exp margin):")
for t in TARGETS:
    print(f"  {t:<9} cons={summ[t]['median_margin_exp']['consensus']:+.2f} tgtonly={summ[t]['median_margin_exp']['target_only']:+.2f} "
          f"proposed={summ[t]['median_margin_exp']['proposed']:+.2f} | oracle={summ[t]['exp_oracle_upper_bound']:+.2f}")
print(f"saved {out}.png/.pdf")
