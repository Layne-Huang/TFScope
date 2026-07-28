"""Fig 2a — Corpus augmentation and contact distillation are two synergistic levers.

Single panel (oracle-aligned r, cluster40 deeppbs test, n=84). Built up from the SAME
training records DeepPBS uses:
  DeepPBS training set (467, sequence-only)  0.548
  + augmented corpus (-> 4,250)              0.596   (+0.05)
  + recognition-contact distillation         0.657   (+0.06)
crossing the structure-based DeepPBS reference (0.626, mean gate-oracle-r, unified all-method
protocol; same value as Fig 1d). Bars and reference are all MEAN gate-oracle-r. Each lever
adds ~0.05-0.06; the
two together carry the sequence-only model from a deficit on DeepPBS's own data to past it.

Numbers from results/baseline_ladder/ladder.json, contamination_sweep_tfscope.json
(40% = deeppbs-only = 0.548) + consolidated ablation findings.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
})

GREY   = "#999999"   # DeepPBS training set (sequence-only)
TEAL   = "#56B4E9"   # + augmented corpus
BLUE   = "#0072B2"   # + contact distillation (full model)
GREEN  = "#009E73"   # augmentation delta
ORANGE = "#E69F00"   # contact delta
DEEPPBS = "#444444"

fig, ax = plt.subplots(figsize=(5.1, 4.3))

labels = ["DeepPBS\ntraining set\n(467, seq-only)", "+ Augmented\ncorpus\n(4,250)",
          "+ Contact\ndistillation"]
vals   = [0.548, 0.596, 0.657]
cols   = [GREY, TEAL, BLUE]
x = np.arange(len(vals))

ax.bar(x, vals, color=cols, width=0.6, edgecolor="white", zorder=3)
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

# DeepPBS (structure) reference
ax.axhline(0.626, ls="--", lw=1.6, color=DEEPPBS, zorder=2)
ax.text(0.0, 0.628, "DeepPBS (structure) 0.626", ha="left", va="bottom",
        fontsize=8.8, color=DEEPPBS,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

# +0.05 augmentation lever
arr1 = FancyArrowPatch((0.08, 0.552), (0.92, 0.592), arrowstyle="-|>",
                       mutation_scale=12, lw=1.5, color=GREEN,
                       connectionstyle="arc3,rad=-0.3", zorder=5)
ax.add_patch(arr1)
ax.text(0.5, 0.586, "+0.05\naugment", ha="center", color=GREEN, fontsize=10,
        fontweight="bold")

# +0.06 contact lever
arr2 = FancyArrowPatch((1.08, 0.600), (2.0, 0.654), arrowstyle="-|>",
                       mutation_scale=12, lw=1.5, color=ORANGE,
                       connectionstyle="arc3,rad=-0.3", zorder=5)
ax.add_patch(arr2)
ax.text(1.5, 0.640, "+0.06\ncontacts", ha="center", color=ORANGE, fontsize=10,
        fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.52, 0.685)
ax.set_ylabel("oracle-aligned r")
ax.set_title("Two synergistic levers beat structure", fontsize=12, fontweight="bold")

fig.tight_layout()
out = "figures/figure2a_ablation/figure2a_ablation"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out + ".png /.pdf")
