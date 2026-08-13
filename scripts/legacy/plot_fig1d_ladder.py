"""Fig 1d - baseline ladder (MEAN gate-oracle-r, cluster40 test n=84).

All rungs scored by ONE unified protocol (rebuild_baseline_ladder_mean.py): each
method's IC-trimmed predicted core is oracle-aligned (+/-10 shift + RC) to the GT core,
Pearson r per TF, MEAN across TFs. Shows TFScope matches structure-based DeepPBS and
beats every sequence-only baseline. Data: results/baseline_ladder/ladder_mean.json.
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})
J = json.load(open("results/baseline_ladder/ladder_mean.json"))
L = J["ladder_mean"]; SIG = J.get("tfscope_vs_deeppbs", {})
order  = ["random_uniform", "random_train_pwm", "nn_pwm_k1", "esm2_linear",
          "deeppbs_structure", "tfscope_combined"]
labels = ["random\n(uniform)", "random\ntrain-PWM", "NN-PWM\n(retrieval)",
          "ESM2-linear", "DeepPBS\n(structure)", "TFScope"]
GREY = "#BBBBBB"; TEAL = "#56B4E9"; DARK = "#444444"; BLUE = "#0072B2"
cols = [GREY, GREY, GREY, TEAL, DARK, BLUE]
vals = [L[k]["mean"] for k in order]
esm_gap = L["tfscope_combined"]["mean"] - L["esm2_linear"]["mean"]
pval    = SIG.get("p_boot", float("nan"))

fig, ax = plt.subplots(figsize=(6.2, 4.2))
x = np.arange(len(vals))
ax.bar(x, vals, color=cols, width=0.66, edgecolor="white", zorder=3)
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.axhline(L["deeppbs_structure"]["mean"], ls="--", lw=1.2, color=DARK, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ymax = max(vals) + 0.055
ax.set_ylim(0.38, ymax); ax.set_ylabel("mean oracle-aligned r")
ax.set_title("Sequence-only matches structure; architecture justified",
             fontsize=11.5, fontweight="bold")
ax.annotate("", xy=(5, vals[5] - 0.004), xytext=(3, vals[3]),
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.8, ls=":"))
ax.text(1.6, max(vals) - 0.02, f"+{esm_gap:.3f} over ESM2-linear\n(architecture justified)",
        fontsize=8, color="#333", ha="center")
fig.tight_layout()
out = "figures/figure1d_baseline_ladder/figure1d_baseline_ladder"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out, "| vals:", {k: round(v, 3) for k, v in zip(order, vals)}, "| p:", pval)
