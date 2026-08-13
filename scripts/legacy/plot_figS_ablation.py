"""Fig S - architecture/design ablation table (mean gate-oracle-r, cluster40 test n=84).

Every TFScope variant scored by the same gate-oracle-r as Fig 1d/2a (LEADERBOARD.json,
eval_full_metrics gate_oracle_r_mean). Shows the chosen model (learned-10 families +
recognition-contact distillation) vs alternatives along three design axes: family
conditioning, contact supervision, and input variants. DeepPBS (0.626) as reference.
(RAG and per-protein-text variants lack a gate-oracle-r readout and are neutral/negative
in the text; omitted here to keep one consistent metric.)
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})
DEEPPBS = 0.626  # unified-protocol mean (ladder_mean.json)

# run -> (label, axis)
META = {
    "fm_deeppbs_contact":     ("Learned-10 families + contact  (TFScope)", "chosen"),
    "semfam34_contact":       ("Semantic families ×34",              "family"),
    "semfam34_contact_fixed": ("Semantic ×34 (fixed embedding)",     "family"),
    "dual_family_rebin34":    ("Dual-family (learned + semantic)",        "family"),
    "semfam46_contact":       ("Semantic families ×46",              "family"),
    "coarse12_matched":       ("Coarse families ×12 (matched)",      "family"),
    "coarse12_contact":       ("Coarse families ×12",                "family"),
    "fm_deeppbs_nocontact":   ("No contact distillation",                 "contact"),
    "dimerdup":               ("Dimer chain-doubling",                    "input"),
}
COLORS = {"chosen": "#0072B2", "family": "#56B4E9", "contact": "#E69F00", "input": "#999999"}
AXIS_LABEL = {"chosen": "Chosen model", "family": "Family conditioning",
              "contact": "Contact supervision", "input": "Input variant"}

lb = {r["run"]: r for r in json.load(open("results/benchmark_all/LEADERBOARD.json"))}
rows = [(META[k][0], lb[k]["gate_or"], META[k][1]) for k in META if lb.get(k, {}).get("gate_or")]
rows.sort(key=lambda z: z[1])                     # ascending -> best on top after barh
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cats = [r[2] for r in rows]
cols = [COLORS[c] for c in cats]

fig, ax = plt.subplots(figsize=(7.4, 4.3))
y = np.arange(len(vals))
ax.barh(y, vals, color=cols, edgecolor="white", zorder=3)
for yi, v in zip(y, vals):
    ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=8.5)
ax.axvline(DEEPPBS, ls="--", lw=1.4, color="#444444", zorder=2)
ax.text(DEEPPBS, len(vals) - 0.3, f"DeepPBS {DEEPPBS:.3f}", rotation=90,
        va="top", ha="right", fontsize=8.5, color="#444444")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0.50, 0.685); ax.set_xlabel("mean oracle-aligned r (gate-active core)")
ax.set_title("Design ablation: learned families + contact distillation is best",
             fontsize=11, fontweight="bold")
handles = [Patch(color=COLORS[k], label=AXIS_LABEL[k]) for k in ["chosen", "family", "contact", "input"]]
ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=False)
fig.tight_layout()
import os
out = "figures/figureS_ablation/figureS_ablation"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out, "| n_variants:", len(vals))
