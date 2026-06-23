"""Fig 4a (titration) — sequence-only resolution of a specificity switch scales with the number of
specificity-determining residues changed. GR<->ER recognition-module swap: single residues / the
3-residue P-box do not switch the predicted motif, but swapping the bulk of the module resolves it.
Reads results/myod1_mut/multimutant_titration.json. Out: figures/figure4a_titration/.
"""
import os, json
import numpy as np
SRC = "results/myod1_mut/multimutant_titration.json"; OUTD = "figures/figure4a_titration"; os.makedirs(OUTD, exist_ok=True)
data = json.load(open(SRC))
COL = {"NR3C1->ESR1": "#D95F4C", "ESR1->NR3C1": "#7B6BB1", "AR->ESR1": "#E69F00"}
LAB = {"NR3C1->ESR1": "GR→ER (GRE→ERE)", "ESR1->NR3C1": "ER→GR (ERE→GRE)", "AR→ESR1": "AR→ER", "AR->ESR1": "AR→ER (GRE→ERE)"}

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.7})
fig, ax = plt.subplots(figsize=(5.4, 4.0))
for r in data:
    key = f"{r['src']}->{r['tgt']}"
    x = [t["frac"] * 100 for t in r["titration"]]
    yt = [t["corr_tgt"] for t in r["titration"]]
    ax.plot(x, yt, "-o", color=COL.get(key, "#444"), lw=1.8, ms=4, label=LAB.get(key, key))
ax.axhline(0.7, color="#999", ls=":", lw=1); ax.text(2, 0.72, "resolved (corr→target ≥0.7)", fontsize=6.5, color="#777")
ax.axvspan(0, 12, color="#f1c40f", alpha=0.18, lw=0)
ax.text(6, 0.05, "single residue /\n3-res P-box", fontsize=6.5, ha="center", color="#a07000")
ax.set_xlabel("% of specificity-determining residues swapped (source → target)", fontsize=8.5)
ax.set_ylabel("predicted-motif correlation to TARGET receptor", fontsize=8.5)
ax.set_title("Sequence-only resolution scales with the size of the determinant change",
             fontsize=8.8, fontweight="bold", loc="left")
ax.set_ylim(0, 1.02); ax.set_xlim(-2, 102)
ax.legend(fontsize=7, frameon=False, loc="center right")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout()
out = f"{OUTD}/figure4a_titration"
for e in ["pdf", "svg"]: fig.savefig(f"{out}.{e}", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
print(f"saved {out}.{{png,pdf,svg}}")
for r in data:
    cr = [t["corr_tgt"] for t in r["titration"]]
    print(f"  {r['src']}->{r['tgt']}: corr-to-target 0%={cr[0]:.2f} -> 100%={cr[-1]:.2f}  (crosses 0.7 by "
          f"{next((int(t['frac']*100) for t in r['titration'] if t['corr_tgt']>=0.7), '>100')}%)")
