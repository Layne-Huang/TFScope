#!/usr/bin/env python
"""Figure for the v24 ResidueMoE expert interpretation.

  A  expert x amino-acid log2 enrichment (family-composition-controlled), seed42
  B  same for all 5 seeds, rows Hungarian-matched onto seed42 -> are archetypes shared?
  C  per-expert DNA-contact log2 enrichment (co-crystal base contacts), all seeds
  D  cross-seed matched profile correlation vs permutation null

  python scripts/plot_v24_ens_expert_residues.py <results.json> [outdir]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AAS = list("ACDEFGHIKLMNPQRSTVWY")
KEY = "aa_enrichment_within_family"

R = json.load(open(sys.argv[1] if len(sys.argv) > 1
                   else "results/moe_expert_interpretation/v24_ens_expert_residues.json"))
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures_v24_ensemble/figure_moe_expert_residues"
os.makedirs(OUT, exist_ok=True)

seeds = list(R["per_seed"])
E = len(R["per_seed"][seeds[0]]["expert_usage"])


def prof(tag):
    ex = R["per_seed"][tag]["experts"]
    return np.array([[ex[str(e)].get(KEY, {}).get(c, 0.0) for c in AAS] for e in range(E)])


def contact(tag):
    ex = R["per_seed"][tag]["experts"]
    return np.array([ex[str(e)].get("contact_log2_enrichment") if
                     ex[str(e)].get("contact_log2_enrichment") is not None else np.nan
                     for e in range(E)])


# Hungarian assignment already stored (seed42 expert i -> other-seed expert j)
assign = {seeds[0]: {i: i for i in range(E)}}
cs = R.get("cross_seed_aa_within_family", R.get("cross_seed_aa", {}))
for s in seeds[1:]:
    assign[s] = {int(k): int(v) for k, v in cs[s]["assignment"].items()}

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.22)

# ── A: seed42 heatmap ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
M = prof(seeds[0])
v = np.nanpercentile(np.abs(M), 98)
im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
ax.set_xticks(range(20)); ax.set_xticklabels(AAS, fontsize=9)
ax.set_yticks(range(E))
ex0 = R["per_seed"][seeds[0]]["experts"]
ax.set_yticklabels([f"e{e} ({ex0[str(e)]['usage']*100:.0f}%)" for e in range(E)], fontsize=9)
ax.set_title(f"A  {seeds[0]}: expert x residue enrichment\n"
             r"log$_2$ P(aa|e) / E$_{family}$[P(aa)]  (family composition controlled)",
             fontsize=10, loc="left")
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

# ── B: all seeds, matched rows, stacked ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
big = np.vstack([prof(s)[[assign[s][i] for i in range(E)]] for s in seeds])
v = np.nanpercentile(np.abs(big), 98)
im = ax.imshow(big, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
for k in range(1, len(seeds)):
    ax.axhline(k * E - 0.5, color="k", lw=1.2)
ax.set_xticks(range(20)); ax.set_xticklabels(AAS, fontsize=9)
ax.set_yticks([k * E + E / 2 - 0.5 for k in range(len(seeds))])
ax.set_yticklabels(seeds, fontsize=9)
ax.set_title("B  all 5 ensemble members, experts Hungarian-matched onto "
             f"{seeds[0]}\n(shared vertical structure = reproducible archetype)",
             fontsize=10, loc="left")
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

# ── C: contact enrichment ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
w = 0.16
for k, s in enumerate(seeds):
    c = contact(s)[[assign[s][i] for i in range(E)]]
    ax.bar(np.arange(E) + (k - 2) * w, c, w, label=s)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(E)); ax.set_xticklabels([f"e{i}" for i in range(E)])
ax.set_xlabel(f"expert (matched index, {seeds[0]} frame)")
ax.set_ylabel(r"log$_2$ enrichment for DNA-contact residues")
ax.set_title("C  do experts concentrate co-crystal base-contacting residues?\n"
             f"background contact rate = {R['per_seed'][seeds[0]]['contact_background_rate']}",
             fontsize=10, loc="left")
ax.legend(fontsize=8, ncol=5)

# ── D: cross-seed reproducibility ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
xs = seeds[1:]
m = [cs[s]["matched_mean_r"] for s in xs]
n = [cs[s]["null_mean_r"] for s in xs]
n95 = [cs[s]["null_p95_r"] for s in xs]
x = np.arange(len(xs))
ax.bar(x - 0.2, m, 0.4, color="#2c7fb8", label="Hungarian-matched")
ax.bar(x + 0.2, n, 0.4, color="#bdbdbd", label="random permutation (mean)")
ax.plot(x + 0.2, n95, "k_", ms=18, label="null p95")
for i, s in enumerate(xs):
    ax.text(i - 0.2, m[i] + 0.01, f"p={cs[s]['p_vs_null']:g}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"{seeds[0]} vs {s}" for s in xs], fontsize=9)
ax.set_ylabel("mean profile correlation")
ax.set_title("D  are expert archetypes reproducible across seeds?", fontsize=10, loc="left")
ax.legend(fontsize=8)

fig.suptitle("v24 per-residue MoE: what each expert reads on the DBD "
             f"({R['n_proteins']} unique DBDs, {R['per_seed'][seeds[0]]['n_tokens']:,} routed residues)",
             fontsize=12)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/figure_moe_expert_residues.{ext}", dpi=200, bbox_inches="tight")
print(f"saved {OUT}/figure_moe_expert_residues.png/pdf")
