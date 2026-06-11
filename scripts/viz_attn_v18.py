#!/usr/bin/env python
"""Visualize v17 (degenerate) vs v18a (repaired) contact attention.

Rows = models (v17, v18a); cols = TF cases (KLF4, MyoD). Each panel is the WT
cross-attention (PWM position x DBD residue) with the mutated residue marked.
Shows the rank-1 stripe/sink collapse in v17 vs the spread, residue-reading
attention in v18a.
"""
import os, sys, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from attn_v18 import load, run, CKPTS, CASES, metrics

MODELS = ["v17", "v18a"]
fig, axes = plt.subplots(len(MODELS), len(CASES), figsize=(11, 7))
for r, name in enumerate(MODELS):
    m, is_v18, cap = load(CKPTS[name])
    for c, (tf, cs) in enumerate(CASES.items()):
        mut = cs["wt"][:cs["mutsite"]] + cs["mut_aa"] + cs["wt"][cs["mutsite"] + 1:]
        aW, pW = run(m, is_v18, cap, cs["wt"], cs["fam"])
        aM, pM = run(m, is_v18, cap, mut, cs["fam"])
        d = metrics(aW, aM, pW, pM, cs["mutsite"])
        ax = axes[r, c]
        im = ax.imshow(aW, aspect="auto", cmap="magma", vmin=0)
        ax.axvline(cs["mutsite"], color="cyan", ls="--", lw=1.3)
        ax.text(cs["mutsite"] + 0.7, 0.4, f"{cs['wt'][cs['mutsite']]}{cs['mutsite']+1}",
                color="cyan", fontsize=8, va="top")
        ax.set_title(f"{name}  |  {tf} WT attention\n"
                     f"row-const={d['rowconst']:.2f}  H={d['ent']:.2f}/{d['maxent']:.2f}  "
                     f"mass@mut={d['mass_wt']:.3f}", fontsize=8.5)
        ax.set_xlabel("DBD residue"); ax.set_ylabel("PWM position")
        plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("v17 (rank-1 stripe/sink collapse, zero mass on causal residue)  →  "
             "v18a (spread attention that reads the mutated residue)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs("results/v18_attn", exist_ok=True)
out = "results/v18_attn/attn_v17_vs_v18a.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("Saved", out)
