#!/usr/bin/env python
"""Payoff figure: sequence-only annotation of poorly-characterized TFs,
validated against the external Codebook expert-curated motifs.

Panel A: distribution of TFScope-vs-Codebook per-TF r over the 48 clean
         held-out dark TFs, with the database-concordance ceiling for reference.
Panel B: named example logos (TFScope prediction vs Codebook experimental) for
         dark zinc-finger TFs and a recognizable control.

Reads results/codebook_external/{scores.json, arrays.npz}.
Out: figures/cs_codebook_external.pdf / .png
"""
import os, sys, json, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "pwm_rosetta")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tfscope.models.alignment import align_pwm
from pwm_hybrid.pwm.viz import makeLogo

S = json.load(open("results/codebook_external/scores.json"))
A = np.load("results/codebook_external/arrays.npz")
OUT = "figures/cs_codebook_external"

rp = np.array([r["r_tfscope_vs_codebook"] for r in S["per_gene"]], float)
rc = np.array([r["r_dbtarget_vs_codebook"] for r in S["per_gene"]], float)

EXAMPLES = ["ZNF516", "ZNF395", "POU5F1"]   # dark ZNF wins + recognizable control (OCT4)
DISPLAY = {"POU5F1": "POU5F1 (OCT4)"}

fig = plt.figure(figsize=(8.6, 6.2))
gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.0], hspace=0.55, wspace=0.28)

# ── Panel A: r distribution ───────────────────────────────────────────────────
axA = fig.add_subplot(gs[:, 0])
bins = np.linspace(-0.4, 1.0, 22)
axA.hist(rp, bins=bins, color="#0072B2", alpha=0.85, edgecolor="white", label="TFScope vs Codebook")
axA.axvline(np.nanmedian(rp), color="#0072B2", ls="-", lw=2,
            label=f"TFScope median r={np.nanmedian(rp):.2f}")
axA.axvline(np.nanmedian(rc), color="#888888", ls="--", lw=2,
            label=f"DB-concordance ceiling median={np.nanmedian(rc):.2f}")
axA.set_xlabel("Per-TF Pearson r vs Codebook experimental motif", fontsize=10)
axA.set_ylabel("number of TFs", fontsize=10)
axA.set_title(f"Sequence-only annotation of {len(rp)} poorly-characterized TFs\n"
              f"(held out at 40% identity; external Codebook validation)",
              fontsize=10.5, fontweight="bold")
axA.legend(fontsize=8, frameon=False, loc="upper left")
axA.spines[["top", "right"]].set_visible(False)
axA.text(0.02, 0.62, f"mean r = {np.nanmean(rp):.2f}\n{int(np.mean(rp>0.5)*100)}% of TFs r>0.5",
         transform=axA.transAxes, fontsize=9, color="#0072B2",
         bbox=dict(boxstyle="round", fc="white", ec="#0072B2", alpha=0.8))

# ── Panel B: example logos ────────────────────────────────────────────────────
def draw(ax, ppm, L, title, color):
    p = np.clip(ppm[:, :L].T, 1e-8, 1); p = p / p.sum(1, keepdims=True)
    makeLogo(p, ax); ax.set_ylim(0, 2); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, color=color, fontweight="bold")

per = {r["gene"]: r for r in S["per_gene"]}
for row, g in enumerate(EXAMPLES):
    cb = A[f"{g}_codebook"]; pred = A[f"{g}_pred"]; gate = A[f"{g}_gate"]
    L = cb.shape[1]
    gm = gate > 0.5
    if gm.sum() < 2: gm = np.ones(pred.shape[1], bool)
    pv = pred[:, gm]
    al, _, _, _ = align_pwm(pv, cb, max_shift=10, consider_revcomp=True)
    name = DISPLAY.get(g, g)
    r = per[g]["r_tfscope_vs_codebook"]
    axc = fig.add_subplot(gs[row, 1])
    if row == 0:
        pass
    # stack: codebook (top half) then prediction — use two mini-axes via inset
    # simpler: draw codebook then prediction side by side in one row using twin subplots
    # We instead draw codebook on this axis and prediction in a sibling created below.
    draw(axc, cb, L, f"{name}  —  Codebook (experiment)", "#333333")

# add a second column of prediction logos by overlaying a thin row beneath each
# (re-layout: use 3x2 where col2 row = codebook, and we annotate prediction inline)
# Simpler robust approach: redo panel B as 6 stacked mini-axes.
for ax in fig.axes:
    if ax is not axA and ax.get_subplotspec().get_geometry()[2] >= 1:
        pass

fig.savefig(OUT + "_tmp.pdf")  # placeholder to ensure layout valid
plt.close(fig)

# ---- Cleaner re-implementation: Panel A + 3 stacked (codebook/pred) pairs -----
fig = plt.figure(figsize=(8.8, 6.4))
outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.26)
axA = fig.add_subplot(outer[0, 0])
axA.hist(rp, bins=bins, color="#0072B2", alpha=0.85, edgecolor="white")
axA.axvline(np.nanmedian(rp), color="#0072B2", ls="-", lw=2)
axA.axvline(np.nanmedian(rc), color="#888888", ls="--", lw=2)
axA.set_xlabel("Per-TF Pearson r vs Codebook experimental motif", fontsize=10)
axA.set_ylabel("number of TFs", fontsize=10)
axA.set_title(f"Sequence-only annotation of {len(rp)}\npoorly-characterized TFs (external validation)",
              fontsize=10.5, fontweight="bold")
axA.spines[["top", "right"]].set_visible(False)
axA.text(0.97, 0.97,
         f"mean r = {np.nanmean(rp):.2f}\nmedian = {np.nanmedian(rp):.2f}\n"
         f"{int(np.mean(rp>0.5)*100)}% of TFs r>0.5\n"
         f"ceiling median = {np.nanmedian(rc):.2f}",
         transform=axA.transAxes, fontsize=8.5, va="top", ha="right",
         bbox=dict(boxstyle="round", fc="white", ec="#0072B2", alpha=0.85))
axA.annotate("held out at 40% identity;\nblue line = TFScope, grey = achievable ceiling",
             xy=(0.5, -0.16), xycoords="axes fraction", ha="center", fontsize=7.5, color="0.4")

inner = outer[0, 1].subgridspec(6, 1, hspace=0.85)
for i, g in enumerate(EXAMPLES):
    cb = A[f"{g}_codebook"]; pred = A[f"{g}_pred"]; gate = A[f"{g}_gate"]
    L = cb.shape[1]; gm = gate > 0.5
    if gm.sum() < 2: gm = np.ones(pred.shape[1], bool)
    al, _, _, _ = align_pwm(pred[:, gm], cb, max_shift=10, consider_revcomp=True)
    name = DISPLAY.get(g, g); r = per[g]["r_tfscope_vs_codebook"]
    ax1 = fig.add_subplot(inner[2*i]); draw(ax1, cb, L, f"{name} — Codebook (experiment)", "#333333")
    ax2 = fig.add_subplot(inner[2*i+1]); draw(ax2, al, min(al.shape[1], L),
                                              f"TFScope (sequence only)  r={r:.2f}", "#0072B2")

fig.suptitle("TFScope provides sequence-only specificity for poorly-characterized TFs",
             fontsize=11.5, fontweight="bold", y=0.98)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
if os.path.exists(OUT + "_tmp.pdf"): os.remove(OUT + "_tmp.pdf")
print("saved", OUT + ".pdf")
print(f"n={len(rp)} mean={np.nanmean(rp):.3f} median={np.nanmedian(rp):.3f} ceiling={np.nanmedian(rc):.3f}")
