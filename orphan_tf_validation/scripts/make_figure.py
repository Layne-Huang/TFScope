"""Summary figure for the orphan-TF ChIP-seq validation MVP (ADNP, ZHX2, ZHX3).
A: predicted motif logos. B: composition-controlled enrichment (log2, vs per-peak dinucleotide
shuffle; z annotated). C: best-hit summit-distance density (centrality). D: null-PWM percentile
(real motif vs 100 column-shuffled PWMs).
"""
import os, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PWM_DIR = os.path.join(ROOT, "..", "results", "genome_cre_scan", "pwms")
TFS = ["ADNP", "ZHX2", "ZHX3"]
CELL = {"ADNP": "K562 (tagged)", "ZHX2": "HepG2", "ZHX3": "HepG2 (CRISPR)"}
R = {tf: json.load(open(f"{ROOT}/results/enrichment/{tf}.json")) for tf in TFS}

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker, pandas as pd

def logo(ax, tf):
    P = np.load(f"{PWM_DIR}/{tf}.npy"); P = P / P.sum(0, keepdims=True)
    ic = np.maximum(2 + (P * np.log2(P)).sum(0), 0)
    logomaker.Logo(pd.DataFrame((P * ic).T, columns=list("ACGT")), ax=ax,
                   color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2)
    ax.set_title(f"{tf}\n{CELL[tf]}", fontsize=8.5)

fig = plt.figure(figsize=(11.5, 6.4))
gs = fig.add_gridspec(3, 3, height_ratios=[0.8, 1.1, 1.1], hspace=0.6, wspace=0.32)

# A logos
for i, tf in enumerate(TFS):
    logo(fig.add_subplot(gs[0, i]), tf)
fig.text(0.5, 0.985, "a  TFScope-predicted orphan-TF motifs", ha="center", fontsize=11, fontweight="bold")

# B composition-controlled enrichment
axb = fig.add_subplot(gs[1, 0])
x = np.arange(len(TFS))
l2 = [R[tf]["log2_enrich"] for tf in TFS]
err = [np.log2((R[tf]["real_hits"]) / (np.array(R[tf]["shuf_counts"]).mean() - np.array(R[tf]["shuf_counts"]).std() + 1e-9) + 1e-9) - R[tf]["log2_enrich"] for tf in TFS]
axb.bar(x, l2, color="#4575b4", width=0.6)
axb.axhline(0, color="k", lw=0.7)
for xi, tf in zip(x, TFS):
    axb.text(xi, l2[list(TFS).index(tf)] + 0.02, f"z={R[tf]['z']:.1f}", ha="center", fontsize=8)
axb.set_xticks(x); axb.set_xticklabels(TFS, fontsize=9)
axb.set_ylabel("log2 enrichment\n(vs dinucleotide shuffle)", fontsize=9)
axb.set_title("b  Composition-controlled\n     enrichment", fontsize=9.5, fontweight="bold", loc="left")
axb.set_ylim(0, max(l2) * 1.3)

# C summit-distance density (centrality)
axc = fig.add_subplot(gs[1, 1:])
bins = np.arange(-250, 250, 25) + 12.5
for tf, col in zip(TFS, ["#d73027", "#1a9850", "#7570b3"]):
    h = np.array(R[tf]["dist_hist"], float); h = h / h.sum()
    axc.plot(bins, h, "-o", ms=3, color=col, label=f"{tf} (≤50bp: {R[tf]['frac_hits_within_50bp']:.2f})")
axc.axhline(1 / len(bins), color="k", ls=":", lw=1)
axc.text(250, 1 / len(bins), " uniform", fontsize=7, va="center")
axc.axvline(0, color="#999", lw=0.8)
axc.set_xlabel("distance of best motif hit to ChIP summit (bp)", fontsize=9)
axc.set_ylabel("fraction of peaks", fontsize=9)
axc.set_title("c  Summit-centered density (weak — hits not summit-localized)", fontsize=9.5, fontweight="bold", loc="left")
axc.legend(fontsize=7.5, frameon=False)

# D null-PWM percentile
axd = fig.add_subplot(gs[2, 0])
pct = [R[tf]["null_pwm_percentile"] for tf in TFS]
cols = ["#1a9850" if p >= 0.95 else "#fdae61" for p in pct]
axd.bar(x, pct, color=cols, width=0.6)
axd.axhline(0.95, color="#c00", ls="--", lw=1); axd.text(2.4, 0.955, "95%", fontsize=7, color="#c00")
for xi, p in zip(x, pct): axd.text(xi, p + 0.01, f"{p:.2f}", ha="center", fontsize=8)
axd.set_xticks(x); axd.set_xticklabels(TFS, fontsize=9); axd.set_ylim(0, 1.05)
axd.set_ylabel("percentile vs 100\ncolumn-shuffled PWMs", fontsize=9)
axd.set_title("d  Motif specificity vs null PWMs", fontsize=9.5, fontweight="bold", loc="left")

# D2 null-PWM enrichment distribution (one example: ADNP)
axe = fig.add_subplot(gs[2, 1:])
for tf, col in zip(TFS, ["#d73027", "#1a9850", "#7570b3"]):
    nz = np.array(R[tf]["null_enrich"])
    axe.hist(nz, bins=20, alpha=0.35, color=col, label=f"{tf} null")
    axe.axvline(R[tf]["real_enrich_sub"], color=col, lw=2)
axe.set_xlabel("enrichment (real motif = vertical line; histogram = column-shuffled PWMs)", fontsize=8.5)
axe.set_ylabel("count", fontsize=9)
axe.set_title("e  Real motif vs column-shuffled-PWM null distribution", fontsize=9.5, fontweight="bold", loc="left")
axe.legend(fontsize=7.5, frameon=False)

fig.suptitle("TFScope orphan-TF motifs are enriched in matched ChIP-seq peaks (composition-controlled)",
             fontsize=12, fontweight="bold", y=1.04)
out = f"{ROOT}/results/figures/orphan_chip_validation"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
print(f"saved {out}.png/.pdf")

# results table
with open(f"{ROOT}/results/results_table.md", "w") as o:
    o.write("| TF | cell | peaks | log2 enrich | z | emp p | hits≤50bp | null-PWM %ile |\n")
    o.write("|----|------|------:|------------:|---:|------:|----------:|-------------:|\n")
    for tf in TFS:
        r = R[tf]
        o.write(f"| {tf} | {CELL[tf]} | {r['n_peaks']} | {r['log2_enrich']:+.2f} | {r['z']:.1f} | "
                f"{r['emp_p']:.3f} | {r['frac_hits_within_50bp']:.2f} | {r['null_pwm_percentile']:.2f} |\n")
print(f"saved {ROOT}/results/results_table.md")
