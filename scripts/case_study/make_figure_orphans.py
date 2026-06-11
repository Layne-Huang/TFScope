#!/usr/bin/env python
"""Figure 8 — sequence-only homeodomain nominations for the remaining orphans
(ADNP2, ZHX2, ZHX3) on the deeppbs v18a RAG checkpoint.

  a  RAG logos (ADNP2 / ZHX2-HD1 / ZHX3-HD1) + canonical TAAT
  b  deeppbs held-out confidence vs accuracy (homeodomain highlighted; orphans marked)
  c  paralog consistency + masked homeodomain control recovery
  d  composition-controlled promoter enrichment (per orphan)
"""
import os, sys, json
sys.path.insert(0, "pwm_rosetta")
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pwm_hybrid.pwm.viz import makeLogo

cfg = yaml.safe_load(open("configs/case_study_orphans_deeppbs.yaml"))
OUT = cfg["output_dir"]; FIG = cfg["figure_dir"]; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

summ = json.load(open(f"{OUT}/predictions/orphan_summaries.json"))
cal = pd.read_csv(f"{OUT}/confidence/heldout_known_confidence.tsv", sep="\t")
cons = pd.read_csv(f"{OUT}/predictions/paralog_consistency.tsv", sep="\t")
ctl = pd.read_csv(f"{OUT}/validation/homeodomain_masked_control_metrics.tsv", sep="\t")
prom = pd.read_csv(f"{OUT}/targets/orphan_promoter_scan_summary.tsv", sep="\t")
md = pd.read_csv(f"{OUT}/predictions/orphan_prediction_metrics.tsv", sep="\t").set_index("gene")
HD = cal[cal.family == "Homeodomain"]
def read_pwm(p): return pd.read_csv(p, sep="\t", index_col=0)[list("ACGT")].values.T
def hd_pwm(c="TAATTA"):
    m = np.full((4, len(c)), 1e-3)
    for j, ch in enumerate(c): m["ACGT".index(ch), j] = 1.0
    return m / m.sum(0, keepdims=True)
def logo(ax, pwm, title=None, ts=7):
    ppm = np.clip(pwm.T, 1e-9, 1); ppm = ppm / ppm.sum(1, keepdims=True)
    makeLogo(ppm, ax); ax.set_ylim(0, 2); ax.set_yticks([0, 1, 2])
    ax.set_ylabel("bits", fontsize=7); ax.tick_params(labelsize=6)
    if title: ax.set_title(title, fontsize=ts)

ORPH = ["ADNP2", "ZHX2", "ZHX3"]; COL = {"ADNP2": "#3b5b92", "ZHX2": "#c0392b", "ZHX3": "#7b3fa0"}
fig = plt.figure(figsize=(7.2, 7.6))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.55, wspace=0.32)

# (a) logos
ga = gs[0, :].subgridspec(1, 4, wspace=0.45)
axa = [fig.add_subplot(ga[0, i]) for i in range(4)]
for i, g in enumerate(ORPH):
    logo(axa[i], read_pwm(f"{OUT}/predictions/{g}_RAG_LGO.pwm.tsv"),
         f"{g}\n{summ[g]['RAG_consensus']}\nconf {summ[g]['confidence']:.2f}", ts=6.5)
logo(axa[3], hd_pwm(cfg["canonical_homeodomain"]), "canonical\nhomeodomain\nTAAT", ts=6.5)
axa[0].text(-0.4, 1.55, "a", transform=axa[0].transAxes, fontweight="bold")

# (b) confidence vs accuracy
axb = fig.add_subplot(gs[1, 0]); axb.set_title("b", loc="left", fontweight="bold")
oth = cal[cal.family != "Homeodomain"]
axb.scatter(oth.confidence, oth.oracle_r, s=10, c="#c7c7c7", alpha=0.7, label="held-out TFs")
axb.scatter(HD.confidence, HD.oracle_r, s=24, c="#4c9f70", edgecolor="k", lw=0.3,
            label=f"held-out homeodomain (n={len(HD)})", zorder=3)
axb.axhline(0.6, color="0.5", lw=0.8, ls=":")
for g in ORPH:
    axb.axvline(summ[g]["confidence"], color=COL[g], lw=1.1, ls="--")
    axb.text(summ[g]["confidence"], 1.02, g, color=COL[g], fontsize=5.5, rotation=90, va="bottom", ha="center")
axb.set_xlabel("calibrated confidence", fontsize=7); axb.set_ylabel("oracle Pearson r", fontsize=7)
axb.set_xlim(0, 1); axb.set_ylim(-0.1, 1.05); axb.legend(fontsize=5.3, loc="lower right")
axb.text(0.02, 0.04, f"deeppbs held-out\nhomeodomain median r={HD.oracle_r.median():.2f}",
         transform=axb.transAxes, fontsize=5.6)

# (c) consistency + masked control
axc = fig.add_subplot(gs[1, 1]); axc.set_title("c", loc="left", fontweight="bold")
labels = list(cons.pair) + list("ctl:" + ctl.gene)
vals = list(cons.r) + list(ctl.r_RAG)
colors = ["#2c3e50"] * len(cons) + ["#4c9f70" if r >= 0.6 else "#d98c5f" for r in ctl.r_RAG]
y = np.arange(len(labels))[::-1]
axc.barh(y, vals, color=colors, ec="k", lw=0.4)
axc.axvline(0.6, color="0.5", ls=":", lw=0.8)
axc.set_yticks(y); axc.set_yticklabels(labels, fontsize=6)
axc.set_xlim(0, 1); axc.set_xlabel("oracle r", fontsize=7); axc.tick_params(labelsize=6)
axc.text(0.5, 1.02, "paralog consistency (top) · masked HD control (bottom)",
         transform=axc.transAxes, fontsize=5.6, ha="center")

# (d) promoter enrichment
axd = fig.add_subplot(gs[2, :]); axd.set_title("d", loc="left", fontweight="bold")
x = np.arange(len(prom)); w = 0.27
axd.bar(x - w, prom.auroc_RAG, w, color="#c0392b", ec="k", lw=0.4, label="RAG motif")
axd.bar(x, prom.auroc_canonicalTAAT, w, color="#4c9f70", ec="k", lw=0.4, label="canonical TAAT")
axd.bar(x + w, prom.auroc_shuffled, w, color="#dcdcdc", ec="k", lw=0.4, label="shuffled control")
axd.axhline(0.5, color="k", ls=":", lw=0.8)
axd.set_xticks(x); axd.set_xticklabels([f"{r.gene}\n({r.verdict})" for _, r in prom.iterrows()], fontsize=6.5)
axd.set_ylabel("AUROC (targets vs dinuc-shuffled)", fontsize=7); axd.set_ylim(0.3, 0.82)
axd.legend(fontsize=6, loc="upper right")
axd.text(0.5, 1.02, "composition-controlled promoter scans (ZHX2/ZHX3 enriched; ADNP2 composition-confounded)",
         transform=axd.transAxes, fontsize=5.8, ha="center")

fig.suptitle("Sequence-only homeodomain motif nominations for orphan TFs ADNP2, ZHX2, ZHX3 "
             "(deeppbs v18a RAG)", fontsize=8.8, y=0.995)
fig.savefig(f"{FIG}/Figure8_orphan_homeodomains.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/Figure8_orphan_homeodomains.png", dpi=300, bbox_inches="tight")
print(f"wrote {FIG}/Figure8_orphan_homeodomains.pdf (+png)")
