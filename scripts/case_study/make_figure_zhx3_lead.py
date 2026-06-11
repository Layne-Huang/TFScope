#!/usr/bin/env python
"""Manuscript application figure — ZHX3-lead orphan-TF nomination (deeppbs v18a RAG).

ZHX3 (orphan, no curated motif) is nominated a canonical homeodomain motif and then
corroborated three independent ways: (i) paralog consistency with ZHX2, (ii) functional
enrichment in its target promoters beyond a composition control, (iii) a calibrated +
masked-control-validated pipeline. ZHX2 = paralog support; ADNP2 = breadth.
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
md = pd.read_csv(f"{OUT}/predictions/orphan_prediction_metrics.tsv", sep="\t").set_index("gene")
cons = pd.read_csv(f"{OUT}/predictions/paralog_consistency.tsv", sep="\t").set_index("pair")["r"]
ctl = pd.read_csv(f"{OUT}/validation/homeodomain_masked_control_metrics.tsv", sep="\t")
cal = pd.read_csv(f"{OUT}/confidence/heldout_known_confidence.tsv", sep="\t")
prom = pd.read_csv(f"{OUT}/targets/orphan_promoter_scan_summary.tsv", sep="\t").set_index("gene")
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

z3 = summ["ZHX3"]; z3_top = prom.loc["ZHX3"]
fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.95, wspace=0.42)

# (a) ZHX3 de-novo nomination (retrieval inert on this checkpoint: noRAG==RAG, r=1.0)
axa0 = fig.add_subplot(gs[0, 0]); axa1 = fig.add_subplot(gs[0, 1])
logo(axa0, read_pwm(f"{OUT}/predictions/ZHX3_RAG_LGO.pwm.tsv"),
     f"ZHX3 de-novo motif\n{z3['RAG_consensus']} · conf {z3['confidence']:.2f}", ts=6.5)
logo(axa1, read_pwm(f"{OUT}/predictions/ZHX2_RAG_LGO.pwm.tsv"),
     f"ZHX2 paralog\n{summ['ZHX2']['RAG_consensus']}", ts=6.5)
axa0.text(-0.4, 1.5, "a", transform=axa0.transAxes, fontweight="bold")
axa0.text(0.5, -0.45, f"orphan, no curated motif; predicted from\nsequence alone (retrieval inert here,\n"
          f"noRAG=RAG r=1.0) · r={md.loc['ZHX3','r_vs_TAAT']:.2f} vs canonical TAAT",
          transform=axa0.transAxes, fontsize=5.7, ha="center", va="top")
axa1.text(0.5, -0.45, f"paralog consistency\nZHX2↔ZHX3 r = {cons['ZHX2 vs ZHX3']:.2f}\n"
          "(ZHX1 motif degenerate → no\ncurated reference exists)",
          transform=axa1.transAxes, fontsize=5.7, ha="center", va="top")

# (b) canonical homeodomain reference
axb = fig.add_subplot(gs[0, 2])
logo(axb, hd_pwm(cfg["canonical_homeodomain"]), "canonical\nhomeodomain TAAT", ts=6.5)
axb.text(-0.35, 1.5, "b", transform=axb.transAxes, fontweight="bold")
axb.text(0.5, -0.45, f"ZHX3 vs canonical TAAT\nr = {md.loc['ZHX3','r_vs_TAAT']:.2f}\n"
         "(canonical, not a curated\nZHX motif)",
         transform=axb.transAxes, fontsize=5.7, ha="center", va="top")

# (c) functional validation: promoter enrichment
axc = fig.add_subplot(gs[1, 0]); axc.set_title("c", loc="left", fontweight="bold")
labs = ["ZHX3\nRAG", "canonical\nTAAT", "shuffled\ncontrol"]
vals = [z3_top["auroc_RAG"], z3_top["auroc_canonicalTAAT"], z3_top["auroc_shuffled"]]
axc.bar(range(3), vals, color=["#c0392b", "#4c9f70", "#dcdcdc"], ec="k", lw=0.4)
axc.axhline(0.5, color="k", ls=":", lw=0.8); axc.set_ylim(0.4, 0.8)
axc.set_xticks(range(3)); axc.set_xticklabels(labs, fontsize=6.5)
axc.set_ylabel("AUROC (targets vs\ndinuc-shuffled)", fontsize=7); axc.tick_params(labelsize=6)
axc.text(0.5, 1.10, f"functional: target-promoter enrichment\nAUROC {z3_top['auroc_RAG']} "
         f"(p={z3_top['p_RAG']}) > shuffled {z3_top['auroc_shuffled']}",
         transform=axc.transAxes, fontsize=5.8, ha="center")
axc.text(0.5, -0.30, "top targets: " + ", ".join(z3_top["top_candidates"].split(";")[:4]),
         transform=axc.transAxes, fontsize=5.6, ha="center", va="top", color="0.25")

# (d) calibration / confidence vs accuracy
axd = fig.add_subplot(gs[1, 1]); axd.set_title("d", loc="left", fontweight="bold")
oth = cal[cal.family != "Homeodomain"]
axd.scatter(oth.confidence, oth.oracle_r, s=10, c="#c7c7c7", alpha=0.7, label="held-out TFs")
axd.scatter(HD.confidence, HD.oracle_r, s=24, c="#4c9f70", edgecolor="k", lw=0.3,
            label=f"homeodomain (n={len(HD)})", zorder=3)
axd.axvline(z3["confidence"], color="#7b3fa0", lw=1.3, ls="--")
axd.axhline(0.6, color="0.5", lw=0.8, ls=":")
axd.text(z3["confidence"], 1.02, "ZHX3", color="#7b3fa0", fontsize=6, rotation=90, va="bottom", ha="center")
axd.set_xlabel("calibrated confidence", fontsize=7); axd.set_ylabel("oracle Pearson r", fontsize=7)
axd.set_xlim(0, 1); axd.set_ylim(-0.1, 1.05); axd.legend(fontsize=5.3, loc="lower right")

# (e) masked homeodomain control
axe = fig.add_subplot(gs[1, 2]); axe.set_title("e", loc="left", fontweight="bold")
axe.bar(ctl.gene, ctl.r_RAG, color=["#4c9f70" if r >= 0.6 else "#d98c5f" for r in ctl.r_RAG], ec="k", lw=0.4)
axe.axhline(0.6, color="0.5", ls=":", lw=0.8); axe.set_ylim(0, 1)
axe.set_ylabel("RAG oracle r vs curated", fontsize=7); axe.tick_params(labelsize=6)
axe.text(0.5, 1.04, "masked homeodomain control", transform=axe.transAxes, fontsize=5.8, ha="center")
for i, r in ctl.iterrows():
    axe.text(i, r.r_RAG + 0.02, r.RAG_consensus, ha="center", fontsize=4.6, rotation=90, va="bottom")

fig.suptitle("TFScope nominates and corroborates a homeodomain motif for the orphan TF ZHX3",
             fontsize=9.3, y=0.99)
fig.savefig(f"{FIG}/Figure_ZHX3_application.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/Figure_ZHX3_application.png", dpi=300, bbox_inches="tight")
print(f"wrote {FIG}/Figure_ZHX3_application.pdf (+png)")
