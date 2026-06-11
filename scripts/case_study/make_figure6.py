#!/usr/bin/env python
"""Assemble Figure 6 — sequence-only GATA-class motif nomination for ZGLP1.

Honest framing (retrieval-supported family inference + divergent-flanking limit):
  a  target card (GATA-type ZF domain + leakage/curated-DB facts)
  b  logo comparison: clean de-novo (family-masked) / production RAG / GATA exemplar /
     ZGLP1 experimental H13CORE (divergent ground truth)
  c  RAG prediction vs each GATA family motif (bar)
  d  double dissociation: retrieval-masked GATA recovery vs family-masked de-novo
  e  ZGLP1 RAG-vs-H13CORE per-column agreement (core matches, flanking diverges)
"""
import os, sys, json
sys.path.insert(0, "pwm_rosetta")
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pwm_hybrid.pwm.viz import makeLogo

cfg = yaml.safe_load(open("configs/case_study_zglp1.yaml"))
OUT = cfg["output_dir"]; FIG = cfg["figure_dir"]; EXT = cfg["extended_figure_dir"]
os.makedirs(FIG, exist_ok=True); os.makedirs(EXT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

summ = json.load(open(f"{OUT}/predictions/ZGLP1_prediction_summary.json"))
gpanel = pd.read_csv(f"{OUT}/predictions/ZGLP1_RAG_vs_GATA_family.tsv", sep="\t")
ctl = pd.read_csv(f"{OUT}/validation/GATA_masked_control_metrics.tsv", sep="\t")
perc = pd.read_csv(f"{OUT}/predictions/ZGLP1_divergence_percolumn.tsv", sep="\t")
df = pd.read_parquet(cfg["donor_parquet"]); df["g"] = df["gene_symbol"].astype(str).str.upper()
def trim(p):
    L = int((p.sum(0) > 1e-6).sum()); return p[:, :L] if L >= 2 else p
def pwm_of(fn): return trim(np.frombuffer(df[df.filename == fn].iloc[0]["pwm"], np.float32).reshape(4, -1))
def read_pwm(path): return pd.read_csv(path, sep="\t", index_col=0)[list("ACGT")].values.T
def logo(ax, pwm, title=None, ts=7):
    ppm = np.clip(pwm.T, 1e-9, 1); ppm = ppm / ppm.sum(1, keepdims=True)
    makeLogo(ppm, ax); ax.set_ylim(0, 2); ax.set_yticks([0, 1, 2])
    ax.set_ylabel("bits", fontsize=7); ax.tick_params(labelsize=6)
    if title: ax.set_title(title, fontsize=ts)

clean = read_pwm(f"{OUT}/predictions/ZGLP1_clean_deNovo.pwm.tsv")
rag   = read_pwm(f"{OUT}/predictions/ZGLP1_prod_RAG_LGO.pwm.tsv")
gata_ex = pwm_of(cfg["gata_exemplar_filename"])
gt = pwm_of(cfg["ground_truth_filename"])

CB = {"clean": "#c7c7c7", "rag": "#3b5b92", "gata": "#4c9f70", "truth": "#c0392b"}
fig = plt.figure(figsize=(7.2, 8.6))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 1.0], hspace=0.55, wspace=0.32)

# (a) target card
axa = fig.add_subplot(gs[0, 0]); axa.set_title("a", loc="left", fontweight="bold"); axa.axis("off")
N = cfg["case_full_length"]; axa.set_xlim(0, N); axa.set_ylim(0, 1)
axa.add_patch(Rectangle((0, 0.66), N, 0.09, fc="0.85", ec="0.4", lw=0.6))
d0, d1 = cfg["case_dbd_start"], cfg["case_dbd_end"]
axa.add_patch(Rectangle((d0, 0.62), d1 - d0, 0.17, fc=CB["rag"], ec="k", lw=0.6))
axa.text((d0 + d1) / 2, 0.86, "GATA-type ZF (208–232)\n+ basic tail", ha="center", fontsize=6)
axa.text(0, 0.57, "1", fontsize=6); axa.text(N, 0.57, str(N), ha="right", fontsize=6)
facts = ("ZGLP1 / GLP-1 (UniProt P0C6A0) · germ-cell / oogenesis GATA-type TF\n"
         "• HumanTFs: likely sequence-specific; no JASPAR / PBM / SELEX motif\n"
         "• no protein–DNA complex structure\n"
         "• HAS a HOCOMOCO H13CORE motif → in TFScope training (encoder-leaky)\n"
         "• clean checkpoint (lofo/Other) held out ALL GATA + ZGLP1\n"
         "Input to TFScope: amino-acid sequence + GATA DBD mask only")
axa.text(0.0, 0.45, facts, fontsize=5.9, va="top", transform=axa.transAxes)

# (b) logo comparison
gb = gs[0, 1].subgridspec(2, 2, wspace=0.55, hspace=0.85)
axb = [fig.add_subplot(gb[i // 2, i % 2]) for i in range(4)]
logo(axb[0], clean, f"clean de-novo\n(family-masked)\n{summ['clean_deNovo_consensus']}", ts=6)
logo(axb[1], rag,   f"production RAG\n(GATA core)\n{summ['prod_RAG_consensus']}", ts=6)
logo(axb[2], gata_ex, "GATA3 exemplar\n(MA0037.3)", ts=6)
logo(axb[3], gt,    f"ZGLP1 experimental\n(H13CORE, divergent)\n{summ['ground_truth_consensus']}", ts=6)
axb[0].text(-0.45, 1.7, "b", transform=axb[0].transAxes, fontweight="bold")

# (c) RAG vs GATA family
axc = fig.add_subplot(gs[1, 0]); axc.set_title("c", loc="left", fontweight="bold")
axc.bar(gpanel.gata_gene, gpanel.r_RAG_vs_this_GATA, color=CB["gata"], ec="k", lw=0.4)
axc.axhline(0.6, color="0.5", ls=":", lw=0.8)
axc.axhline(summ["r_RAG_vs_ZGLP1_H13CORE"], color=CB["truth"], ls="--", lw=1.2)
axc.text(5.4, summ["r_RAG_vs_ZGLP1_H13CORE"] + 0.02, "vs ZGLP1\nH13CORE", color=CB["truth"],
         fontsize=5.5, ha="right", va="bottom")
axc.set_ylabel("RAG prediction · oracle r", fontsize=7); axc.set_ylim(0, 1)
axc.tick_params(labelsize=6.5)
axc.text(0.5, 0.93, f"ZGLP1 RAG recovers GATA core\nfamily mean r={summ['gata_family_r_mean']:.2f}",
         transform=axc.transAxes, fontsize=6, ha="center")

# (d) double dissociation control
axd = fig.add_subplot(gs[1, 1]); axd.set_title("d", loc="left", fontweight="bold")
x = np.arange(len(ctl)); w = 0.38
axd.bar(x - w/2, ctl.r_prod_RAG, w, color=CB["rag"], ec="k", lw=0.4, label="retrieval-masked RAG")
axd.bar(x + w/2, ctl.r_clean_deNovo, w, color=CB["clean"], ec="k", lw=0.4, label="family-masked de-novo")
axd.axhline(0.6, color="0.5", ls=":", lw=0.8)
axd.set_xticks(x); axd.set_xticklabels(ctl.gene, fontsize=6.5)
axd.set_ylabel("oracle r vs known GATA motif", fontsize=7); axd.set_ylim(0, 1)
axd.legend(fontsize=5.4, loc="upper right")
axd.text(0.5, -0.30, "retrieval recovers GATA (100%); de-novo cannot (0%)\n"
         "→ ZGLP1 nomination is retrieval-driven, not memorized",
         transform=axd.transAxes, fontsize=5.8, ha="center", va="top")

# (e) per-column divergence
axe = fig.add_subplot(gs[2, :]); axe.set_title("e", loc="left", fontweight="bold")
cols = ["#4c9f70" if r >= 0.5 else "#d98c5f" for r in perc.column_r]
axe.bar(perc.pos, perc.column_r, color=cols, ec="k", lw=0.3)
axe.axhline(0.5, color="0.5", ls=":", lw=0.8)
for _, r in perc.iterrows():
    axe.text(r["pos"], -0.12, f"{r['gt_base']}\n{r['pred_base']}", ha="center", fontsize=6,
             color="0.2")
axe.set_xticks([]); axe.set_ylim(-0.25, 1.0)
axe.set_ylabel("per-column r\n(pred vs ZGLP1 H13CORE)", fontsize=7)
axe.text(0.5, 0.93, "GATA core columns agree; divergent flanking columns disagree "
         "→ testable ZGLP1-specific flanking hypothesis", transform=axe.transAxes,
         fontsize=6, ha="center")
axe.text(0.005, -0.22, "top = experimental base · bottom = TFScope base", transform=axe.transAxes,
         fontsize=5.5, color="0.35")

fig.suptitle("TFScope nominates a GATA-class motif for the orphan germ-cell TF ZGLP1 "
             "(retrieval-supported; divergent flanking)", fontsize=8.8, y=0.997)
fig.savefig(f"{FIG}/Figure6_ZGLP1_case_study.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/Figure6_ZGLP1_case_study.png", dpi=300, bbox_inches="tight")
print(f"wrote {FIG}/Figure6_ZGLP1_case_study.pdf (+png)")
