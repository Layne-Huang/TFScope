#!/usr/bin/env python
"""Assemble Figure 7 — sequence-only homeodomain motif nomination for ADNP.

  a  confidence distribution (held-out known TFs by success; homeodomain subset) + calibration inset
  b  ADNP target card (multi-ZF + single homeobox architecture; leakage facts)
  c  logos: ADNP noRAG / ADNP RAG / ADNP2 companion / PBX1 (TALE retrieved reference)
  d  held-out confidence vs accuracy (homeodomain highlighted; ADNP marker)
  e  masked homeodomain positive control (RAG recovery of EN1/PITX1/ISL1/PBX1)
"""
import os, sys, json
sys.path.insert(0, "pwm_rosetta")
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pwm_hybrid.pwm.viz import makeLogo

cfg = yaml.safe_load(open("configs/case_study_adnp.yaml"))
OUT = cfg["output_dir"]; FIG = cfg["figure_dir"]; EXT = cfg["extended_figure_dir"]
os.makedirs(FIG, exist_ok=True); os.makedirs(EXT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

summ = json.load(open(f"{OUT}/predictions/ADNP_prediction_summary.json"))
cal = pd.read_csv(cfg["calibration_table"], sep="\t")
cbins = pd.read_csv(cfg["calibration_bins"], sep="\t")
ctl = pd.read_csv(f"{OUT}/validation/homeodomain_masked_control_metrics.tsv", sep="\t")
ADN = float(summ["confidence"])
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

norag = read_pwm(f"{OUT}/predictions/ADNP_noRAG.pwm.tsv")
rag = read_pwm(f"{OUT}/predictions/ADNP_RAG_LGO.pwm.tsv")
adnp2 = read_pwm(f"{OUT}/predictions/ADNP2_RAG_LGO.pwm.tsv")
pbx = pwm_of(cfg["pbx_reference_filename"])
HD = cal[cal.family == "Homeodomain"]

fig = plt.figure(figsize=(7.2, 8.8))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.5, wspace=0.3)

# (a) confidence distribution + calibration inset
axa = fig.add_subplot(gs[0, 0]); axa.set_title("a", loc="left", fontweight="bold")
suc = cal[cal.success == 1].confidence; fail = cal[cal.success == 0].confidence
bins = np.linspace(0, 1, 16)
axa.hist(fail, bins=bins, color="#c7c7c7", alpha=0.85, label=f"held-out, miss (n={len(fail)})")
axa.hist(suc, bins=bins, color="#4c9f70", alpha=0.8, label=f"held-out, hit r≥0.6 (n={len(suc)})")
axa.hist(HD.confidence, bins=bins, histtype="step", color="#7b3fa0", lw=1.4,
         label=f"held-out homeodomain (n={len(HD)})")
axa.axvline(ADN, color="#c0392b", lw=1.6)
axa.text(ADN - 0.02, axa.get_ylim()[1]*0.9, f"ADNP\n{ADN:.2f}", color="#c0392b",
         fontsize=6.5, ha="right", va="top")
axa.set_xlabel("calibrated confidence", fontsize=7); axa.set_ylabel("count", fontsize=7)
axa.legend(fontsize=5.0, loc="upper left", framealpha=0.9)
axi = axa.inset_axes([0.62, 0.55, 0.36, 0.42])
xs = [0.1, 0.3, 0.5, 0.7, 0.9][:len(cbins)]
axi.plot(xs, cbins.success_fraction, "o-", color="#2c3e50", ms=3, lw=1)
axi.set_xlabel("conf.", fontsize=5.5); axi.set_ylabel("frac r≥0.6", fontsize=5.5)
axi.tick_params(labelsize=5); axi.set_title("calibration", fontsize=5.5)

# (b) target card
axb = fig.add_subplot(gs[0, 1]); axb.set_title("b", loc="left", fontweight="bold"); axb.axis("off")
N = cfg["case_full_length"]; axb.set_xlim(0, N); axb.set_ylim(0, 1)
axb.add_patch(Rectangle((0, 0.66), N, 0.08, fc="0.85", ec="0.4", lw=0.6))
for zs, ze in [(74, 97), (107, 129), (165, 188), (221, 244), (447, 469)]:   # C2H2 ZFs
    axb.add_patch(Rectangle((zs, 0.63), ze - zs, 0.14, fc="#b9c4d6", ec="0.4", lw=0.4))
d0, d1 = cfg["case_dbd_start"], cfg["case_dbd_end"]
axb.add_patch(Rectangle((d0, 0.60), d1 - d0, 0.20, fc="#7b3fa0", ec="k", lw=0.7))
axb.text(d0, 0.84, "homeobox (754–814)", ha="center", fontsize=6, color="#5a2d77")
axb.text(150, 0.55, "C2H2 ZFs", fontsize=5.5, color="#5b6b86")
axb.text(0, 0.52, "1", fontsize=6); axb.text(N, 0.52, str(N), ha="right", fontsize=6)
facts = ("ADNP (UniProt Q9H2P0) · neurodevelopmental TF (ChAHP)\n"
         "• Helsmoortel–Van der Aa syndrome; top autism gene\n"
         "• HumanTFs: likely seq-specific; no curated motif\n"
         "• no protein–DNA complex structure\n"
         "• absent from TFScope training / retrieval / benchmarks\n"
         f"• ≤{summ['max_train_dbd_identity']*100:.0f}% DBD identity to any training TF\n"
         "Input to TFScope: amino-acid sequence + homeobox mask only")
axb.text(0.0, 0.43, facts, fontsize=5.8, va="top", transform=axb.transAxes)

# (c) logos
gc = gs[1, :].subgridspec(1, 4, wspace=0.45)
axc = [fig.add_subplot(gc[0, i]) for i in range(4)]
logo(axc[0], norag, "ADNP noRAG\n(weak prior)", ts=6.5)
logo(axc[1], rag,   f"ADNP LGO-RAG\n({summ['RAG_consensus']})", ts=6.5)
logo(axc[2], adnp2, f"ADNP2 companion\n({summ['ADNP2_consensus']})", ts=6.5)
logo(axc[3], pbx,   "PBX1 (TALE neighbour)\nMA/H13CORE", ts=6.5)
axc[0].text(-0.35, 1.45, "c", transform=axc[0].transAxes, fontweight="bold")
axc[1].text(0.5, -0.42, f"ADNP RAG: IC={summ['mean_IC_RAG']:.2f} bits · conf={ADN:.2f} ({summ['confidence_class']})\n"
            f"r vs canonical TAAT={summ['r_RAG_vs_TAAT']:.2f} · vs PBX1={summ['r_RAG_vs_PBX1']:.2f} · "
            f"ADNP↔ADNP2 r={summ['r_ADNP_vs_ADNP2']:.2f}",
            transform=axc[1].transAxes, fontsize=6, ha="center", va="top")

# (d) confidence vs accuracy
axd = fig.add_subplot(gs[2, 0]); axd.set_title("d", loc="left", fontweight="bold")
oth = cal[cal.family != "Homeodomain"]
axd.scatter(oth.confidence, oth.oracle_r, s=8, c="#c7c7c7", alpha=0.7, label="held-out TFs")
axd.scatter(HD.confidence, HD.oracle_r, s=20, c="#7b3fa0", edgecolor="k", lw=0.3,
            label=f"held-out homeodomain (n={len(HD)})", zorder=3)
axd.axvline(ADN, color="#c0392b", lw=1.4, ls="--"); axd.axhline(0.6, color="0.5", lw=0.8, ls=":")
axd.text(ADN + 0.01, 0.04, "ADNP", color="#c0392b", fontsize=6, rotation=90, va="bottom")
axd.set_xlabel("calibrated confidence", fontsize=7); axd.set_ylabel("oracle Pearson r", fontsize=7)
axd.set_xlim(0, 1); axd.set_ylim(-0.1, 1.0); axd.legend(fontsize=5.5, loc="upper left")
axd.text(0.98, 0.02, f"held-out homeodomain median r={HD.oracle_r.median():.2f}",
         transform=axd.transAxes, fontsize=5.8, ha="right")

# (e) masked homeodomain positive control
axe = fig.add_subplot(gs[2, 1]); axe.set_title("e", loc="left", fontweight="bold")
axe.bar(ctl.gene, ctl.r_RAG, color="#4c9f70", ec="k", lw=0.4)
axe.axhline(0.6, color="0.5", ls=":", lw=0.8)
axe.set_ylabel("RAG oracle r vs curated motif", fontsize=7); axe.set_ylim(0, 1)
axe.tick_params(labelsize=6.5)
axe.text(0.5, 0.93, f"retrieval-masked homeodomain recovery\n{ctl.recovered.mean()*100:.0f}% "
         f"({ctl.recovered.sum()}/{len(ctl)}) recover known motif",
         transform=axe.transAxes, fontsize=6, ha="center")
for i, r in ctl.iterrows():
    axe.text(i, r.r_RAG + 0.02, r.RAG_consensus, ha="center", fontsize=5, rotation=90, va="bottom")

fig.suptitle("Sequence-only homeodomain motif nomination for the neurodevelopmental orphan TF ADNP",
             fontsize=9.3, y=0.997)
fig.savefig(f"{FIG}/Figure7_ADNP_case_study.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/Figure7_ADNP_case_study.png", dpi=300, bbox_inches="tight")
print(f"wrote {FIG}/Figure7_ADNP_case_study.pdf (+png)")
