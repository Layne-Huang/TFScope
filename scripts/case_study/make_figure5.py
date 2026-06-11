#!/usr/bin/env python
"""Assemble Figure 5 (confidence-calibrated SOHLH1 nomination) + Extended attention.

Panels:
  a  confidence distribution (held-out known TFs by success, orphan bHLH) + calibration inset
  b  SOHLH1 target card (domain + leakage facts)
  c  4-logo comparison (SOHLH1 noRAG / RAG / SOHLH2 / canonical E-box)
  d  held-out confidence vs accuracy (bHLH highlighted; SOHLH1 marker)
  e  retrieval-masked SOHLH2 positive control (noRAG -> RAG recovers E-box)
Extended: SOHLH1 cross-attention with shaded basic region + column-sum bar.
"""
import os, sys, json
sys.path.insert(0, "pwm_rosetta")
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pwm_hybrid.pwm.viz import makeLogo

cfg = yaml.safe_load(open("configs/case_study_sohlh1.yaml"))
OUT = cfg["output_dir"]; FIG = cfg["figure_dir"]; EXT = cfg["extended_figure_dir"]
os.makedirs(FIG, exist_ok=True); os.makedirs(EXT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

# ── data ─────────────────────────────────────────────────────────────────────
cal = pd.read_csv(f"{OUT}/confidence/heldout_known_confidence.tsv", sep="\t")
orph = pd.read_csv(f"{OUT}/orphan_tf_confidence_table.tsv", sep="\t")
summ = json.load(open(f"{OUT}/predictions/SOHLH1_prediction_summary.json"))
mc = pd.read_csv(f"{OUT}/validation/SOHLH2_masked_control_metrics.tsv", sep="\t").set_index("metric")["value"]
SOH = float(summ["confidence"])

def read_pwm(path): return pd.read_csv(path, sep="\t", index_col=0)[list("ACGT")].values.T
def parse_meme(path):
    rows, on = [], False
    for ln in open(path):
        if ln.startswith("letter-probability"): on = True; continue
        if on:
            p = ln.split()
            if len(p) == 4:
                try: rows.append([float(x) for x in p])
                except ValueError: break
            elif rows: break
    return np.array(rows).T
def ebox():
    m = np.zeros((4, 6))
    for j, ch in enumerate("CACGTG"): m["ACGT".index(ch), j] = 1.0
    return m
def logo(ax, pwm, title=None, ts=8):
    ppm = np.clip(pwm.T, 1e-9, 1); ppm = ppm / ppm.sum(1, keepdims=True)
    makeLogo(ppm, ax); ax.set_ylim(0, 2); ax.set_yticks([0, 1, 2])
    ax.set_ylabel("bits", fontsize=7); ax.tick_params(labelsize=6)
    if title: ax.set_title(title, fontsize=ts)

os.makedirs(f"{OUT}/attention", exist_ok=True)
norag = read_pwm(f"{OUT}/predictions/SOHLH1_noRAG.pwm.tsv")
rag   = read_pwm(f"{OUT}/predictions/SOHLH1_RAG_LGO.pwm.tsv")
_df = pd.read_parquet(cfg["donor_parquet"])
_s2row = _df[_df.filename == cfg["paralog_reference_filename"]].iloc[0]
s2 = np.frombuffer(_s2row["pwm"], np.float32).reshape(4, -1)

# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.2, 8.8))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.5, wspace=0.3)

# (a) confidence distribution + calibration inset
axa = fig.add_subplot(gs[0, 0]); axa.set_title("a", loc="left", fontweight="bold")
suc = cal[cal.success == 1].confidence; fail = cal[cal.success == 0].confidence
bins = np.linspace(0, 1, 16)
axa.hist(fail, bins=bins, color="#c7c7c7", alpha=0.85, label=f"held-out, miss (n={len(fail)})")
axa.hist(suc, bins=bins, color="#4c9f70", alpha=0.8, label=f"held-out, hit r≥0.6 (n={len(suc)})")
axa.hist(orph.confidence, bins=bins, histtype="step", color="#3b5b92", lw=1.4,
         label=f"orphan bHLH (n={len(orph)})")
axa.axvline(SOH, color="#c0392b", lw=1.6)
axa.text(SOH - 0.02, axa.get_ylim()[1]*0.9, f"SOHLH1\n{SOH:.2f}", color="#c0392b",
         fontsize=6.5, ha="right", va="top")
axa.set_xlabel("calibrated confidence", fontsize=7); axa.set_ylabel("count", fontsize=7)
axa.legend(fontsize=5.2, loc="upper left", framealpha=0.9)
# calibration inset
axi = axa.inset_axes([0.62, 0.55, 0.36, 0.42])
cb = pd.read_csv(f"{OUT}/confidence/confidence_calibration_bins.tsv", sep="\t")
xs = [0.1, 0.3, 0.5, 0.7, 0.9][:len(cb)]
axi.plot(xs, cb.success_fraction, "o-", color="#2c3e50", ms=3, lw=1)
axi.set_xlabel("conf.", fontsize=5.5); axi.set_ylabel("frac r≥0.6", fontsize=5.5)
axi.tick_params(labelsize=5); axi.set_ylim(0, max(0.3, cb.success_fraction.max()*1.2))
axi.set_title("calibration", fontsize=5.5)

# (b) target card
axb = fig.add_subplot(gs[0, 1]); axb.set_title("b", loc="left", fontweight="bold"); axb.axis("off")
axb.set_xlim(0, cfg["case_full_length"]); axb.set_ylim(0, 1)
axb.add_patch(Rectangle((0, 0.62), cfg["case_full_length"], 0.10, fc="0.85", ec="0.4", lw=0.6))
d0, d1 = cfg["case_dbd_start"], cfg["case_dbd_end"]
axb.add_patch(Rectangle((d0, 0.58), d1 - d0, 0.18, fc="#3b5b92", ec="k", lw=0.6))
axb.text((d0 + d1) / 2, 0.84, "bHLH DBD (53–104)", ha="center", fontsize=6.5)
axb.text(0, 0.52, "1", fontsize=6); axb.text(cfg["case_full_length"], 0.52, str(cfg["case_full_length"]),
         ha="right", fontsize=6)
facts = ("SOHLH1 (UniProt Q5JUK2) · germ-cell bHLH TF\n"
         "• no curated motif in JASPAR / HOCOMOCO / CIS-BP\n"
         "• no protein–DNA complex structure\n"
         "• absent from TFScope training / retrieval / benchmarks\n"
         f"• ≤{summ['max_train_dbd_identity']*100:.0f}% DBD identity to any training TF\n"
         "Input to TFScope: amino-acid sequence + bHLH DBD mask only")
axb.text(0.0, 0.40, facts, fontsize=6.0, va="top", transform=axb.transAxes)

# (c) 4-logo comparison
gc = gs[1, :].subgridspec(1, 4, wspace=0.45)
axc = [fig.add_subplot(gc[0, i]) for i in range(4)]
logo(axc[0], norag, "SOHLH1 noRAG\n(weak prior)", ts=6.5)
logo(axc[1], rag,   "SOHLH1 LGO-RAG\n(E-box hypothesis)", ts=6.5)
logo(axc[2], s2,    "SOHLH2 reference\n(JASPAR MA1560.1)", ts=6.5)
logo(axc[3], ebox(), "canonical E-box\n(CACGTG)", ts=6.5)
axc[0].text(-0.35, 1.45, "c", transform=axc[0].transAxes, fontweight="bold")
axc[1].text(0.5, -0.42, f"RAG vs SOHLH2 r={summ['r_RAG_vs_SOHLH2']:.2f} · vs E-box r={summ['r_RAG_vs_Ebox']:.2f}\n"
            f"IC={summ['mean_IC_RAG']:.2f} bits · confidence={SOH:.2f} (decisiveness)",
            transform=axc[1].transAxes, fontsize=6, ha="center", va="top")

# (d) held-out confidence vs accuracy
axd = fig.add_subplot(gs[2, 0]); axd.set_title("d", loc="left", fontweight="bold")
oth = cal[cal.family != "bHLH"]; bh = cal[cal.family == "bHLH"]
axd.scatter(oth.confidence, oth.oracle_r, s=8, c="#c7c7c7", alpha=0.7, label="held-out TFs")
axd.scatter(bh.confidence, bh.oracle_r, s=20, c="#3b5b92", edgecolor="k", lw=0.3,
            label=f"held-out bHLH (n={len(bh)})", zorder=3)
axd.axvline(SOH, color="#c0392b", lw=1.4, ls="--"); axd.axhline(0.6, color="0.5", lw=0.8, ls=":")
axd.text(SOH + 0.01, 0.04, "SOHLH1", color="#c0392b", fontsize=6, rotation=90, va="bottom")
axd.set_xlabel("calibrated confidence", fontsize=7); axd.set_ylabel("oracle Pearson r", fontsize=7)
axd.set_xlim(0, 1); axd.set_ylim(-0.1, 1.0); axd.legend(fontsize=5.5, loc="upper left")
axd.text(0.98, 0.02, f"held-out bHLH median r={bh.oracle_r.median():.2f}",
         transform=axd.transAxes, fontsize=5.8, ha="right")

# (e) retrieval-masked SOHLH2 positive control
ge = gs[2, 1].subgridspec(1, 3, wspace=0.5)
axe = [fig.add_subplot(ge[0, i]) for i in range(3)]
s2_nr = read_pwm(f"{OUT}/validation/SOHLH2_masked_noRAG.pwm.tsv")
s2_rag = read_pwm(f"{OUT}/validation/SOHLH2_masked_RAG_LGO.pwm.tsv")
logo(axe[0], s2_nr, "noRAG", ts=6.5); logo(axe[1], s2_rag, "LGO-RAG", ts=6.5)
logo(axe[2], s2, "JASPAR", ts=6.5)
axe[0].text(-0.5, 1.5, "e", transform=axe[0].transAxes, fontweight="bold")
axe[1].text(0.5, 1.62, "retrieval-masked SOHLH2 control", transform=axe[1].transAxes,
            fontsize=6.3, ha="center")
axe[1].text(0.5, -0.42, f"RAG recovers known motif\nr={float(mc['r_RAG_vs_JASPAR_MA1560.1']):.2f} vs JASPAR",
            transform=axe[1].transAxes, fontsize=6, ha="center", va="top")

fig.suptitle("Confidence-calibrated TFScope nomination of an E-box motif for SOHLH1",
             fontsize=9.5, y=0.995)
fig.savefig(f"{FIG}/Figure5_SOHLH1_case_study.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/Figure5_SOHLH1_case_study.png", dpi=300, bbox_inches="tight")
print(f"wrote {FIG}/Figure5_SOHLH1_case_study.pdf (+png)")

# ── Extended: attention map ──────────────────────────────────────────────────
attn = np.load(f"{OUT}/predictions/SOHLH1_attention.npy")
seq1 = cfg["case_dbd_sequence"]
if attn.size:
    A = attn[:rag.shape[1], :]
    fe = plt.figure(figsize=(7.0, 3.0))
    g2 = fe.add_gridspec(2, 1, height_ratios=[1, 3.2], hspace=0.05)
    axtop = fe.add_subplot(g2[0]); axtop.bar(range(len(seq1)), A.mean(0), color="#3b5b92", width=0.9)
    axtop.set_xlim(-0.5, len(seq1) - 0.5); axtop.set_xticks([]); axtop.tick_params(labelsize=6)
    axtop.set_ylabel("mean\nattn", fontsize=6)
    axm = fe.add_subplot(g2[1]); im = axm.imshow(A, aspect="auto", cmap="magma")
    axm.set_yticks(range(A.shape[0])); axm.set_yticklabels([f"m{j+1}" for j in range(A.shape[0])], fontsize=6)
    axm.set_xticks(range(len(seq1))); axm.set_xticklabels(list(seq1), fontsize=5)
    axm.set_xlabel("SOHLH1 bHLH DBD residue (UniProt 53→110)", fontsize=7)
    axm.set_ylabel("motif position", fontsize=7)
    axm.axvspan(-0.5, 15.5, color="cyan", alpha=0.12)
    axm.text(7.5, -0.7, "basic region (DNA-contacting)", color="#0b6b8f", fontsize=6, ha="center")
    cb2 = fe.colorbar(im, ax=[axtop, axm], fraction=0.02, pad=0.01); cb2.ax.tick_params(labelsize=5)
    fe.suptitle("Extended Data: SOHLH1 cross-attention onto the bHLH DBD", fontsize=8)
    fe.savefig(f"{EXT}/extended_sohlh1_attention_map.pdf", bbox_inches="tight")
    pd.DataFrame(A, columns=list(seq1)).to_csv(f"{OUT}/attention/SOHLH1_attention.tsv", sep="\t")
    print(f"wrote {EXT}/extended_sohlh1_attention_map.pdf")
