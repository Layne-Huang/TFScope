"""Figure 3 — orphan-TF motif validation against matched public ChIP-seq (fully reproducible).

Panels (all from REAL repository data; nothing synthetic):
  a  TFScope-predicted motif logos (results/genome_cre_scan/pwms/*.npy)
  b  Representative ChIP peaks: real ENCODE fold-change bigWig signal at a documented
     auto-selected locus per TF (all four), real peak summit + best MOODS motif hit
  c  Composition-controlled enrichment (log2 vs dinucleotide shuffle; z) from per-TF JSON
  d  Specificity vs 100 column-shuffled-PWM nulls: percentile + null distributions (per-TF JSON)

Config: configs/figure3_orphan_tf_validation.yaml
Outputs: results/figure3/figure3_orphan_tf_validation.{pdf,svg,png}, figure3_plot_data.tsv,
         README.md, figure3_caption_draft.md
Usage: python scripts/plot_figure3_orphan_tf_validation.py
"""
import os, sys, json, subprocess
import numpy as np, pandas as pd, yaml
import MOODS.scan, MOODS.tools
import pyBigWig

CFG = "configs/figure3_orphan_tf_validation.yaml"
OUT = "results/figure3"; os.makedirs(OUT, exist_ok=True)
cfg = yaml.safe_load(open(CFG))
GENOME = cfg["genome"]["fasta"]
BG = [0.295, 0.205, 0.205, 0.295]
PVAL = float(cfg["plot"]["pval_threshold"])
FLANK = int(cfg["plot"]["locus_flank_bp"])
TFS = ["SOHLH1", "ADNP", "ZHX2", "ZHX3"]
COL = cfg["colors"]
CHROMS = {f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]}

# ───────────────────────── load motifs + enrichment summaries ─────────────────────────
pwm = {tf: np.load(cfg["tfs"][tf]["motif"]) for tf in TFS}
pwm = {tf: P / P.sum(0, keepdims=True) for tf, P in pwm.items()}
enr = {tf: json.load(open(cfg["tfs"][tf]["enrichment"])) for tf in TFS}

# recompute percentile + empirical p directly from the stored null arrays (internal consistency)
spec = {}
for tf in TFS:
    null = np.array(enr[tf]["null_enrich"], float); real = float(enr[tf]["real_enrich_sub"])
    pct = float(np.mean(null < real))
    emp_p = (1 + np.sum(null >= real)) / (len(null) + 1)
    spec[tf] = dict(null=null, real=real, percentile=pct, emp_p=emp_p, n_null=len(null))

# ───────────────────────── blacklist + MOODS helpers ─────────────────────────
bl = {}
for line in open(cfg["genome"]["blacklist"]):
    p = line.split(); bl.setdefault(p[0], []).append((int(p[1]), int(p[2])))
def in_blacklist(c, s, e):
    return any(s < be and e > bs for bs, be in bl.get(c, []))

def faidx(region):
    out = subprocess.run(["samtools", "faidx", GENOME, region], capture_output=True, text=True)
    return "".join(l.strip() for l in out.stdout.splitlines() if not l.startswith(">")).upper()

def moods_scanner(P):
    cm = P.tolist(); lo = MOODS.tools.log_odds(cm, BG, 0.01)
    t = MOODS.tools.threshold_from_p(lo, BG, PVAL)
    rc = MOODS.tools.reverse_complement(lo)
    sc = MOODS.scan.Scanner(7); sc.set_motifs([lo, rc], BG, [t, t])
    return sc

def best_hit(sc, seq, L):
    best = None
    for mi, mm in enumerate(sc.scan(seq)):
        strand = "+" if mi == 0 else "-"
        for m in mm:
            if best is None or m.score > best[2]:
                best = (m.pos, strand, m.score)
    return best   # (pos_in_seq, strand, score) or None

# ───────────────────────── panel b: select representative peaks ─────────────────────────
def select_rep(tf, sizes):
    """Documented rule: among top-signal peaks with a significant motif hit, pick the
    highest-signal peak whose best motif score is >= the 75th percentile of motif scores."""
    P = pwm[tf]; L = P.shape[1]; sc = moods_scanner(P)
    rows = []
    for line in open(cfg["tfs"][tf]["peaks"]):
        p = line.split(); c = p[0]
        if c not in CHROMS: continue
        s, e, sig, off = int(p[1]), int(p[2]), float(p[6]), int(p[9])
        summit = s + (off if off >= 0 else (e - s) // 2)
        if summit - FLANK < 0 or summit + FLANK > sizes[c]: continue
        if in_blacklist(c, summit - FLANK, summit + FLANK): continue
        rows.append((c, s, e, summit, sig, p[3]))
    rows.sort(key=lambda r: -r[4]); rows = rows[:3000]          # top signal pool
    scored = []
    for (c, s, e, summit, sig, pid) in rows:
        seq = faidx(f"{c}:{summit-FLANK+1}-{summit+FLANK}")
        bh = best_hit(sc, seq, L)
        if bh is None: continue
        scored.append(dict(chrom=c, start=s, end=e, summit=summit, signal=sig, peak_id=pid,
                           hit_pos=bh[0], hit_strand=bh[1], hit_score=bh[2], L=L))
    if not scored: return None
    q75 = np.percentile([d["hit_score"] for d in scored], 75)
    elig = [d for d in scored if d["hit_score"] >= q75]
    elig.sort(key=lambda d: -d["signal"])                       # highest-signal among strong-motif peaks
    return elig[0]

# ───────────────────────── figure ─────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import logomaker
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": 0.7})
sizes = {l.split()[0]: int(l.split()[1]) for l in open(cfg["genome"]["chrom_sizes"])}

from matplotlib.lines import Line2D
fig = plt.figure(figsize=(7.5, 5.9))                            # ~190 mm wide, no-overlap priority
outer = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.92], width_ratios=[1.0, 1.05],
                 hspace=0.55, wspace=0.40, left=0.085, right=0.985, top=0.885, bottom=0.075)

def qtitle(x, y, letter, text):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", va="bottom", ha="left")
    fig.text(x + 0.022, y, text, fontsize=8.8, fontweight="bold", va="bottom", ha="left")

def logo(ax, tf):
    P = pwm[tf]; ic = np.maximum(2 + (P * np.log2(np.clip(P, 1e-9, 1))).sum(0), 0)
    logomaker.Logo(pd.DataFrame((P * ic).T, columns=list("ACGT")), ax=ax,
                   color_scheme={"A": "#2CA02C", "C": "#1F77B4", "G": "#FF7F0E", "T": "#D62728"},
                   show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([0, 2]); ax.set_ylim(0, 2); ax.tick_params(length=2)
    cav = cfg["tfs"][tf]["caveat"]
    sub = f"{cfg['tfs'][tf]['cell_line']}" + (f" · {cav}" if cav else "")
    ax.text(0.0, 1.04, tf, transform=ax.transAxes, fontsize=8, color=COL[tf], fontweight="bold", ha="left", va="bottom")
    ax.text(1.0, 1.04, sub, transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#666")

# (a) 4 logos stacked
qtitle(0.085, 0.905, "a", "TFScope-predicted orphan-TF motifs")
gsa = GridSpecFromSubplotSpec(4, 1, outer[0, 0], hspace=1.05)
for i, tf in enumerate(TFS):
    ax = fig.add_subplot(gsa[i]); logo(ax, tf)
    if i == 3: ax.set_xlabel("5′ → 3′", fontsize=6.5, labelpad=1)

# (b) representative ChIP peaks (real bigWig signal)
qtitle(0.55, 0.905, "b", "Representative matched ChIP-seq peaks")
reps = {}; plot_rows = []
reptfs = cfg["plot"]["representative_tfs"]
gsb = GridSpecFromSubplotSpec(len(reptfs), 1, outer[0, 1], hspace=0.85)
for i, tf in enumerate(reptfs):
    ax = fig.add_subplot(gsb[i])
    d = select_rep(tf, sizes); reps[tf] = d
    c, summit = d["chrom"], d["summit"]; w0, w1 = summit - FLANK, summit + FLANK
    bw = pyBigWig.open(cfg["tfs"][tf]["bigwig"])
    vals = np.nan_to_num(np.array(bw.values(c, w0, w1), dtype=float)); bw.close()
    ax.fill_between(np.arange(w0, w1), 0, vals, color=COL[tf], lw=0, alpha=0.85)
    ax.set_xlim(w0, w1); ax.set_ylim(0, max(vals.max() * 1.18, 1))
    ax.axvline(summit, color="#333", lw=0.7, ls=(0, (2, 2)))
    hs = w0 + d["hit_pos"]; he = hs + d["L"]
    ax.axvspan(hs, he, color="#000", alpha=0.14, lw=0)
    ax.plot([(hs + he) / 2], [0], marker="^", ms=4, color="#000", clip_on=False)  # motif marker on baseline
    ax.set_yticks([]); ax.tick_params(length=2)
    ax.text(0.0, 1.06, f"{tf}", transform=ax.transAxes, fontsize=7, color=COL[tf], fontweight="bold", va="bottom")
    ax.text(1.0, 1.06, f"{c}:{w0:,}-{w1:,}  motif {d['hit_strand']}", transform=ax.transAxes,
            fontsize=5.8, color="#555", ha="right", va="bottom")
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
    ax.set_xticks([w0, summit, he - d["L"] / 2 if False else summit])
    ax.set_xticks([w0, summit, w1]); ax.set_xticklabels([f"{w0/1e6:.3f}", "summit", f"{w1/1e6:.3f}"], fontsize=5.8)
    if i == len(reptfs) - 1:
        ax.set_xlabel("genomic position (Mb)  ·  ▲ motif hit  ·  ChIP-seq signal", fontsize=6, labelpad=1)
    plot_rows.append(dict(panel="b", tf=tf, **{k: d[k] for k in
                          ["chrom", "start", "end", "summit", "signal", "peak_id", "hit_pos", "hit_strand", "hit_score", "L"]},
                          locus=f"{c}:{w0}-{w1}", signal_max=round(float(vals.max()), 2)))

# (c) composition-controlled enrichment
qtitle(0.085, 0.435, "c", "Enriched in matched ChIP peaks")
axc = fig.add_subplot(outer[1, 0])
x = np.arange(len(TFS)); l2 = [enr[tf]["log2_enrich"] for tf in TFS]
axc.bar(x, l2, color=[COL[tf] for tf in TFS], width=0.62, edgecolor="k", lw=0.5)
for xi, tf in zip(x, TFS): axc.text(xi, enr[tf]["log2_enrich"] + 0.012, f"z={enr[tf]['z']:.1f}", ha="center", fontsize=6.3)
axc.axhline(0, color="k", lw=0.7)
axc.set_xticks(x); axc.set_xticklabels(TFS, fontsize=6.8)
axc.set_ylabel("log$_2$ enrichment\n(vs dinucleotide shuffle)", fontsize=7.2)
axc.set_ylim(0, max(l2) * 1.30); axc.tick_params(length=2)
for sp in ["top", "right"]: axc.spines[sp].set_visible(False)

# (d) specificity vs column-shuffled PWMs (two coordinated subplots)
qtitle(0.55, 0.435, "d", "Specificity vs column-shuffled-PWM nulls")
gsd = GridSpecFromSubplotSpec(1, 2, outer[1, 1], wspace=0.5, width_ratios=[0.72, 1.28])
axd1 = fig.add_subplot(gsd[0])
pcts = [spec[tf]["percentile"] for tf in TFS]
axd1.bar(x, pcts, color=[COL[tf] for tf in TFS], width=0.66, edgecolor="k", lw=0.5)
axd1.axhline(0.95, color="#c00", ls="--", lw=0.8)
axd1.text(-0.45, 0.95, "0.95", fontsize=5.8, color="#c00", va="center", ha="right")
for xi, tf in zip(x, TFS): axd1.text(xi, spec[tf]["percentile"] + 0.02, f"{spec[tf]['percentile']:.2f}", ha="center", fontsize=5.8)
axd1.set_xticks(x); axd1.set_xticklabels(TFS, fontsize=6, rotation=40, ha="right")
axd1.set_ylim(0, 1.12); axd1.set_ylabel("percentile vs null PWMs", fontsize=7); axd1.tick_params(length=2)
for sp in ["top", "right"]: axd1.spines[sp].set_visible(False)

axd2 = fig.add_subplot(gsd[1])
allv = np.concatenate([spec[tf]["null"] for tf in TFS] + [[spec[tf]["real"] for tf in TFS]])
bins = np.linspace(allv.min() * 0.95, np.percentile(allv, 99), 24)
for tf in TFS:
    axd2.hist(spec[tf]["null"], bins=bins, histtype="step", color=COL[tf], lw=1.0)
    axd2.axvline(spec[tf]["real"], color=COL[tf], lw=1.5)
axd2.set_xlabel("enrichment  (vertical line = real motif)", fontsize=6.5)
axd2.set_ylabel("count", fontsize=7); axd2.tick_params(length=2)
for sp in ["top", "right"]: axd2.spines[sp].set_visible(False)
axd2.legend([Line2D([0], [0], color=COL[tf], lw=1.3) for tf in TFS],
            [f"{tf}" for tf in TFS], title=f"null PWMs n={spec['ADNP']['n_null']}",
            fontsize=5.6, title_fontsize=5.6, frameon=False, loc="upper right",
            handlelength=1.0, labelspacing=0.18, borderaxespad=0.1)

for ext in ["pdf", "svg"]:
    fig.savefig(f"{OUT}/figure3_orphan_tf_validation.{ext}", bbox_inches="tight")
fig.savefig(f"{OUT}/figure3_orphan_tf_validation.png", dpi=600, bbox_inches="tight")

# ───────────────────────── plot data + QC table ─────────────────────────
qc = []
for tf in TFS:
    qc.append(dict(TF=tf, motif_len=pwm[tf].shape[1], n_peaks=enr[tf]["n_peaks"],
                   log2_enrichment=enr[tf]["log2_enrich"], z_score=enr[tf]["z"],
                   emp_p_dinuc=enr[tf]["emp_p"], n_null=spec[tf]["n_null"],
                   real_enrich=round(spec[tf]["real"], 3), percentile=round(spec[tf]["percentile"], 3),
                   empirical_p_nullPWM=round(spec[tf]["emp_p"], 4),
                   cell_line=cfg["tfs"][tf]["cell_line"], source=cfg["tfs"][tf]["source"],
                   caveat=cfg["tfs"][tf]["caveat"]))
qc = pd.DataFrame(qc)
# attach rep-peak rows below
with open(f"{OUT}/figure3_plot_data.tsv", "w") as o:
    o.write("# Panel c/d per-TF metrics\n"); qc.to_csv(o, sep="\t", index=False)
    o.write("\n# Panel b representative peaks (selection rule: top-signal peak with best motif score >= 75th pct, "
            "standard chroms, not in ENCODE blacklist)\n")
    pd.DataFrame(plot_rows).to_csv(o, sep="\t", index=False)
print("=== QC table ==="); print(qc.to_string(index=False))
print("\n=== representative peaks (panel b) ===")
for r in plot_rows: print(f"  {r['tf']}: {r['locus']} summit={r['summit']} motif@{r['hit_pos']}({r['hit_strand']}) "
                          f"score={r['hit_score']:.1f} signal={r['signal']:.1f}")
print(f"\nsaved {OUT}/figure3_orphan_tf_validation.{{pdf,svg,png}} + figure3_plot_data.tsv")
