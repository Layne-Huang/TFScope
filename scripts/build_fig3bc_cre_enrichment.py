"""Fig 3b-c — TFScope-nominated orphan-TF motifs are enriched in cis-regulatory elements,
once base composition is controlled for.

Six orphan TFs (SOHLH1, ADNP, ADNP2, ZHX2, ZHX3, ZGLP1) get a TFScope motif from the canonical
combined no-RAG model (v19_combined_fm_deeppbs_contact); each motif is scanned genome-wide (hg38,
MOODS) and its hit density in ENCODE cCREs (promoters / enhancers) is compared to two baselines:
(b) the whole genome — naive, GC-confounded → AT-rich motifs falsely appear depleted; and
(c) a dinucleotide-shuffle of the cCREs themselves — the composition control, which removes the
GC artifact and reveals 5/6 motifs genuinely enriched. Candidate-level functional plausibility,
not occupancy.

Source: results/genome_cre_scan/{pwms/*.npy, cre_enrichment.json, cre_shuffle_enrichment.json}
Out: figures/figure3bc_cre_enrichment/figure3bc_cre_enrichment.{png,pdf}
"""
import os, json
import numpy as np

SRC = "results/genome_cre_scan"
OUTD = "figures/figure3bc_cre_enrichment"; os.makedirs(OUTD, exist_ok=True)
naive = json.load(open(f"{SRC}/cre_enrichment.json"))
ctrl = json.load(open(f"{SRC}/cre_shuffle_enrichment.json"))

ORDER = ["SOHLH1", "ADNP", "ADNP2", "ZHX2", "ZHX3", "ZGLP1"]
FAM = {"SOHLH1": "bHLH", "ADNP": "Homeo", "ADNP2": "Homeo", "ZHX2": "Homeo",
       "ZHX3": "Homeo", "ZGLP1": "GATA-like"}
pwms = {t: np.load(f"{SRC}/pwms/{t}.npy") for t in ORDER}

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker, pandas as pd
def logo(ax, pwm, title):
    pwm = np.clip(pwm, 1e-9, 1); pwm = pwm / pwm.sum(0, keepdims=True)
    ic = np.maximum(2 + (pwm * np.log2(pwm)).sum(0), 0)
    df = pd.DataFrame((pwm * ic).T, columns=list("ACGT"))
    logomaker.Logo(df, ax=ax, color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2); ax.set_title(title, fontsize=8, pad=3)

fig = plt.figure(figsize=(11.5, 6.6))
gs = fig.add_gridspec(3, 6, height_ratios=[0.8, 1.1, 1.1], hspace=0.55, wspace=0.35)

# (a) predicted orphan motifs
for i, t in enumerate(ORDER):
    logo(fig.add_subplot(gs[0, i]), pwms[t], f"{t}\n({FAM[t]})")
fig.text(0.5, 0.985, "a  TFScope-nominated orphan-TF motifs (combined, no-RAG)",
         ha="center", fontsize=10.5, fontweight="bold")

def bars(ax, get, title, sub, stars=None):
    x = np.arange(len(ORDER)); w = 0.38
    pm = np.array([np.log2(max(get(t)[0], 1e-3)) for t in ORDER])
    en = np.array([np.log2(max(get(t)[1], 1e-3)) for t in ORDER])
    ax.bar(x - w / 2, pm, w, color="#4575b4", label="promoter")
    ax.bar(x + w / 2, en, w, color="#d98c4a", label="enhancer")
    ax.axhline(0, color="k", lw=0.7)
    if stars:
        for xi, t in zip(x, ORDER):
            for off, key in [(-w / 2, 0), (w / 2, 1)]:
                if stars(t)[key]:
                    h = (pm if key == 0 else en)[list(ORDER).index(t)]
                    ax.text(xi + off, h + (0.06 if h >= 0 else -0.18), "*", ha="center",
                            fontsize=13, color="#c00", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{t}\n({FAM[t]})" for t in ORDER], fontsize=7.5)
    ax.set_ylabel("log2 enrichment", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.text(0.0, 1.005, sub, transform=ax.transAxes, fontsize=8, color="#555")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)

# (b) naive vs whole genome
axb = fig.add_subplot(gs[1, :])
bars(axb, lambda t: (naive[t]["prom_enrich"], naive[t]["enh_enrich"]),
     "b  Naive baseline (vs whole genome)", "AT-rich motifs appear DEPLETED — a GC artifact")

# (c) composition-controlled vs dinucleotide shuffle
axc = fig.add_subplot(gs[2, :])
def cget(t): return (ctrl["promoter"][t]["enrich"], ctrl["enhancer"][t]["enrich"])
def cstar(t):
    p, e = ctrl["promoter"][t], ctrl["enhancer"][t]
    return (p["enrich"] > 1 and p["z"] > 2, e["enrich"] > 1 and e["z"] > 2)
bars(axc, cget, "c  Composition-controlled baseline (vs dinucleotide shuffle)",
     "removes GC confound → 5/6 motifs ENRICHED in cCREs (*: z>2)", stars=cstar)

fig.suptitle("Nominated orphan-TF motifs localize to cis-regulatory elements once composition is controlled",
             fontsize=12, fontweight="bold", y=1.04)
out = f"{OUTD}/figure3bc_cre_enrichment"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
# summary line
enr = sum(1 for t in ORDER if max(cget(t)) > 1 and max(ctrl["promoter"][t]["z"], ctrl["enhancer"][t]["z"]) > 2)
print(f"saved {out}.png/.pdf  | {enr}/6 motifs cCRE-enriched (composition-controlled)")
for t in ORDER:
    print(f"  {t:<7} prom {ctrl['promoter'][t]['enrich']:.2f}x (z={ctrl['promoter'][t]['z']:.1f})  "
          f"enh {ctrl['enhancer'][t]['enrich']:.2f}x (z={ctrl['enhancer'][t]['z']:.1f})")
