#!/usr/bin/env python
"""Figure 5 — sequence-only motif nomination for the orphan bHLH SOHLH1.

Panels:
  a  selection cascade (HumanTFs -> orphan bHLH -> SOHLH1)
  b  SOHLH1 domain schematic + annotation
  c  TFScope predicted logos (noRAG vs RAG_LGO) + confidence
  d  reference comparison: RAG_LGO vs SOHLH2 paralog motif vs canonical E-box
  e  cross-attention from motif positions onto the bHLH DBD residues
"""
import os, sys, json
sys.path.insert(0, "pwm_rosetta")
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrow
from pwm_hybrid.pwm.viz import makeLogo

HERE = os.path.dirname(os.path.abspath(__file__))
CFG  = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
OUT  = CFG["case_study"]["output_dir"]
FIG  = f"{OUT}/figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.6,
                     "svg.fonttype": "none", "pdf.fonttype": 42})

def read_pwm(path):                    # tsv pos x ACGT -> (4,L)
    return pd.read_csv(path, sep="\t", index_col=0)[list("ACGT")].values.T

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
    return np.array(rows).T            # (4,L)

def logo(ax, pwm, title=None):
    ppm = np.clip(pwm.T, 1e-9, 1); ppm = ppm / ppm.sum(1, keepdims=True)
    makeLogo(ppm, ax)
    ax.set_ylim(0, 2); ax.set_yticks([0, 1, 2])
    ax.set_ylabel("bits", fontsize=7)
    if title: ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)

# ── data ───────────────────────────────────────────────────────────────────────
norag = read_pwm(f"{OUT}/predictions/SOHLH1_noRAG.pwm.tsv")
rag   = read_pwm(f"{OUT}/predictions/SOHLH1_RAG_LGO.pwm.tsv")
s2    = parse_meme(f"{OUT}/validation/SOHLH2_reference.meme")
# canonical E-box CACGTG one-hot
ebox = np.zeros((4, 6))
for j, ch in enumerate("CACGTG"): ebox["ACGT".index(ch), j] = 1.0
attn  = np.load(f"{OUT}/predictions/SOHLH1_attention.npy")
conf  = pd.read_csv(f"{OUT}/predictions/SOHLH1_confidence.tsv", sep="\t").iloc[0]
cmp   = pd.read_csv(f"{OUT}/validation/sohlh1_vs_sohlh2_similarity.tsv", sep="\t")
nbrs  = pd.read_csv(f"{OUT}/metadata/sohlh1_retrieval_neighbors.tsv", sep="\t")
seq1  = CFG["sohlh1"]["dbd_sequence"]
r_rag = float(cmp[cmp.prediction_mode == "RAG_LGO"]["r_vs_SOHLH2_JASPAR"])
r_eb  = float(cmp[cmp.prediction_mode == "RAG_LGO"]["r_vs_canonical_Ebox"])

# ── combined figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 8.4))
gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.7, 1.0, 1.1], hspace=0.65, wspace=0.32)

# (a) selection cascade
axa = fig.add_subplot(gs[0, 0]); axa.axis("off"); axa.set_title("a", loc="left", fontweight="bold")
steps = ["Human TFs\n(~1,639; Lambert 2018)", "Sequence-specific\nDNA binders",
         "No curated motif\n(orphan TFs)", "Orphan bHLH\nfamily", "SOHLH1\n(germ-cell bHLH)"]
cols = ["#cdd7e6", "#a9c0dd", "#86a8d4", "#5f88c4", "#c0504d"]
y = 0.92
for i, (s, c) in enumerate(zip(steps, cols)):
    w = 0.96 - i * 0.08
    axa.add_patch(FancyBboxPatch(((1 - w) / 2, y - 0.13), w, 0.12,
                  boxstyle="round,pad=0.005", fc=c, ec="none", transform=axa.transAxes))
    axa.text(0.5, y - 0.07, s, ha="center", va="center", fontsize=6.0,
             color=("white" if i == 4 else "black"), transform=axa.transAxes)
    if i < 4:
        axa.annotate("", xy=(0.5, y - 0.155), xytext=(0.5, y - 0.13),
                     arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.4"), transform=axa.transAxes)
    y -= 0.185

# (b) domain schematic
axb = fig.add_subplot(gs[0, 1]); axb.set_title("b", loc="left", fontweight="bold")
axb.set_xlim(0, CFG["sohlh1"]["full_length"]); axb.set_ylim(0, 1); axb.axis("off")
axb.add_patch(Rectangle((0, 0.45), CFG["sohlh1"]["full_length"], 0.12, fc="0.85", ec="0.4", lw=0.6))
d0, d1 = CFG["sohlh1"]["dbd_window_start"], CFG["sohlh1"]["dbd_window_end"]
axb.add_patch(Rectangle((d0, 0.40), d1 - d0, 0.22, fc="#5f88c4", ec="k", lw=0.6))
axb.text((d0 + d1) / 2, 0.73, "bHLH DBD\n(53–104)", ha="center", fontsize=6.5)
axb.text(0, 0.33, "1", fontsize=6); axb.text(CFG["sohlh1"]["full_length"], 0.33,
         str(CFG["sohlh1"]["full_length"]), ha="right", fontsize=6)
axb.text(0.5, 0.12, "SOHLH1 (Q5JUK2)\nno curated motif · no protein–DNA structure",
         ha="center", va="center", fontsize=6.3, transform=axb.transAxes)

# (c) predicted logos
axc1 = fig.add_subplot(gs[1, 0]); axc2 = fig.add_subplot(gs[1, 1])
logo(axc1, norag, "noRAG (de-novo)"); logo(axc2, rag, "RAG (leave-gene-out)")
axc1.text(-0.18, 1.30, "c", transform=axc1.transAxes, fontweight="bold", ha="right")
axc2.text(0.99, 1.30, f"conf={conf.confidence_score:.2f} ({conf.confidence_class}), IC={conf.motif_information_content*2:.2f} bits",
          transform=axc2.transAxes, fontsize=6, va="bottom", ha="right")

# (d) reference comparison
axd = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
logo(axd[0], rag, "SOHLH1 RAG prediction")
logo(axd[1], s2, "SOHLH2 paralog (JASPAR MA1560.1)")
axd[0].text(-0.18, 1.30, "d", transform=axd[0].transAxes, fontweight="bold", ha="right")
axd[0].text(0.99, 1.30, f"r$_{{SOHLH2}}$={r_rag:.2f}, r$_{{E-box}}$={r_eb:.2f}",
            transform=axd[0].transAxes, fontsize=6, va="bottom", ha="right")

# (e) attention over DBD residues
axe = fig.add_subplot(gs[3, :]); axe.set_title("e", loc="left", fontweight="bold")
A = attn[:rag.shape[1], :]            # active motif positions x DBD residues
im = axe.imshow(A, aspect="auto", cmap="magma", interpolation="nearest")
axe.set_yticks(range(A.shape[0])); axe.set_yticklabels([f"m{j+1}" for j in range(A.shape[0])], fontsize=6)
axe.set_xticks(range(len(seq1))); axe.set_xticklabels(list(seq1), fontsize=4.5)
axe.set_xlabel("SOHLH1 bHLH DBD residue", fontsize=7)
axe.set_ylabel("motif pos", fontsize=7)
# mark the basic region (DNA-contacting) — first ~15 residues of bHLH
axe.add_patch(Rectangle((-0.5, -0.5), 16, A.shape[0], fill=False, ec="cyan", lw=1.0))
cb = fig.colorbar(im, ax=axe, fraction=0.025, pad=0.01); cb.ax.tick_params(labelsize=6)
cb.set_label("attention", fontsize=6)

fig.suptitle("TFScope nominates an E-box specificity for the orphan germ-cell TF SOHLH1",
             fontsize=9, y=0.995)
fig.savefig(f"{FIG}/fig5_combined.pdf", bbox_inches="tight")
fig.savefig(f"{FIG}/fig5_combined.png", dpi=300, bbox_inches="tight")
print(f"wrote {FIG}/fig5_combined.pdf (+png)")

# ── individual panels (saved separately for layout flexibility) ─────────────────
def save_one(plotfn, name, figsize):
    f, a = plt.subplots(figsize=figsize); plotfn(a); f.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); plt.close(f)
save_one(lambda a: logo(a, norag, "SOHLH1 noRAG"), "fig5c_noRAG_logo", (3.0, 1.4))
save_one(lambda a: logo(a, rag,   "SOHLH1 RAG-LGO"), "fig5c_RAG_logo", (3.0, 1.4))
save_one(lambda a: logo(a, s2,    "SOHLH2 JASPAR"), "fig5d_SOHLH2_logo", (3.0, 1.4))
print("wrote individual logo panels")
