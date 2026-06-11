#!/usr/bin/env python
"""Generate all four case-study figures from cached arrays.

CS1  → figures/cs1_egr1_headtohead.pdf/.png
CS2  → figures/cs2_klf4_attention_repair.pdf/.png
CS3  → figures/cs3_family_frontier.pdf/.png  (Panel A already exists; regenerated here)
       figures/cs3_znf_examples.pdf/.png     (Panel B: ZNF76 vs ZNF649)
CS-RAG → figures/cs_rag_ablation.pdf/.png
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyArrowPatch
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

# ── logo helper ───────────────────────────────────────────────────────────────
import sys; sys.path.insert(0, "pwm_rosetta")
try:
    from pwm_hybrid.pwm.viz import makeLogo
    HAS_MAKELOGO = True
except ImportError:
    HAS_MAKELOGO = False

BASE_COLORS = {"A": "#2CA02C", "C": "#1F77B4", "G": "#FF7F0E", "T": "#D62728"}
BASES = list("ACGT")

def draw_logo(ax, pwm, L, title="", title_color="black", fontsize=8):
    """Draw a sequence logo on ax. pwm shape (4, L_full); uses first L cols."""
    ppm = np.clip(pwm[:, :L].T, 1e-8, 1.0)
    ppm /= ppm.sum(1, keepdims=True)
    if HAS_MAKELOGO:
        makeLogo(ppm, ax)
        ax.set_ylim(0, 2)
    else:
        # Manual IC-scaled stacked bar as fallback
        ic = 2 + (ppm * np.log2(ppm + 1e-9)).sum(1)
        bottom = np.zeros(L)
        order = np.argsort(ppm, axis=1)
        for pos in range(L):
            for b in order[pos]:
                h = ic[pos] * ppm[pos, b]
                ax.bar(pos, h, bottom=bottom[pos], color=BASE_COLORS[BASES[b]],
                       width=0.9, linewidth=0)
                bottom[pos] += h
        ax.set_xlim(-0.5, L - 0.5); ax.set_ylim(0, 2)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fontsize, color=title_color,
                     fontweight="bold", pad=2)

# ── load arrays ──────────────────────────────────────────────────────────────
d = np.load("results/case_study_arrays.npz", allow_pickle=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CS1 — Egr1 head-to-head (encoder vignette)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building CS1 …")
L = int(d["egr1_L"][0])
tgt   = d["egr1_truth"]
tfsc  = d["egr1_tfscope"]
dpbs  = d["egr1_deeppbs"]
r_t   = float(d["egr1_r_tfscope"][0])
r_d   = float(d["egr1_r_dpbs"][0])

fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.2))
panels = [
    (tgt,  L, "Experimental motif\n(Egr1 / 1a1g_A)",  "#2E7D32"),
    (tfsc, L, f"TFScope  r = {r_t:.3f}\n(sequence only)",          "#1565C0"),
    (dpbs, L, f"DeepPBS  r = {r_d:.3f}\n(crystal structure required)", "#C62828"),
]
for ax, (pwm, l, title, col) in zip(axes, panels):
    draw_logo(ax, pwm, l, title=title, title_color=col)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

# background shading to mark "no structure needed"
axes[0].set_facecolor("#F1F8E9"); axes[0].set_facecolor("#F1F8E9")
axes[1].set_facecolor("#E3F2FD")
axes[2].set_facecolor("#FFEBEE")

fig.suptitle("TFScope (sequence only) predicts the Egr1 binding motif better than\n"
             "structure-based DeepPBS on the same crystal complex",
             fontsize=10, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig("figures/cs1_egr1_headtohead.pdf", bbox_inches="tight")
fig.savefig("figures/cs1_egr1_headtohead.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  CS1 done  TFScope r={r_t:.3f}  DeepPBS r={r_d:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CS2 — KLF4 / MyoD attention repair
# ═══════════════════════════════════════════════════════════════════════════════
print("Building CS2 …")

CASES_META = {
    "KLF4": dict(mutsite=18, mut_label="K409", wt_aa="K", mut_aa="Q",
                 dbd_len=83, fam_label="C2H2 zinc finger"),
    "MyoD": dict(mutsite=11, mut_label="L122", wt_aa="L", mut_aa="R",
                 dbd_len=52, fam_label="bHLH"),
}

fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
MODELS = ["v17", "v18a"]
MODEL_LABELS = {
    "v17": "TFScope (−contact branch)\nrank-1 collapse: all columns attend the same residues",
    "v18a": "TFScope (contact branch active)\nattention spread; model reads the specificity-switching residue",
}
MODEL_COLORS = {"v17": "#B71C1C", "v18a": "#1565C0"}

for r, mname in enumerate(MODELS):
    for c, tf in enumerate(["KLF4", "MyoD"]):
        key  = f"cs2_{mname}_{tf}_attn"
        aW   = d[key]              # shape (20, dbd_len)
        meta = CASES_META[tf]
        ax   = axes[r, c]
        im   = ax.imshow(aW, aspect="auto", cmap="magma", vmin=0,
                         interpolation="nearest")

        # mark the causal residue
        ms = meta["mutsite"]
        ax.axvline(ms, color="cyan", ls="--", lw=1.4, zorder=5)
        ax.text(ms + 0.8, 0.5, f"{meta['wt_aa']}{meta['mut_label']}",
                color="cyan", fontsize=8.5, va="top", fontweight="bold")

        # row-constancy + entropy
        p    = aW / (aW.sum(1, keepdims=True) + 1e-9)
        C    = np.corrcoef(aW); iu = np.triu_indices(C.shape[0], 1)
        rc   = float(np.nanmean(C[iu]))
        ent  = float(-(p * np.log(p + 1e-12)).sum(1).mean())
        maxe = float(np.log(aW.shape[1]))
        mass = float(aW[:, ms].sum())

        stats = f"row-const={rc:.2f}  H={ent:.2f}/{maxe:.2f}  mass@causal={mass:.3f}"
        ax.set_title(f"{MODEL_LABELS[mname].split(chr(10))[0]}\n"
                     f"{tf} ({meta['fam_label']})  |  {stats}",
                     fontsize=8, color=MODEL_COLORS[mname], fontweight="bold")
        ax.set_xlabel("DBD residue position", fontsize=7)
        ax.set_ylabel("PWM column", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

# row titles
for r, mname in enumerate(MODELS):
    axes[r, 0].annotate(MODEL_LABELS[mname].split("\n")[0],
                        xy=(-0.18, 0.5), xycoords="axes fraction",
                        fontsize=9, color=MODEL_COLORS[mname],
                        fontweight="bold", rotation=90, va="center", ha="right")

fig.suptitle("TFScope cross-attention: degenerate collapse repaired by the contact branch\n"
             "Cyan dashed line = known specificity-determining residue",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0.04, 0, 1, 0.96])
fig.savefig("figures/cs2_klf4_attention_repair.pdf", bbox_inches="tight")
fig.savefig("figures/cs2_klf4_attention_repair.png", dpi=200, bbox_inches="tight")
plt.close()
print("  CS2 done")


# ═══════════════════════════════════════════════════════════════════════════════
# CS3 Panel A — Family frontier (box + strip)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building CS3-A …")
lfo_raw = json.load(open("results/lofo/per_tf_oracle_r.json"))
rows = [(f, np.array([x["oracle_r"] for x in lst], float))
        for f, lst in lfo_raw.items()]
rows.sort(key=lambda x: np.median(x[1]))
fams  = [r[0] for r in rows]; data = [r[1] for r in rows]
ns    = [len(r) for r in data]; overall_mean = np.concatenate(data).mean()
C2H2  = {"C2H2_long", "C2H2_medium", "C2H2_short"}
COL_LONG = "#9E2A00"; COL_C2H2 = "#D55E00"; COL_OTHER = "#0072B2"

fig, ax = plt.subplots(figsize=(7.5, 5.2))
ypos = np.arange(len(fams))
for i, rs in enumerate(data):
    is_long = fams[i] == "C2H2_long"
    is_c2h2 = fams[i] in C2H2
    face = COL_LONG if is_long else (COL_C2H2 if is_c2h2 else COL_OTHER)
    ax.boxplot(rs, positions=[i], vert=False, widths=0.62, showfliers=False,
               patch_artist=True, medianprops=dict(color="black", lw=1.6),
               whiskerprops=dict(color=face, lw=1.1),
               capprops=dict(color=face, lw=1.1),
               boxprops=dict(facecolor=face, edgecolor=face, alpha=0.30))
    jit = (np.random.RandomState(0).rand(len(rs)) - 0.5) * 0.42
    ax.scatter(rs, i + jit, s=4, color=face, alpha=0.35, linewidths=0, zorder=3)

ax.axvline(overall_mean, color="0.35", ls="--", lw=1.2)
ax.text(overall_mean + 0.01, len(fams) - 0.25, f"LFO floor\n(mean r = {overall_mean:.2f})",
        fontsize=8, color="0.30", va="top")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{f}  (n={n})" for f, n in zip(fams, ns)], fontsize=9)
ax.set_xlabel("Per-TF oracle Pearson r (leave-family-out)", fontsize=10)
ax.set_xlim(-0.45, 1.05); ax.axvline(0, color="0.8", lw=0.8, zorder=0)
ax.set_title("Family-conditioned TFScope: LFO transfer to an unseen family\n"
             "Long C2H2 zinc-finger arrays show the widest range — the prediction frontier",
             fontsize=10.5, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
leg = [Patch(facecolor=COL_LONG, alpha=0.55, label="C2H2 long array (frontier)"),
       Patch(facecolor=COL_C2H2, alpha=0.55, label="C2H2 medium / short"),
       Patch(facecolor=COL_OTHER, alpha=0.55, label="single-consensus families")]
ax.legend(handles=leg, fontsize=8, loc="lower right", frameon=False,
          bbox_to_anchor=(1.0, 0.01))

# annotate best/worst ZNF
long_lst = lfo_raw["C2H2_long"]
best  = max(long_lst, key=lambda x: x["oracle_r"])
worst = min(long_lst, key=lambda x: x["oracle_r"])
yl    = fams.index("C2H2_long")
def gene(fn): return fn.split("_", 2)[-1].split(".")[0] if "__" not in fn else fn.split("__")[1].split(".")[0]
ax.annotate(f"{gene(best['fn'])}\nr={best['oracle_r']:.2f}",
            xy=(best["oracle_r"], yl), xytext=(0.74, yl + 1.3),
            fontsize=7.5, color=COL_LONG, ha="center",
            arrowprops=dict(arrowstyle="->", color=COL_LONG, lw=0.9))
ax.annotate(f"{gene(worst['fn'])}\nr={worst['oracle_r']:.2f}",
            xy=(worst["oracle_r"], yl), xytext=(-0.22, yl + 1.3),
            fontsize=7.5, color=COL_LONG, ha="center",
            arrowprops=dict(arrowstyle="->", color=COL_LONG, lw=0.9))
fig.tight_layout()
fig.savefig("figures/cs3_family_frontier.pdf", bbox_inches="tight")
fig.savefig("figures/cs3_family_frontier.png", dpi=200, bbox_inches="tight")
plt.close()
print("  CS3-A done")

# ═══════════════════════════════════════════════════════════════════════════════
# CS3 Panel B — ZNF76 vs ZNF649 example logos
# ═══════════════════════════════════════════════════════════════════════════════
print("Building CS3-B …")
fig, axes = plt.subplots(2, 2, figsize=(8, 3.8))
for row, label in enumerate(["ZNF76", "ZNF649"]):
    r  = float(d[f"cs3_{label}_r"][0])
    L  = int(d[f"cs3_{label}_L"][0])
    tr = d[f"cs3_{label}_truth"]
    pr = d[f"cs3_{label}_pred"]
    col_t = "#2E7D32"; col_p = "#1565C0" if r > 0.5 else "#C62828"
    verdict = "✓ predicted well" if r > 0.5 else "✗ prediction fails"
    draw_logo(axes[row, 0], tr, L, title=f"{label}  truth", title_color=col_t)
    draw_logo(axes[row, 1], pr, L,
              title=f"{label}  TFScope  r = {r:.3f}  ({verdict})",
              title_color=col_p)
    for ax in axes[row]: ax.spines[["top","right","left","bottom"]].set_visible(False)

axes[0, 0].set_ylabel("ZNF76\n(ZF1 R→RDER, short)", fontsize=8, labelpad=4)
axes[1, 0].set_ylabel("ZNF649\n(long tandem array)", fontsize=8, labelpad=4)
fig.suptitle("Same C2H2_long family — opposite LFO outcomes\n"
             "Simple 2-finger recognition (ZNF76) transfers; "
             "complex long-array grammar (ZNF649) does not",
             fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig("figures/cs3_znf_examples.pdf", bbox_inches="tight")
fig.savefig("figures/cs3_znf_examples.png", dpi=200, bbox_inches="tight")
plt.close()
print("  CS3-B done")


# ═══════════════════════════════════════════════════════════════════════════════
# CS-RAG — Retrieval ablation figure
# ═══════════════════════════════════════════════════════════════════════════════
print("Building CS-RAG …")
panel = json.load(open("results/full_metrics/panel.json"))

metrics_show = ["r", "top1", "mcc", "auc", "mae", "ce"]
metric_labels = {
    "r":    "Mean Pearson r",
    "top1": "Top-1 accuracy",
    "mcc":  "MCC",
    "auc":  "AUC",
    "mae":  "MAE ↓",
    "ce":   "Cross-entropy ↓",
}
better_high = {"r", "top1", "mcc", "auc", "icr", "f1"}

models    = ["v18a_noRAG", "v18a", "DeepPBS"]
labels    = ["TFScope\n(−RAG)", "TFScope\n(+RAG)", "DeepPBS\n(+structure)"]
colors    = ["#78909C", "#1565C0", "#C62828"]

x = np.arange(len(metrics_show)); width = 0.24
fig, ax = plt.subplots(figsize=(9.5, 4.0))
for i, (mname, label, color) in enumerate(zip(models, labels, colors)):
    vals = [panel[mname][m] for m in metrics_show]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=label,
                  color=color, alpha=0.82, edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([metric_labels[m] for m in metrics_show], fontsize=9)
ax.set_ylabel("Metric value", fontsize=10)
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Retrieval augmentation (RAG) drives TFScope above the structure-based baseline\n"
             "Without RAG, TFScope ties DeepPBS; with RAG it exceeds on 4/6 metrics shown",
             fontsize=10.5, fontweight="bold")

# delta annotations on the r bar (most important)
ri = metrics_show.index("r")
r_norag = panel["v18a_noRAG"]["r"]; r_rag = panel["v18a"]["r"]; r_dpbs = panel["DeepPBS"]["r"]
ax.annotate(f"+{r_rag - r_norag:.3f}", xy=(ri + 0 * width, r_rag),
            xytext=(ri + 0 * width, r_rag + 0.025),
            fontsize=8, ha="center", color="#1565C0",
            arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.8))

fig.tight_layout()
fig.savefig("figures/cs_rag_ablation.pdf", bbox_inches="tight")
fig.savefig("figures/cs_rag_ablation.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  CS-RAG done  v18a_noRAG r={r_norag:.3f}  v18a r={r_rag:.3f}  DeepPBS r={r_dpbs:.3f}")

print("\nAll case-study figures saved to figures/")
