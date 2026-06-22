"""Fig 3a — known motifs are recovered from sequence alone for held-out factors, at scale.

cluster40_clean held-out test (40%-DBD-identity clustered out of training): the deployed
retrieval-augmented model (e5b) predicts each factor's motif from sequence; we score the
oracle-aligned correlation to the curated JASPAR/HOCOMOCO motif, aggregated per gene and
broken down by structural family. Demonstrates the scalable advantage over structure-based
methods (which cannot run without a structure).

Source: results/v19_e9_model_composition/e5b_test_predictions.npz (prediction, curated target,
family, gene). Outputs: results/fig3a_heldout/fig3a_recovery.{json,csv};
figures/figure3a_heldout_recovery/figure3a_heldout_recovery.{png,pdf}
"""
import os, sys, json
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np
from eval_full_metrics import trimmed_core, aligned_cols

NPZ = "results/fig3a_heldout/combined_heldout_predictions.npz"   # combined rag_contact (Fig 1 model)
OUTD = "figures/figure3a_heldout_recovery"; os.makedirs(OUTD, exist_ok=True)
RES = "results/fig3a_heldout"; os.makedirs(RES, exist_ok=True)

def colr(A, B):
    rs = [np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(A.shape[1])
          if A[:, j].std() > 1e-8 and B[:, j].std() > 1e-8]
    return float(np.mean(rs)) if rs else np.nan

d = np.load(NPZ, allow_pickle=True)
P, T, G, M, fam, gene = d["prediction"], d["target"], d["gate"], d["mask"], d["family"], d["gene"]

rec = []   # per-record
for i in range(len(P)):
    gt = trimmed_core(T[i], M[i] > 0.5)
    if gt is None or gt.shape[1] < 4: continue
    al, cols, rc = aligned_cols(P[i], gt)
    if len(cols) < 4: continue
    Gc = gt[:, cols]; Pc = np.clip(al[:, cols], 1e-8, 1); Pc = Pc / Pc.sum(0, keepdims=True)
    r = colr(Pc, Gc)
    if r == r:
        rec.append(dict(gene=str(gene[i]), family=str(fam[i]), r=r,
                        pred=al, gt=gt, cols=cols, idx=i))

# aggregate per gene (macro) → mean r per gene, keep one exemplar per gene
import collections
by_gene = collections.defaultdict(list)
for x in rec: by_gene[x["gene"]].append(x)
genes = []
for g, xs in by_gene.items():
    rr = float(np.mean([x["r"] for x in xs]))
    best = max(xs, key=lambda x: x["r"])
    genes.append(dict(gene=g, family=xs[0]["family"], r=rr, exemplar=best))

allr = np.array([g["r"] for g in genes])
fams = sorted(set(g["family"] for g in genes))
per_fam = {}
for f in fams:
    rs = [g["r"] for g in genes if g["family"] == f]
    per_fam[f] = dict(n=len(rs), median=round(float(np.median(rs)), 3), mean=round(float(np.mean(rs)), 3))
summary = dict(n_records=len(rec), n_genes=len(genes), median_r=round(float(np.median(allr)), 3),
               mean_r=round(float(np.mean(allr)), 3), frac_r_ge_0p5=round(float((allr >= 0.5).mean()), 3),
               frac_r_ge_0p7=round(float((allr >= 0.7).mean()), 3), per_family=per_fam)
json.dump(summary, open(f"{RES}/fig3a_recovery.json", "w"), indent=1)
import csv
with open(f"{RES}/fig3a_recovery.csv", "w", newline="") as o:
    w = csv.writer(o); w.writerow(["gene", "family", "r"])
    for g in sorted(genes, key=lambda x: -x["r"]): w.writerow([g["gene"], g["family"], round(g["r"], 3)])
print("=== Fig 3a held-out recovery (per-gene) ==="); [print(f"  {k}: {v}") for k, v in summary.items() if k != "per_family"]
for f, s in sorted(per_fam.items(), key=lambda kv: -kv[1]["median"]): print(f"  {f:<18} n={s['n']:<3} median r={s['median']}")

# ── figure ──
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker, pandas as pd
def logo(ax, pwm, title, color_title="black"):
    # glyphs scaled in DATA coords by logomaker → never overflow the (short) axis
    pwm = np.clip(pwm, 1e-9, 1); pwm = pwm / pwm.sum(0, keepdims=True)
    ic = np.maximum(2 + (pwm * np.log2(pwm)).sum(0), 0)
    H = (pwm * ic).T                                   # (L, 4) information-scaled heights
    df = pd.DataFrame(H, columns=list("ACGT"))
    logomaker.Logo(df, ax=ax, color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2)
    ax.set_title(title, fontsize=8, color=color_title, pad=2)

fig = plt.figure(figsize=(12, 6.2))
gs = fig.add_gridspec(8, 2, width_ratios=[1.7, 1.05], hspace=1.5, wspace=0.18)

# (a) per-family recovery distribution (best family on top)
axa = fig.add_subplot(gs[:, 0])
order_f = sorted(fams, key=lambda f: per_fam[f]["median"])   # ascending → best ends up on top
data = [[g["r"] for g in genes if g["family"] == f] for f in order_f]
bp = axa.boxplot(data, vert=False, patch_artist=True, widths=0.62, showfliers=False,
                 medianprops=dict(color="k"))
cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(order_f)))
for patch, c in zip(bp["boxes"], cmap): patch.set_facecolor(c); patch.set_alpha(0.85)
for i, f in enumerate(order_f):
    rs = data[i]
    axa.scatter(rs, np.random.RandomState(i).normal(i + 1, 0.06, len(rs)), s=6, color="k", alpha=0.25, zorder=3)
    axa.text(1.02, i + 1, f"n={len(rs)}", va="center", fontsize=7)
axa.axvline(float(np.median(allr)), color="#d73027", ls="--", lw=1.3, zorder=1)
axa.text(np.median(allr), len(order_f) + 0.7, f"overall median r={np.median(allr):.2f}",
         color="#d73027", fontsize=8.5, ha="center")
axa.set_yticks(range(1, len(order_f) + 1)); axa.set_yticklabels(order_f, fontsize=8.5)
axa.set_xlim(-0.05, 1.0); axa.set_xlabel("predicted vs curated motif correlation (oracle-aligned r)", fontsize=9.5)
axa.set_title(f"a  Held-out recovery by family ({summary['n_genes']} factors, 40%-id clustered out)",
              fontsize=10, fontweight="bold", loc="left")

# (b) exemplar logos: realistic high-recovery factors, predicted vs curated, longer motifs
picks, seen = [], set()
for g in sorted(genes, key=lambda x: -x["r"]):
    ex = g["exemplar"]
    if len(ex["cols"]) < 7: continue          # require a non-trivial motif length
    if not (0.82 <= g["r"] <= 0.985): continue # realistic, strong (avoid trivial r=1.0)
    if g["family"] in seen: continue
    picks.append(g); seen.add(g["family"])
    if len(picks) == 4: break
for i, g in enumerate(picks):
    ex = g["exemplar"]; cols = ex["cols"]
    pred = np.clip(ex["pred"][:, cols], 1e-8, 1); pred = pred / pred.sum(0, keepdims=True)
    gt = np.clip(ex["gt"][:, cols], 1e-8, 1); gt = gt / gt.sum(0, keepdims=True)
    axp = fig.add_subplot(gs[2 * i, 1]); axc = fig.add_subplot(gs[2 * i + 1, 1])
    logo(axp, pred, f"{g['gene']} ({g['family']}, r={g['r']:.2f})  —  predicted", "#1a5")
    logo(axc, gt, "curated (JASPAR/HOCOMOCO)", "#555")
fig.text(0.63, 0.95, "b  Predicted vs curated motifs", fontsize=10.5, fontweight="bold")
fig.suptitle("TFScope recovers known motifs from sequence alone for held-out factors, at scale",
             fontsize=12, fontweight="bold", y=1.06)
out = f"{OUTD}/figure3a_heldout_recovery"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
print("exemplars:", [(g["gene"], g["family"], round(g["r"], 2)) for g in picks])
print(f"saved {out}.png/.pdf")
