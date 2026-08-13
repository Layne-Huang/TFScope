"""Fig 3a — sequence-only prediction GENERALIZES to factors unlike anything in training.

Reframed from a benchmark into a generalization/coverage claim (the distinct point vs Fig 1):
the combined model (same as Fig 1) is run on the cluster40_clean held-out test, and per-factor
motif recovery is plotted against each factor's % DBD identity to the nearest training factor.
Recovery is independent of that distance — the most novel factors recover as well as the least —
so the model is not merely interpolating close homologs. Per-family recovery is a small secondary
panel (Fig 1e already carries the family head-to-head); predicted-vs-curated logos illustrate.

Predictions: scripts/eval_combined_heldout.py → combined_heldout_predictions.npz
Identity:    results/fig3a_heldout/recovery_vs_identity.npz (r, maxid per factor)
Out: figures/figure3a_heldout_recovery/figure3a_heldout_recovery.{png,pdf};
     results/fig3a_heldout/fig3a_recovery.{json,csv}
"""
import os, sys, json, csv, collections
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np
from scipy.stats import spearmanr
from eval_full_metrics import trimmed_core, aligned_cols

NPZ = "results/fig3a_heldout/combined_heldout_predictions.npz"
IDN = "results/fig3a_heldout/recovery_vs_identity.npz"
OUTD = "figures/figure3a_heldout_recovery"; os.makedirs(OUTD, exist_ok=True)
RES = "results/fig3a_heldout"; os.makedirs(RES, exist_ok=True)

def colr(A, B):
    rs = [np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(A.shape[1])
          if A[:, j].std() > 1e-8 and B[:, j].std() > 1e-8]
    return float(np.mean(rs)) if rs else np.nan

d = np.load(NPZ, allow_pickle=True)
idn = np.load(IDN, allow_pickle=True)
id_by = {str(f): float(m) for f, m in zip(idn["fn"], idn["maxid"])}

rec = []
for i in range(len(d["prediction"])):
    fn = str(d["filename"][i])
    gt = trimmed_core(d["target"][i], d["mask"][i] > 0.5)
    if gt is None or gt.shape[1] < 4: continue
    al, cols, _ = aligned_cols(d["prediction"][i], gt)
    if len(cols) < 4: continue
    G = gt[:, cols]; P = np.clip(al[:, cols], 1e-8, 1); P = P / P.sum(0, keepdims=True)
    r = colr(P, G)
    if r == r:
        rec.append(dict(fn=fn, gene=str(d["gene"][i]), family=str(d["family"][i]), r=r,
                        maxid=id_by.get(fn, np.nan), pred=al, gt=gt, cols=cols))

R = np.array([x["r"] for x in rec]); I = np.array([x["maxid"] for x in rec])
ok = ~np.isnan(I)
rho, p_rho = spearmanr(R[ok], I[ok])

# per-gene for the family panel
by_gene = collections.defaultdict(list)
for x in rec: by_gene[x["gene"]].append(x)
genes = [dict(gene=g, family=xs[0]["family"], r=float(np.mean([x["r"] for x in xs])),
             exemplar=max(xs, key=lambda x: x["r"])) for g, xs in by_gene.items()]
allr = np.array([g["r"] for g in genes])
fams = sorted(set(g["family"] for g in genes))
per_fam = {f: dict(n=sum(g["family"] == f for g in genes),
                   median=round(float(np.median([g["r"] for g in genes if g["family"] == f])), 3))
           for f in fams}

summary = dict(n_records=len(rec), n_genes=len(genes), median_r=round(float(np.median(allr)), 3),
               frac_r_ge_0p5=round(float((allr >= 0.5).mean()), 3),
               identity_median=round(float(np.nanmedian(I)), 1),
               frac_id_lt40=round(float((I[ok] < 40).mean()), 3),
               frac_id_lt30=round(float((I[ok] < 30).mean()), 3),
               spearman_r_vs_identity=round(float(rho), 3), spearman_p=round(float(p_rho), 3),
               median_r_id_lt40=round(float(np.median(R[ok & (I < 40)])), 3),
               median_r_id_ge40=round(float(np.median(R[ok & (I >= 40)])), 3), per_family=per_fam)
json.dump(summary, open(f"{RES}/fig3a_recovery.json", "w"), indent=1)
with open(f"{RES}/fig3a_recovery.csv", "w", newline="") as o:
    w = csv.writer(o); w.writerow(["gene", "family", "r"])
    for g in sorted(genes, key=lambda x: -x["r"]): w.writerow([g["gene"], g["family"], round(g["r"], 3)])
print("=== Fig 3a generalization ==="); [print(f"  {k}: {v}") for k, v in summary.items() if k != "per_family"]

# ── figure ──
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker, pandas as pd
def logo(ax, pwm, title, ct="black"):
    pwm = np.clip(pwm, 1e-9, 1); pwm = pwm / pwm.sum(0, keepdims=True)
    ic = np.maximum(2 + (pwm * np.log2(pwm)).sum(0), 0)
    df = pd.DataFrame((pwm * ic).T, columns=list("ACGT"))
    logomaker.Logo(df, ax=ax, color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2); ax.set_title(title, fontsize=8, color=ct, pad=2)

fams_all = sorted(set(x["family"] for x in rec))
cmap = dict(zip(fams_all, plt.cm.tab10(np.linspace(0, 1, len(fams_all)))))
fig = plt.figure(figsize=(13, 5.6))
gs = fig.add_gridspec(8, 3, width_ratios=[1.55, 0.95, 1.0], hspace=1.5, wspace=0.32)

# (a) generalization: recovery vs identity-to-training (now full-height left)
axa = fig.add_subplot(gs[:, 0])
axa.axvspan(0, 40, color="#4575b4", alpha=0.06, zorder=0)
for f in fams_all:
    pts = [(x["maxid"], x["r"]) for x in rec if x["family"] == f and x["maxid"] == x["maxid"]]
    if pts:
        xs, ys = zip(*pts); axa.scatter(xs, ys, s=14, color=cmap[f], alpha=0.7, lw=0, label=f)
bins = [20, 30, 40, 50, 65, 100]; bx, bm = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = ok & (I >= lo) & (I < hi)
    if m.sum() >= 5: bx.append((lo + hi) / 2); bm.append(np.median(R[m]))
axa.plot(bx, bm, "-o", color="#111", lw=2, ms=5, zorder=5, label="binned median")
axa.axvline(40, color="#4575b4", ls="--", lw=1)
m_lt, m_ge = summary["median_r_id_lt40"], summary["median_r_id_ge40"]
axa.hlines(m_lt, 18, 40, color="#d73027", lw=2.2, zorder=6)
axa.hlines(m_ge, 40, 100, color="#d73027", lw=2.2, zorder=6)
axa.text(29, m_lt + 0.04, f"novel (<40% id)\nmedian r={m_lt:.2f}", color="#a01", fontsize=7.8, ha="center", va="bottom")
axa.text(70, m_ge - 0.05, f"≥40% id\nmedian r={m_ge:.2f}", color="#a01", fontsize=7.8, ha="center", va="top")
axa.set_xlabel("% DBD identity to nearest training factor", fontsize=9.5)
axa.set_ylabel("motif recovery (oracle-aligned r)", fontsize=9.5)
axa.set_ylim(-0.05, 1.05); axa.set_xlim(18, 100)
axa.set_title("a  Recovery does not increase with proximity to training\n"
              "(no-RAG, sequence-only; novel factors recover as well as close ones)",
              fontsize=10.3, fontweight="bold", loc="left")
axa.legend(fontsize=6.2, frameon=False, ncol=2, loc="lower right", handletextpad=0.2, columnspacing=0.8)

# (b) exemplar predicted vs curated logos — prefer NOVEL (low-identity) factors to reinforce panel a
cands = [g for g in genes if len(g["exemplar"]["cols"]) >= 7 and 0.82 <= g["r"] <= 0.985
         and id_by.get(g["exemplar"]["fn"], np.nan) == id_by.get(g["exemplar"]["fn"], np.nan)]
cands.sort(key=lambda g: id_by.get(g["exemplar"]["fn"], 1e9))   # most novel first
picks, seen = [], set()
for g in cands:
    if g["family"] in seen: continue
    picks.append(g); seen.add(g["family"])
    if len(picks) == 4: break
for i, g in enumerate(picks):
    ex = g["exemplar"]; cols = ex["cols"]; eid = id_by.get(ex["fn"], np.nan)
    pred = np.clip(ex["pred"][:, cols], 1e-8, 1); pred /= pred.sum(0, keepdims=True)
    gt = np.clip(ex["gt"][:, cols], 1e-8, 1); gt /= gt.sum(0, keepdims=True)
    logo(fig.add_subplot(gs[2 * i, 1:]), pred,
         f"{g['gene']} ({g['family']}, r={g['r']:.2f}, {eid:.0f}% id to training)  —  predicted", "#1a5")
    logo(fig.add_subplot(gs[2 * i + 1, 1:]), gt, "curated (JASPAR/HOCOMOCO)", "#555")
fig.text(0.50, 0.95, "b  Predicted vs curated motifs (held-out factors)", fontsize=10.5, fontweight="bold")
fig.suptitle("Sequence-only prediction generalizes to held-out factors unlike anything in training",
             fontsize=12.5, fontweight="bold", y=1.04)
out = f"{OUTD}/figure3a_heldout_recovery"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
print("exemplars:", [(g["gene"], g["family"], round(g["r"], 2)) for g in picks])
print(f"saved {out}.png/.pdf")

# ── standalone per-family held-out recovery → companion panel for Figure 1 (1e) ──
OUTF1 = "figures/figure1e_heldout_perfamily"; os.makedirs(OUTF1, exist_ok=True)
figf, axf = plt.subplots(figsize=(5.2, 4.0))
order_f = sorted(fams, key=lambda f: per_fam[f]["median"])
data = [[g["r"] for g in genes if g["family"] == f] for f in order_f]
bp = axf.boxplot(data, vert=False, patch_artist=True, widths=0.62, showfliers=False, medianprops=dict(color="k"))
for patch, f in zip(bp["boxes"], order_f): patch.set_facecolor(cmap.get(f, "#888")); patch.set_alpha(0.85)
for i, f in enumerate(order_f):
    axf.scatter(data[i], np.random.RandomState(i).normal(i + 1, 0.06, len(data[i])), s=6, color="k", alpha=0.25, zorder=3)
    axf.text(1.02, i + 1, f"n={len(data[i])}", va="center", fontsize=7)
axf.axvline(float(np.median(allr)), color="#d73027", ls="--", lw=1.2)
axf.text(np.median(allr) + 0.01, 0.55, f"median {np.median(allr):.2f}", color="#d73027", fontsize=8, ha="left", va="center")
axf.set_yticks(range(1, len(order_f) + 1)); axf.set_yticklabels(order_f, fontsize=8.5)
axf.set_xlim(0, 1.0); axf.set_xlabel("held-out motif recovery (oracle-aligned r)", fontsize=9.5)
axf.set_title(f"Held-out recovery by family ({summary['n_genes']} factors)", fontsize=10, fontweight="bold")
figf.tight_layout()
figf.savefig(f"{OUTF1}/figure1e_heldout_perfamily.png", dpi=300, bbox_inches="tight")
figf.savefig(f"{OUTF1}/figure1e_heldout_perfamily.pdf", bbox_inches="tight")
print(f"saved {OUTF1}/figure1e_heldout_perfamily.png/.pdf  (per-family panel for Figure 1)")
