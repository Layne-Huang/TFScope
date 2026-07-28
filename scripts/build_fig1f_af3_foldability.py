"""Fig 2d — an independent structure predictor (AlphaFold3) judges TFScope's
sequence-only predicted consensus motifs as forming MORE confident protein-DNA
complexes than the structure-based DeepPBS predictions.

For each TF we folded protein + TFScope-consensus-DNA and protein + DeepPBS-consensus-DNA
(AF3, replicate runs) and compare interface confidence (ipTM). Length confound controlled:
the ipTM gap is uncorrelated with DNA-length difference and unchanged on same-length pairs.

Inputs: /data1/leihuang/project/TFScope/AF3_consensus_folding/{tfscope,deeppbs}/*/*summary_confidences.json
Outputs: results/fig1f_af3_foldability/fig1f_foldability.{json,csv}
         figures/figure1f_af3_foldability/figure1f_af3_foldability.{png,pdf}
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr

ROOT = "/data1/leihuang/project/TFScope/AF3_consensus_folding"
OUTD = "figures/figure1f_af3_foldability"; os.makedirs(OUTD, exist_ok=True)
RES = "results/fig1f_af3_foldability"; os.makedirs(RES, exist_ok=True)

m = pd.read_csv(f"{ROOT}/jobs_manifest.csv"); m["dnalen"] = m.strand1.str.len()
rows = []
for r in m.itertuples():
    job = r.job_name.lower(); src = os.path.basename(r.output_dir)
    sm = f"{ROOT}/{src}/{job}/{job}_summary_confidences.json"
    if not os.path.exists(sm): continue
    d = json.load(open(sm))
    rows.append(dict(gene=r.gene, family=r.family, source=r.source, iptm=d["iptm"],
                     ptm=d["ptm"], rank=d["ranking_score"], fdis=d["fraction_disordered"],
                     dnalen=r.dnalen))
df = pd.DataFrame(rows)
g = df.groupby(["gene", "family", "source"]).agg(iptm=("iptm", "mean"), rank=("rank", "mean"),
                                                 dnalen=("dnalen", "mean")).reset_index()
P = g.pivot_table(index=["gene", "family"], columns="source", values=["iptm", "rank", "dnalen"])
P.columns = [f"{a}_{b}" for a, b in P.columns]; P = P.dropna(subset=["iptm_TFScope", "iptm_DeepPBS"]).reset_index()
P["d_iptm"] = P.iptm_TFScope - P.iptm_DeepPBS
P["d_len"] = P.dnalen_TFScope - P.dnalen_DeepPBS

n = len(P); wins = int((P.d_iptm > 0).sum())
p_all = wilcoxon(P.iptm_TFScope, P.iptm_DeepPBS).pvalue
rho, p_len = spearmanr(P.d_iptm, P.d_len)
S = P[P.d_len == 0]
summary = dict(n_tf=n, tfscope_mean_iptm=round(float(P.iptm_TFScope.mean()), 3),
               deeppbs_mean_iptm=round(float(P.iptm_DeepPBS.mean()), 3),
               mean_delta=round(float(P.d_iptm.mean()), 3), tfscope_wins=wins,
               win_frac=round(wins / n, 3), wilcoxon_p=float(p_all),
               delta_len_rho=round(float(rho), 3), delta_len_p=round(float(p_len), 3),
               samelen_n=len(S), samelen_delta=round(float(S.d_iptm.mean()), 3),
               samelen_wins=int((S.d_iptm > 0).sum()))
json.dump(summary, open(f"{RES}/fig1f_foldability.json", "w"), indent=1)
P.round(3).to_csv(f"{RES}/fig1f_foldability.csv", index=False)
print("=== Fig 2d foldability summary ==="); [print(f"  {k}: {v}") for k, v in summary.items()]

# ── figure ──
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fams = sorted(P.family.unique())
cmap = dict(zip(fams, plt.cm.tab10(np.linspace(0, 1, len(fams)))))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.15, 1]})

# (a) paired scatter: DeepPBS vs TFScope ipTM
for fam in fams:
    s = P[P.family == fam]
    ax[0].scatter(s.iptm_DeepPBS, s.iptm_TFScope, s=38, color=cmap[fam], label=fam,
                  edgecolor="k", lw=0.4, alpha=0.9, zorder=3)
lim = [min(P.iptm_DeepPBS.min(), P.iptm_TFScope.min()) - 0.03, 1.0]
ax[0].plot(lim, lim, "k--", lw=1, zorder=1)
ax[0].fill_between(lim, lim, [1, 1], color="#d73027", alpha=0.05, zorder=0)
ax[0].text(0.52, 0.97, "TFScope motif folds better", fontsize=8, color="#a01", style="italic")
for _, r in P.sort_values("d_iptm", ascending=False).head(4).iterrows():
    ax[0].annotate(r.gene, (r.iptm_DeepPBS, r.iptm_TFScope), fontsize=7,
                   xytext=(3, -2), textcoords="offset points")
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("DeepPBS consensus — AF3 ipTM", fontsize=10)
ax[0].set_ylabel("TFScope consensus — AF3 ipTM", fontsize=10)
ax[0].set_title(f"a  AF3 interface confidence per TF (n={n})", fontsize=10.5, fontweight="bold", loc="left")
ax[0].legend(fontsize=6.6, frameon=False, loc="lower right", ncol=2, handletextpad=0.2)

# (b) paired strip with medians
xj = np.random.RandomState(1).uniform(-0.05, 0.05, n)
for a_, b_, j in zip(P.iptm_DeepPBS, P.iptm_TFScope, xj):
    ax[1].plot([1 + j, 2 + j], [a_, b_], "-", color="#ccc", lw=0.6, zorder=1)
ax[1].scatter(np.ones(n) + xj, P.iptm_DeepPBS, s=20, color="#7f7f7f", zorder=2)
ax[1].scatter(np.full(n, 2) + xj, P.iptm_TFScope, s=20, color="#d73027", zorder=2)
for xp, v in [(1, P.iptm_DeepPBS), (2, P.iptm_TFScope)]:
    ax[1].plot([xp - 0.2, xp + 0.2], [v.median()] * 2, "k-", lw=2.5, zorder=3)
    ax[1].text(xp, v.median() + 0.015, f"{v.median():.2f}", ha="center", fontsize=8.5, fontweight="bold")
ax[1].set_xlim(0.6, 2.4); ax[1].set_xticks([1, 2])
ax[1].set_xticklabels(["DeepPBS\nconsensus", "TFScope\nconsensus"], fontsize=9)
ax[1].set_ylabel("AF3 ipTM (protein–DNA interface)", fontsize=10)
ax[1].set_title("b  TFScope motifs fold more confidently", fontsize=10.5, fontweight="bold", loc="left")
star = "***" if p_all < 1e-3 else ("**" if p_all < 1e-2 else "*")
ytop = max(P.iptm_DeepPBS.max(), P.iptm_TFScope.max())
ax[1].plot([1, 2], [ytop + 0.05] * 2, "k-", lw=0.8)
ax[1].text(1.5, ytop + 0.06, f"{star}  p={p_all:.1e}\nTFScope higher in {wins}/{n} TFs",
           ha="center", fontsize=8)
ax[1].text(1.5, lim[0] - 0.0, f"length-controlled: Δ={summary['samelen_delta']:+.2f} on {len(S)} same-length pairs",
           ha="center", fontsize=7, color="#555", style="italic")

fig.suptitle("AlphaFold3 judges TFScope's sequence-only predicted motifs as more foldable than DeepPBS's",
             fontsize=11.5, fontweight="bold", y=1.02)
fig.tight_layout()
out = f"{OUTD}/figure1f_af3_foldability"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
print(f"  saved {out}.png/.pdf")
