"""Structure-less generalization panel: TFScope nominates DNA motifs for TFs that have
NO experimental structure (so a structure-based method like DeepPBS cannot run at all),
and AlphaFold3 confirms the predicted protein+consensus complexes fold with high
interface confidence (ipTM).

Inputs: /data1/leihuang/project/TFScope/structureless_af3_folding/*/*_summary_confidences.json
        results/structureless_af3_inputs/af3_sequences.json   (consensus + family)
Outputs: results/structureless_af3_inputs/structureless_foldability.{json,csv}
         figures/figure2d_structureless_foldability/structureless_foldability.{png,pdf}
"""
import os, json, glob
import numpy as np, pandas as pd

ROOT = "/data1/leihuang/project/TFScope/structureless_af3_folding"
OUTD = "figures/figure2d_structureless_foldability"; os.makedirs(OUTD, exist_ok=True)
RES = "results/structureless_af3_inputs"
meta = {r["gene"].upper(): r for r in json.load(open(f"{RES}/af3_sequences.json"))}

rows = []
for d in sorted(glob.glob(f"{ROOT}/structureless_*/")):
    gene = os.path.basename(d.rstrip("/")).split("_", 2)[-1].upper()
    sm = glob.glob(f"{d}*_summary_confidences.json")
    if not sm: continue
    j = json.load(open(sm[0])); mm = meta.get(gene, {})
    rows.append(dict(gene=gene, family=mm.get("family", "?"), oligomer=mm.get("oligomer", "?"),
                     consensus=mm.get("motif_core", mm.get("dna_top", "")), iptm=j["iptm"],
                     ptm=j["ptm"], rank=j["ranking_score"], frac_dis=j["fraction_disordered"]))
df = pd.DataFrame(rows).sort_values("iptm", ascending=True).reset_index(drop=True)
summ = dict(n=len(df), mean_iptm=round(float(df.iptm.mean()), 3), median_iptm=round(float(df.iptm.median()), 3),
            n_ge_080=int((df.iptm >= 0.8).sum()), n_ge_060=int((df.iptm >= 0.6).sum()))
df.round(3).to_csv(f"{RES}/structureless_foldability.csv", index=False)
json.dump(summ, open(f"{RES}/structureless_foldability.json", "w"), indent=1)
print("=== structure-less foldability ==="); [print(f"  {k}: {v}") for k, v in summ.items()]

# ── figure: horizontal ipTM bars per TF, coloured by family, motif annotated ──
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fams = sorted(df.family.unique())
cmap = dict(zip(fams, plt.cm.tab10(np.linspace(0, 1, len(fams)))))
fig, ax = plt.subplots(figsize=(8.2, 4.6))
y = np.arange(len(df))
ax.barh(y, df.iptm, color=[cmap[f] for f in df.family], edgecolor="k", lw=0.5)
ax.axvline(0.8, color="#444", ls="--", lw=1)
ax.text(0.8, len(df) - 0.3, " high-confidence (ipTM 0.8)", fontsize=7.5, color="#444", va="top")
for i, r in df.iterrows():
    ax.text(r.iptm - 0.01, i, f"{r.iptm:.2f}", va="center", ha="right", fontsize=7.5,
            color="white", fontweight="bold")
    ax.text(0.012, i, f"{r.gene}  ({r.consensus})", va="center", ha="left", fontsize=8,
            color="black")
ax.set_yticks([]); ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(df) - 0.4)
ax.set_xlabel("AlphaFold3 interface confidence, ipTM (protein–DNA)", fontsize=10)
ax.set_title(f"TFScope nominates foldable motifs for structure-less TFs "
             f"({summ['n_ge_080']}/{summ['n']} fold at ipTM≥0.8)\n"
             f"DeepPBS cannot predict these — it requires an input structure",
             fontsize=10.5, fontweight="bold")
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[f]) for f in fams]
ax.legend(handles, fams, fontsize=7.5, frameon=False, loc="lower right", title="family", title_fontsize=8)
fig.tight_layout()
out = f"{OUTD}/structureless_foldability"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
print(f"  saved {out}.png/.pdf")
