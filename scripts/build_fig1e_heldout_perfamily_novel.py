#!/usr/bin/env python
"""Fig 1e companion — held-out per-family recovery restricted to TRULY NOVEL factors.

Identical predictions + metric to build_fig3a_heldout_recovery.py, but keeps only records
whose TF is ABSENT from the combined model's training set (tf_name-level, not filename-level).
The default Fig 1e "190 factors" is held out only at the FILENAME level: 66.8% of its records
share a TF with training and 65.3% share the exact DBD sequence. This variant reports the
81 genes / 195 records that are genuinely unseen.

Out (NEW names, existing figure untouched):
  figures/figure1e_heldout_perfamily/figure1e_heldout_perfamily_novel81.{png,pdf}
  results/fig3a_heldout/fig1e_novel81.json
"""
import os, sys, json, collections
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from eval_full_metrics import trimmed_core, aligned_cols

NPZ="results/fig3a_heldout/combined_heldout_predictions.npz"
OUTF1="figures/figure1e_heldout_perfamily"; os.makedirs(OUTF1,exist_ok=True)
RES="results/fig3a_heldout"

def colr(A,B):
    rs=[np.corrcoef(A[:,j],B[:,j])[0,1] for j in range(A.shape[1])
        if A[:,j].std()>1e-8 and B[:,j].std()>1e-8]
    return float(np.mean(rs)) if rs else np.nan

# TF names present in the combined model's TRAINING split
comb=pd.read_parquet("data/processed/tf_pwm_combined_fm_deeppbs.parquet",columns=["filename","tf_name","sequence"])
tr=set(json.load(open("data/processed/splits/combined_fm_deeppbs/split.json"))["train"])
train_tf=set(comb[comb.filename.isin(tr)].tf_name)
train_seq=set(comb[comb.filename.isin(tr)].sequence)
# filename -> tf_name / sequence for the held-out records (from the aug_dbd parquet they came from)
aug=pd.read_parquet("data/processed/tf_pwm_aug_dbd.parquet",columns=["filename","tf_name","sequence"])
tf_by=dict(zip(aug.filename,aug.tf_name)); seq_by=dict(zip(aug.filename,aug.sequence))

d=np.load(NPZ,allow_pickle=True)
rec=[]; n_seen_tf=0
for i in range(len(d["prediction"])):
    fn=str(d["filename"][i])
    tf=tf_by.get(fn); sq=seq_by.get(fn)
    novel = (tf not in train_tf) and (sq not in train_seq)   # unseen TF AND unseen DBD sequence
    if not novel:
        n_seen_tf+=1; continue
    gt=trimmed_core(d["target"][i], d["mask"][i]>0.5)
    if gt is None or gt.shape[1]<4: continue
    al,cols,_=aligned_cols(d["prediction"][i],gt)
    if len(cols)<4: continue
    G=gt[:,cols]; P=np.clip(al[:,cols],1e-8,1); P=P/P.sum(0,keepdims=True)
    r=colr(P,G)
    if r==r: rec.append(dict(gene=str(d["gene"][i]),family=str(d["family"][i]),r=r))

# per-RECORD (every sample shown; no per-gene averaging)
allr=np.array([x["r"] for x in rec])
fams=sorted(set(x["family"] for x in rec))
per_fam={f:dict(n=sum(x["family"]==f for x in rec),
                median=round(float(np.median([x["r"] for x in rec if x["family"]==f])),3)) for f in fams}
summary=dict(scope="novel-only per-RECORD (TF name & DBD sequence absent from combined training)",
             n_records=len(rec),n_genes=len({x["gene"] for x in rec}),
             median_r=round(float(np.median(allr)),3),frac_r_ge_0p5=round(float((allr>=0.5).mean()),3),
             excluded_seen_records=n_seen_tf,per_family=per_fam)
json.dump(summary,open(f"{RES}/fig1e_novel81_persample.json","w"),indent=1)
print("=== Fig 1e NOVEL-only ==="); [print(f"  {k}: {v}") for k,v in summary.items() if k!="per_family"]
for f in fams: print(f"    {f:18s} n={per_fam[f]['n']:2d}  median r={per_fam[f]['median']}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cmap=dict(zip(fams,plt.cm.tab10(np.linspace(0,1,len(fams)))))
figf,axf=plt.subplots(figsize=(5.2,4.0))
order_f=sorted(fams,key=lambda f:per_fam[f]["median"])
data=[[x["r"] for x in rec if x["family"]==f] for f in order_f]
bp=axf.boxplot(data,vert=False,patch_artist=True,widths=0.62,showfliers=False,medianprops=dict(color="k"))
for patch,f in zip(bp["boxes"],order_f): patch.set_facecolor(cmap.get(f,"#888")); patch.set_alpha(0.85)
for i,f in enumerate(order_f):
    axf.scatter(data[i],np.random.RandomState(i).normal(i+1,0.06,len(data[i])),s=8,color="k",alpha=0.3,zorder=3)
    axf.text(1.02,i+1,f"n={len(data[i])}",va="center",fontsize=7)
axf.axvline(float(np.median(allr)),color="#d73027",ls="--",lw=1.2)
axf.text(np.median(allr)+0.01,0.55,f"median {np.median(allr):.2f}",color="#d73027",fontsize=8,ha="left",va="center")
axf.set_yticks(range(1,len(order_f)+1)); axf.set_yticklabels(order_f,fontsize=8.5)
axf.set_xlim(0,1.0); axf.set_xlabel("held-out motif recovery (oracle-aligned r)",fontsize=9.5)
axf.set_title(f"Held-out recovery by family — novel samples ({summary['n_records']} samples, {summary['n_genes']} factors)",
              fontsize=9.8,fontweight="bold")
figf.tight_layout()
figf.savefig(f"{OUTF1}/figure1e_heldout_perfamily_novel81_persample.png",dpi=300,bbox_inches="tight")
figf.savefig(f"{OUTF1}/figure1e_heldout_perfamily_novel81_persample.pdf",bbox_inches="tight")
print(f"\nsaved {OUTF1}/figure1e_heldout_perfamily_novel81_persample.png/.pdf")
