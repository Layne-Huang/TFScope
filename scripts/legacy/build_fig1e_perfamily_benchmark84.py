#!/usr/bin/env python
"""Fig 1e per-family recovery on the ACTUAL benchmark test set (deeppbs_cluster40, 84 rows / 41 TFs)
— the same set and metric as Fig 1d/2a/1f, instead of the mislabeled cluster40_clean/test (which
mixes in the combined model's own train+val). Leaked TFs (present in combined_fm_deeppbs train) are
drawn hollow so the clean subset is visible.
Metric: gate-active predicted core, align_pwm oracle-aligned r (RC, ±10 shift) to IC-trimmed GT core.
Out: figures/figure1e_heldout_perfamily/figure1e_perfamily_benchmark84.{png,pdf}
     results/fig3a_heldout/fig1e_benchmark84.{json,csv}
"""
import os,sys,json
os.environ["TORCH_HOME"]="/data1/leihuang/.cache/torch"; os.environ["HF_HOME"]="/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/case_study")
import numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
from cs_utils import load_model, active_cols, infer, device
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm

CK="/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
OUTF="figures/figure1e_heldout_perfamily"; RES="results/fig3a_heldout"; os.makedirs(RES,exist_ok=True)
comb=pd.read_parquet("data/processed/tf_pwm_combined_fm_deeppbs.parquet",columns=["filename","tf_name"])
train_tf=set(comb[comb.filename.isin(set(json.load(open("data/processed/splits/combined_fm_deeppbs/split.json"))["train"]))].tf_name)
dp=pd.read_parquet("data/processed/tf_pwm_deeppbs_only_canon_trim.parquet")
te=dp[dp.filename.isin(set(json.load(open("data/processed/splits/deeppbs_cluster40/split.json"))["test"]))].reset_index(drop=True)
def dec(v): return np.frombuffer(v,dtype=np.float32).reshape(4,-1).copy()
def ict(p,t=0.25):
    q=np.clip(p,1e-8,1);ic=2+(q*np.log2(q)).sum(0);i=np.where(ic>=t)[0]
    return p if len(i)==0 else p[:,i[0]:i[-1]+1]
m,_=load_model(CK,force_retrieval=False)
rows=[]
for _,r in te.iterrows():
    seq=r.sequence
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in seq]],dtype=torch.long,device=device)
    mask=torch.ones(1,len(seq),dtype=torch.bool,device=device)
    g,pp,_=infer(m,tok,mask,int(r.family_id),ret=None)
    pred=ict(pp[:,active_cols(g,0.5)]); tg=ict(dec(r.pwm))
    if pred.shape[1]==0 or tg.shape[1]==0: continue
    _,_,_,sc=align_pwm(pred,tg,max_shift=10,consider_revcomp=True)
    rows.append(dict(gene=r.gene_symbol,fam=r.family_name,leaked=r.tf_name in train_tf,r=float(sc)))
df=pd.DataFrame(rows)
clean=df[~df.leaked]
fams=sorted(df.fam.unique())
per_fam={f:dict(n=int((df.fam==f).sum()), n_clean=int(((df.fam==f)&~df.leaked).sum()),
                median=round(float(df[df.fam==f].r.median()),3),
                median_clean=(round(float(clean[clean.fam==f].r.median()),3) if ((clean.fam==f).any()) else None)) for f in fams}
summary=dict(set="deeppbs_cluster40 test (benchmark)", n=len(df), n_tf=te.tf_name.nunique(),
             n_leaked=int(df.leaked.sum()),
             median_all=round(float(df.r.median()),3), mean_all=round(float(df.r.mean()),3),
             median_clean=round(float(clean.r.median()),3), mean_clean=round(float(clean.r.mean()),3),
             per_family=per_fam)
json.dump(summary,open(f"{RES}/fig1e_benchmark84.json","w"),indent=1)
df.round(3).to_csv(f"{RES}/fig1e_benchmark84.csv",index=False)
print("=== Fig 1e on benchmark-84 ==="); [print(f"  {k}: {v}") for k,v in summary.items() if k!="per_family"]
for f in fams: print(f"    {f:18s} n={per_fam[f]['n']:2d} (clean {per_fam[f]['n_clean']:2d})  median {per_fam[f]['median']}  clean {per_fam[f]['median_clean']}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cmap=dict(zip(fams,plt.cm.tab10(np.linspace(0,1,len(fams)))))
order=sorted(fams,key=lambda f:per_fam[f]["median"])
data=[df[df.fam==f].r.values for f in order]
fig,ax=plt.subplots(figsize=(5.8,4.2))
bp=ax.boxplot(data,vert=False,patch_artist=True,widths=0.62,showfliers=False,medianprops=dict(color="k"))
for patch,f in zip(bp["boxes"],order): patch.set_facecolor(cmap[f]); patch.set_alpha(0.85)
rng=np.random.RandomState(0)
for i,f in enumerate(order):
    s=df[df.fam==f]
    y=rng.normal(i+1,0.07,len(s))
    ax.scatter(s.r.values, y, s=16, facecolor="k", edgecolor="k", lw=0.4, alpha=0.5, zorder=3)
    ax.text(1.02,i+1,f"n={len(s)}",va="center",fontsize=7.5)
ax.axvline(float(df.r.median()),color="#d73027",ls="--",lw=1.3)
ax.text(df.r.median()+0.01,0.45,f"median {df.r.median():.2f}",color="#d73027",fontsize=8)
ax.set_yticks(range(1,len(order)+1)); ax.set_yticklabels(order,fontsize=8.5)
ax.set_xlim(0,1.0); ax.set_xlabel("motif recovery (oracle-aligned r)",fontsize=9.5)
ax.set_title(f"Benchmark recovery by family (deeppbs_cluster40 test, {len(df)} samples / {te.tf_name.nunique()} TFs)",
             fontsize=9.3,fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUTF}/figure1e_perfamily_benchmark84.png",dpi=300,bbox_inches="tight")
fig.savefig(f"{OUTF}/figure1e_perfamily_benchmark84.pdf",bbox_inches="tight")
print(f"\nsaved {OUTF}/figure1e_perfamily_benchmark84.png/.pdf")
