#!/usr/bin/env python
"""Rebuilt Fig 3a on a PROPER held-out pool (TF name & DBD seq NOT in combined-train),
identity measured against the model's ACTUAL training (combined_fm_deeppbs).
Panel a: recovery vs %identity (family colours + binned median).
Panel b: predicted vs curated logos for novel held-out exemplars (tall, aligned).
Out: figures/figure3a_heldout_recovery/figure3a_heldout_clean.{png,pdf}
"""
import os,sys,json
os.environ["TORCH_HOME"]="/data1/leihuang/.cache/torch"; os.environ["HF_HOME"]="/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/case_study")
import numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
from cs_utils import load_model, active_cols, infer, device
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
CK="/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
OUTD="figures/figure3a_heldout_recovery"; RES="results/fig3a_heldout"; os.makedirs(OUTD,exist_ok=True)
FID={"C2H2_short":0,"C2H2_medium":1,"C2H2_long":2,"bHLH":3,"Homeodomain":4,"bZIP":5,
     "Nuclear_Receptor":6,"Forkhead":7,"ETS":8,"Other":9}
comb=pd.read_parquet("data/processed/tf_pwm_combined_fm_deeppbs.parquet",columns=["filename","tf_name","sequence"])
ctr=set(json.load(open("data/processed/splits/combined_fm_deeppbs/split.json"))["train"])
train=comb[comb.filename.isin(ctr)]; train_tf=set(train.tf_name); train_seqset=set(train.sequence)
aug=pd.read_parquet("data/processed/tf_pwm_aug_dbd.parquet").drop_duplicates("sequence")
held=aug[(~aug.tf_name.isin(train_tf)) & (~aug.sequence.isin(train_seqset))].copy()
held=held.merge(pd.read_parquet("/tmp/claude-27813/-afs-csail-mit-edu-u-l-leihuang-project-TFScope/fdcc1f98-59bc-4c28-84e4-4be69cca02a0/scratchpad/heldout_pool.parquet")[["sequence","mid"]],on="sequence",how="inner")
print(f"pool {len(held)}")
def dec(v): return np.frombuffer(v,dtype=np.float32).reshape(4,-1).copy()
def ict(p,t=0.25):
    q=np.clip(p,1e-8,1);ic=2+(q*np.log2(q)).sum(0);i=np.where(ic>=t)[0]
    return p if len(i)==0 else p[:,i[0]:i[-1]+1]
def colr(A,B):
    rs=[np.corrcoef(A[:,j],B[:,j])[0,1] for j in range(A.shape[1]) if A[:,j].std()>1e-8 and B[:,j].std()>1e-8]
    return float(np.mean(rs)) if rs else np.nan
m,_=load_model(CK,force_retrieval=False)
recs=[]
for r in held.itertuples():
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in r.sequence]],dtype=torch.long,device=device)
    mask=torch.ones(1,len(r.sequence),dtype=torch.bool,device=device)
    g,pp,_=infer(m,tok,mask,FID.get(r.family_name,9),ret=None)
    pred=ict(pp[:,active_cols(g,0.5)]); tg=ict(dec(r.pwm))
    if pred.shape[1]==0 or tg.shape[1]==0: continue
    al,sh,o,sc=align_pwm(pred,tg,max_shift=10,consider_revcomp=True)
    L=min(al.shape[1],tg.shape[1])
    A=al[:,:L]; G=tg[:,:L]
    honest=colr(np.clip(A,1e-8,1)/np.clip(A,1e-8,1).sum(0,keepdims=True), G/G.sum(0,keepdims=True))
    predic=float(np.maximum(2+(np.clip(A,1e-8,1)*np.log2(np.clip(A,1e-8,1))).sum(0),0).mean())
    cov=L/max(pred.shape[1],tg.shape[1])
    recs.append(dict(gene=r.gene_symbol,fam=r.family_name,fn=str(r.filename),
                     r=float(sc),honest=float(honest) if honest==honest else 0.0,
                     predic=predic,cov=float(cov),mid=float(r.mid),predal=A,gt=G))
df=pd.DataFrame([{k:v for k,v in x.items() if k not in("predal","gt")} for x in recs])
np.save(f"{RES}/heldout_clean_recs.npy",np.array(recs,dtype=object))
lt=df[df.mid<40]; ge=df[df.mid>=40]
print(f"scored {len(df)}  <40% id median r={lt.r.median():.3f} (n={len(lt)})  >=40% median r={ge.r.median():.3f} (n={len(ge)})")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import logomaker
fams=sorted(df.fam.unique()); cmap=dict(zip(fams,plt.cm.tab10(np.linspace(0,1,len(fams)))))
fig=plt.figure(figsize=(16,6.4))
gs=fig.add_gridspec(1,2,width_ratios=[0.92,1.28],wspace=0.14)

# panel a
axa=fig.add_subplot(gs[0,0])
axa.axvspan(0,40,color="#4575b4",alpha=0.06,zorder=0)
for f in fams:
    s=df[df.fam==f]; axa.scatter(s.mid,s.r,s=15,color=cmap[f],alpha=0.72,lw=0,label=f,zorder=3)
axa.axvline(40,color="#4575b4",ls="--",lw=1)
axa.hlines(lt.r.median(),18,40,color="#d73027",lw=2.4,zorder=7)
axa.hlines(ge.r.median(),40,100,color="#d73027",lw=2.4,zorder=7)
axa.text(29,lt.r.median()+0.035,f"novel (<40% id)\nmedian r={lt.r.median():.2f}",color="#a01",fontsize=8.5,ha="center")
axa.text(84,ge.r.median()-0.05,f"≥40% id\nmedian r={ge.r.median():.2f}",color="#a01",fontsize=8.5,ha="center",va="top")
axa.set_xlabel("% DBD identity to nearest training factor",fontsize=10.5)
axa.set_ylabel("motif recovery (oracle-aligned r)",fontsize=10.5)
axa.set_ylim(-0.05,1.05); axa.set_xlim(18,100)
axa.set_title("a   Novel factors recover as well as close homologs",fontsize=12,fontweight="bold",loc="left")
axa.legend(fontsize=7,frameon=False,ncol=2,loc="lower right",handletextpad=0.2,columnspacing=0.8)

# panel b: pick novel high-r exemplars, distinct families, tall logos
def logo(ax,p,title,c,ylab=False):
    p=np.clip(p,1e-9,1); p=p/p.sum(0,keepdims=True)
    ic=np.maximum(2+(p*np.log2(p)).sum(0),0)
    logomaker.Logo(pd.DataFrame((p*ic).T,columns=list("ACGT")),ax=ax,color_scheme="classic",show_spines=False,vpad=0.03)
    ax.set_xticks([]); ax.set_ylim(0,2)
    if ylab: ax.set_yticks([0,1,2]); ax.tick_params(labelsize=6)
    else: ax.set_yticks([])
    ax.set_title(title,fontsize=9.5,color=c,loc="left",pad=3)
def source_id(fn):
    body=fn.rsplit(".txt",1)[0].split("_",2)[-1]          # {TF}.{motifID}
    mid=body.split(".",1)[1] if "." in body else body
    u=mid.upper()
    if u.startswith("MA") and any(ch.isdigit() for ch in mid): return f"JASPAR {mid}"
    if "H11MO" in u or "H13CORE" in u or "_HUMAN" in u or "_MOUSE" in u: return f"HOCOMOCO {mid}"
    if u.startswith("M0") or u.startswith("M1"): return f"CIS-BP {mid}"
    return mid
cand=[x for x in recs if x["mid"]<40 and x["honest"]>=0.70 and x["predic"]>=0.65
      and x["cov"]>=0.7 and x["predal"].shape[1]>=6 and len(str(x["gene"]))>=3 and str(x["gene"])[0].isupper()]
cand.sort(key=lambda x:-x["honest"])
picks,seen=[],set()
for x in cand:
    if x["fam"] in seen: continue
    seen.add(x["fam"]); picks.append(x)
    if len(picks)==4: break
N=len(picks)
gsb_outer=gs[0,1].subgridspec(2,1,height_ratios=[0.12,0.88],hspace=0.02)
gsb=gsb_outer[1].subgridspec(N,2,hspace=0.85,wspace=0.10)
for i,x in enumerate(picks):
    L=min(x["predal"].shape[1],x["gt"].shape[1])
    logo(fig.add_subplot(gsb[i,0]), x["predal"][:,:L],
         f"{x['gene']}  ({x['fam']}, {x['mid']:.0f}% id, r={x['honest']:.2f})","#1a7d3a",ylab=True)
    logo(fig.add_subplot(gsb[i,1]), x["gt"][:,:L], source_id(x["fn"]),"#555")
axh=fig.add_subplot(gsb_outer[0]); axh.axis("off")
axh.text(0.0,0.95,"b",fontsize=13,fontweight="bold",transform=axh.transAxes,va="top")
hL=gsb[0,0].get_position(fig); hR=gsb[0,1].get_position(fig)
fig.text((hL.x0+hL.x1)/2,0.862,"TFScope predicted",fontsize=10.5,fontweight="bold",color="#1a7d3a",ha="center")
fig.text((hR.x0+hR.x1)/2,0.862,"Curated database motif",fontsize=10.5,fontweight="bold",color="#555",ha="center")
fig.suptitle("Sequence-only prediction generalizes to held-out transcription factors",
             fontsize=14,fontweight="bold",y=1.02)
out=f"{OUTD}/figure3a_heldout_clean"
fig.savefig(out+".png",dpi=250,bbox_inches="tight"); fig.savefig(out+".pdf",bbox_inches="tight")
print("exemplars:",[(x["gene"],x["fam"],round(x["r"],2),round(x["mid"])) for x in picks])
print("saved",out+".png")
