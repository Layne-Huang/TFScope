#!/usr/bin/env python
"""EXPERIMENT: recovery vs identity-to-training on a fresh, properly held-out set,
BALANCED to equal N per identity bin. Identity is to the combined model's ACTUAL training
(combined_fm_deeppbs train). Two designs:
  (A) equal-N per identity bin across ALL families (identity effect, family uncontrolled)
  (B) equal-N per identity bin WITHIN C2H2_long only (identity effect, family held constant)
Out: results/exp_balanced_identity/*.json + figures/exp_balanced_identity/*.png
"""
import os,sys,json
os.environ["TORCH_HOME"]="/data1/leihuang/.cache/torch"; os.environ["HF_HOME"]="/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/case_study")
import numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
from cs_utils import load_model, active_cols, infer, device
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm
rng=np.random.default_rng(0)
CK="/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
OUTD="results/exp_balanced_identity"; FIGD="figures/exp_balanced_identity"
os.makedirs(OUTD,exist_ok=True); os.makedirs(FIGD,exist_ok=True)

FID={"C2H2_short":0,"C2H2_medium":1,"C2H2_long":2,"bHLH":3,"Homeodomain":4,"bZIP":5,
     "Nuclear_Receptor":6,"Forkhead":7,"ETS":8,"Other":9}
comb=pd.read_parquet("data/processed/tf_pwm_combined_fm_deeppbs.parquet",columns=["filename","tf_name","sequence"])
ctr=set(json.load(open("data/processed/splits/combined_fm_deeppbs/split.json"))["train"])
train=comb[comb.filename.isin(ctr)]; train_tf=set(train.tf_name); train_seqset=set(train.sequence)
# full pool WITH pwm
aug=pd.read_parquet("data/processed/tf_pwm_aug_dbd.parquet")
aug=aug.drop_duplicates("sequence")
held=aug[(~aug.tf_name.isin(train_tf)) & (~aug.sequence.isin(train_seqset))].copy()
mid=pd.read_parquet("/tmp/claude-27813/-afs-csail-mit-edu-u-l-leihuang-project-TFScope/fdcc1f98-59bc-4c28-84e4-4be69cca02a0/scratchpad/heldout_pool.parquet")[["sequence","mid"]]
held=held.merge(mid,on="sequence",how="inner")
print(f"held-out pool with identity + pwm: {len(held)}")

def dec(v): return np.frombuffer(v,dtype=np.float32).reshape(4,-1).copy()
def ict(p,t=0.25):
    q=np.clip(p,1e-8,1);ic=2+(q*np.log2(q)).sum(0);i=np.where(ic>=t)[0]
    return p if len(i)==0 else p[:,i[0]:i[-1]+1]
m,_=load_model(CK,force_retrieval=False)
def recover(seq,pwm,fam):
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in seq]],dtype=torch.long,device=device)
    mask=torch.ones(1,len(seq),dtype=torch.bool,device=device)
    g,pp,_=infer(m,tok,mask,FID.get(fam,9),ret=None)
    pred=ict(pp[:,active_cols(g,0.5)]); tg=ict(dec(pwm))
    if pred.shape[1]==0 or tg.shape[1]==0: return np.nan
    _,_,_,sc=align_pwm(pred,tg,max_shift=10,consider_revcomp=True); return float(sc)

def balanced(df, edges, per_bin):
    picks=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        s=df[(df.mid>=lo)&(df.mid<hi)]
        take=min(per_bin,len(s))
        picks.append(s.sample(take,random_state=0))
    return pd.concat(picks)

def run(df,label,edges):
    per=min((((df.mid>=lo)&(df.mid<hi)).sum()) for lo,hi in zip(edges[:-1],edges[1:]))
    sel=balanced(df,edges,per)
    sel=sel.copy(); sel["r"]=[recover(r.sequence,r.pwm,r.family_name) for r in sel.itertuples()]
    sel=sel.dropna(subset=["r"])
    out=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        s=sel[(sel.mid>=lo)&(sel.mid<hi)]
        out.append(dict(bin=f"{lo}-{hi}",mid=(lo+hi)/2,n=len(s),
                        median=round(float(s.r.median()),3) if len(s) else None,
                        c2h2=round(float(s.family_name.str.startswith("C2H2").mean()),2) if len(s) else None))
    from scipy.stats import spearmanr
    rho,p=spearmanr(sel.mid,sel.r)
    print(f"\n=== {label}  (equal N={per}/bin, {len(sel)} total) ===")
    for o in out: print(f"  id {o['bin']:>7s}: n={o['n']:2d}  median r={o['median']}  C2H2 frac={o['c2h2']}")
    print(f"  Spearman r-vs-identity: rho={rho:+.3f}  p={p:.1e}")
    return sel,out,dict(rho=float(rho),p=float(p),per_bin=int(per))

edA=[20,30,40,50,60,70,80,90,101]
selA,binA,statA=run(held,"(A) ALL families, balanced per identity bin",edA)
c2=held[held.family_name.str.startswith("C2H2_long")]
edB=[30,40,50,60,70,80]
selB,binB,statB=run(c2,"(B) WITHIN C2H2_long, balanced per identity bin",edB)
json.dump(dict(A=dict(bins=binA,**statA),B=dict(bins=binB,**statB)),open(f"{OUTD}/balanced_identity.json","w"),indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
for a,(sel,binr,stat,title) in zip(ax,[(selA,binA,statA,"(A) all families, N balanced per bin"),
                                       (selB,binB,statB,"(B) within C2H2_long, N balanced per bin")]):
    a.scatter(sel.mid,sel.r,s=14,color="#4575b4",alpha=0.5,lw=0)
    bx=[b["mid"] for b in binr if b["median"] is not None]; bm=[b["median"] for b in binr if b["median"] is not None]
    a.plot(bx,bm,"-o",color="#c0392b",lw=2.2,ms=6)
    for b in binr:
        if b["median"] is not None: a.text(b["mid"],b["median"]+0.03,f"n={b['n']}",fontsize=7,ha="center")
    a.set_xlabel("% DBD identity to combined-train"); a.set_ylabel("recovery (oracle-aligned r)")
    a.set_ylim(0,1.02); a.set_title(f"{title}\nSpearman rho={stat['rho']:+.2f}, p={stat['p']:.0e}",fontsize=9.5,fontweight="bold")
fig.suptitle("Balanced recovery-vs-identity on a properly held-out set (TF & seq not in combined train)",
             fontsize=11.5,fontweight="bold",y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGD}/balanced_identity_recovery.png",dpi=200,bbox_inches="tight")
fig.savefig(f"{FIGD}/balanced_identity_recovery.pdf",bbox_inches="tight")
print(f"\nsaved {FIGD}/balanced_identity_recovery.png")
