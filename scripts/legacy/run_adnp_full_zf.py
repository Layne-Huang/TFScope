#!/usr/bin/env python
"""ADNP full 9-finger C2H2 zinc-finger array as input (window 74-686, all 9 ZFs masked,
homeobox excluded). Compare to Ostapcuk 2018 Nature motif GCCCCCTGGAG and JASPAR UN0305.1.
"""
import os, sys
os.environ["TORCH_HOME"]="/data1/leihuang/.cache/torch"; os.environ["HF_HOME"]="/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/case_study")
import numpy as np, torch
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import logomaker, pandas as pd
from cs_utils import load_model, active_cols, infer, device
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm

SEQ=open("/tmp/adnp_seq.txt").read().strip()
ZFS=[(74,97),(107,129),(165,188),(221,244),(447,469),(489,510),(512,535),(622,647),(662,686)]
MODELS={"combined":"/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt",
        "residue-MoE(base)":"/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42/ckpt_best.pt",
        "residue-MoE(deeptune-ddp)":"/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe_deeptune/deeptune_ddp_seed42/ckpt_best.pt"}
FID_C2H2=2  # C2H2_long (9 fingers)
start,end=71,689
win=SEQ[start-1:end]; mask=[False]*len(win)
for a,b in ZFS:
    for i in range(a,b+1):
        j=i-start
        if 0<=j<len(win): mask[j]=True
print(f"window {start}-{end} ({len(win)} aa), 9 ZFs masked ({sum(mask)} residues)")

def onehot(c):
    P=np.full((4,len(c)),0.02,np.float32)
    for j,ch in enumerate(c): P["ACGT".index(ch),j]=0.94
    return P
NAT=onehot("GCCCCCTGGAG"); UN=np.load("/tmp/un0305_pwm.npy")
def ict(p,t=0.25):
    p=np.clip(p,1e-8,1);ic=2+(p*np.log2(p)).sum(0);i=np.where(ic>=t)[0]
    return p if len(i)==0 else p[:,i[0]:i[-1]+1]
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))
def gc(p): return float(p[[1,2]].sum(0).mean())
def r(a,b):
    _,_,_,x=align_pwm(ict(a),ict(b),max_shift=12,consider_revcomp=True); return float(x)
def run(model):
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in win]],dtype=torch.long,device=device)
    mk=torch.tensor([mask],dtype=torch.bool,device=device)
    g,p,_=infer(model,tok,mk,FID_C2H2,ret=None); return ict(p[:,active_cols(g,0.5)])

rows=[]
print(f"\n{'model':26s} {'consensus':18s} {'GC':>5s} {'r_GCCCCCTGGAG':>14s} {'r_UN0305':>9s}")
for mn,ck in MODELS.items():
    model,_=load_model(ck,force_retrieval=False); p=run(model)
    rows.append((mn,p)); print(f"{mn:26s} {cons(p):18s} {gc(p):5.2f} {r(p,NAT):14.3f} {r(p,UN):9.3f}",flush=True)
    del model; torch.cuda.empty_cache()

def logo(ax,p,t,c):
    q=np.clip(p,1e-8,1);ic=np.maximum(2+(q*np.log2(q)).sum(0),0)
    logomaker.Logo(pd.DataFrame((p*ic).T,columns=list("ACGT")),ax=ax,color_scheme="classic",show_spines=False,vpad=0.02)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_ylim(0,2);ax.set_title(t,fontsize=8.5,color=c,loc="left",pad=2)
allrows=[("experimental: GCCCCCTGGAG (Nature 2018)",NAT,"#c0392b")]+[
    (f"{mn} (9-ZF array)   {cons(p)}   r={r(p,NAT):.2f}",p,"#0072B2" if "residue" in mn else "#009E73") for mn,p in rows]
fig,axes=plt.subplots(len(allrows),1,figsize=(5.8,1.2*len(allrows)+0.4))
for ax,(t,p,c) in zip(axes,allrows): logo(ax,p,t,c)
fig.suptitle("ADNP full 9-finger ZF array vs the GC-rich Nature motif",fontsize=10,fontweight="bold",y=1.0)
fig.tight_layout()
os.makedirs("figures/figure_adnp_multidomain",exist_ok=True)
for e in ["png","pdf"]: fig.savefig(f"figures/figure_adnp_multidomain/adnp_full_zf.{e}",dpi=200,bbox_inches="tight")
print("\nsaved figures/figure_adnp_multidomain/adnp_full_zf.png")
