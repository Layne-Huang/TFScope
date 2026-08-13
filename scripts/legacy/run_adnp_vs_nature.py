#!/usr/bin/env python
"""Which ADNP prediction matches the Ostapcuk et al. 2018 Nature motif GCCCCCTGGAG?
Runs 3 crops x 3 models, scores each against the (GC-rich) experimental consensus.
"""
import os, sys
os.environ["TORCH_HOME"]="/data1/leihuang/.cache/torch"; os.environ["HF_HOME"]="/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/case_study")
import numpy as np, torch
import warnings; warnings.filterwarnings("ignore")
from cs_utils import load_model, active_cols, infer, device
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm

SEQ=open("/tmp/adnp_seq.txt").read().strip()
MODELS={"combined":"/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt",
        "residue-MoE(base)":"/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42/ckpt_best.pt",
        "residue-MoE(deeptune-ddp)":"/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe_deeptune/deeptune_ddp_seed42/ckpt_best.pt"}
def sl(a,b): return SEQ[a-1:b]
CROPS={"HD(752-822)":(sl(752,822),[True]*71,4),
       "C2H2 array(447-535)":(sl(447,535),[True]*(535-447+1),1)}
md=sl(447,822); m=[False]*len(md)
for a,b in [(447,535),(752,814)]:
    for i in range(a-447,b-447+1): m[i]=True
CROPS["Multi-domain(447-822)"]=(md,m,4)

def onehot(cons):
    P=np.full((4,len(cons)),0.02,np.float32)
    for j,ch in enumerate(cons): P["ACGT".index(ch),j]=0.94
    return P
NAT=onehot("GCCCCCTGGAG")
def ic_trim(p,t=0.25):
    p=np.clip(p,1e-8,1);ic=2+(p*np.log2(p)).sum(0);i=np.where(ic>=t)[0]
    return p if len(i)==0 else p[:,i[0]:i[-1]+1]
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))
def gc(p): return float(p[[1,2]].sum(0).mean())   # mean G+C prob
def r(a,b):
    _,_,_,x=align_pwm(ic_trim(a),ic_trim(b),max_shift=12,consider_revcomp=True); return float(x)

def run(model,s,mk,fid):
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in s]],dtype=torch.long,device=device)
    mask=torch.tensor([mk],dtype=torch.bool,device=device)
    g,p,_=infer(model,tok,mask,fid,ret=None); return ic_trim(p[:,active_cols(g,0.5)])

print(f"{'model':26s} {'crop':22s} {'consensus':16s} {'GC':>5s} {'r_vs_GCCCCCTGGAG':>16s}")
for mn,ck in MODELS.items():
    model,_=load_model(ck,force_retrieval=False)
    for cn,(s,mk,fid) in CROPS.items():
        p=run(model,s,mk,fid)
        print(f"{mn:26s} {cn:22s} {cons(p):16s} {gc(p):5.2f} {r(p,NAT):16.3f}",flush=True)
    del model; torch.cuda.empty_cache()
print(f"\nreference GCCCCCTGGAG  GC-content = {gc(NAT):.2f}")
