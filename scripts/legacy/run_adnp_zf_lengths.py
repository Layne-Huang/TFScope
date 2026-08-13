#!/usr/bin/env python
"""ADNP: sweep zinc-finger crop LENGTH (which ZF span best reproduces the GC-rich
Nature motif GCCCCCTGGAG), combined + MoE(base) + MoE(deeptune)."""
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
ZFS={"ZF5":(447,469),"ZF6":(489,510),"ZF7":(512,535),"ZF8":(622,647),"ZF9":(662,686),
     "ZF1":(74,97),"ZF2":(107,129),"ZF3":(165,188),"ZF4":(221,244)}
# crops: (window_start, window_end, [ZF names to mask], FID)
CROPS={
 "ZF5-7 (447-535)":   (447,535,["ZF5","ZF6","ZF7"],1),
 "ZF5-8 (447-647)":   (447,647,["ZF5","ZF6","ZF7","ZF8"],2),
 "ZF5-9 (447-686)":   (447,686,["ZF5","ZF6","ZF7","ZF8","ZF9"],2),
 "ZF1-9 all (74-686)":(74,686,list(ZFS),2),
}
MODELS={"combined":"/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt",
        "MoE(base)":"/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42/ckpt_best.pt",
        "MoE(deeptune)":"/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe_deeptune/deeptune_seed42/ckpt_best.pt"}
def onehot(c):
    P=np.full((4,len(c)),0.02,np.float32)
    for j,ch in enumerate(c): P["ACGT".index(ch),j]=0.94
    return P
NAT=onehot("GCCCCCTGGAG")
def ict(p,t=0.25):
    p=np.clip(p,1e-8,1);ic=2+(p*np.log2(p)).sum(0);i=np.where(ic>=t)[0]
    return p if len(i)==0 else p[:,i[0]:i[-1]+1]
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))
def rc(s): return s[::-1].translate(str.maketrans("ACGT","TGCA"))
def gc(p): return float(p[[1,2]].sum(0).mean())
def r(a,b):
    _,_,_,x=align_pwm(ict(a),ict(b),max_shift=12,consider_revcomp=True); return float(x)
def build(ws,we,zfnames):
    win=SEQ[ws-1:we]; m=[False]*len(win)
    for z in zfnames:
        a,b=ZFS[z]
        for i in range(a,b+1):
            j=i-ws
            if 0<=j<len(win): m[j]=True
    return win,m
def run(model,win,mk,fid):
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in win]],dtype=torch.long,device=device)
    mask=torch.tensor([mk],dtype=torch.bool,device=device)
    g,p,_=infer(model,tok,mask,fid,ret=None); return ict(p[:,active_cols(g,0.5)])

print(f"{'model':14s} {'crop':20s} {'consensus':16s} {'GC':>5s} {'r_GCCCCCTGGAG':>14s}")
for mn,ck in MODELS.items():
    model,_=load_model(ck,force_retrieval=False)
    for cn,(ws,we,zf,fid) in CROPS.items():
        win,mk=build(ws,we,zf); p=run(model,win,mk,fid)
        c=cons(p); star=" *" if gc(p)>=0.5 and r(p,NAT)>=0.5 else ""
        print(f"{mn:14s} {cn:20s} {c:16s} {gc(p):5.2f} {r(p,NAT):14.3f}{star}",flush=True)
    del model; torch.cuda.empty_cache()
print("\n* = GC-rich AND r>=0.5 (genuine GC-rich match). experimental GCCCCCTGGAG GC=0.79")
