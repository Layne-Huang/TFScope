#!/usr/bin/env python
"""ZHX2 crop comparison, MoE-base ONLY. Five crops, one reference (AGGCTGAG, Zhang 2023
Nat Commun 14:7527 -- ChIP-seq + nuclear-extract EMSA; domain NOT mapped by the paper).
Tests the domain-swamping hypothesis: pooled DBD mean is dominated by the largest domain.
Out: figures/figure_zhx2_crops/
"""
import os,sys
os.environ["TORCH_HOME"]="/data1/leihuang/.cache/torch"; os.environ["HF_HOME"]="/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/case_study")
import numpy as np, torch, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import logomaker, pandas as pd
from cs_utils import load_model, active_cols, infer, device
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm
CK="/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42/ckpt_best.pt"
seq=open("results/orphan_multidomain/ZHX2.seq.txt").read().strip()
ZF=[(78,101),(110,133)]; HD=[(263,324),(439,501),(530,591),(628,690)]
HDcore=[(444,496),(537,582)]
# (label, win_start, win_end, masks, FID)
CROPS=[("1. ZF1+ZF2 only",        65,146, ZF,                 0),
       ("2. HD2+HD3 (primary)",  420,620, HDcore,             4),
       ("3. ZF + HD2+HD3",        75,620, ZF+HDcore,          4),
       ("4. all-DBD (6 domains)", 75,693, ZF+HD,              4),
       ("5. full sequence",        1,837, ZF+HD,              4)]
MAXW=1000
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))
def gc(p): return float(p[[1,2]].sum(0).mean())
def lc(p):
    c=cons(p); return max(c.count(b) for b in "ACGT")/len(c)
def ict(p,t=0.25):
    q=np.clip(p,1e-8,1);ic=2+(q*np.log2(q)).sum(0);i=np.where(ic>=t)[0]
    return p if len(i)==0 else p[:,i[0]:i[-1]+1]
def onehot(c):
    P=np.full((4,len(c)),0.02,np.float32)
    for j,ch in enumerate(c): P["ACGT".index(ch),j]=0.94
    return P/P.sum(0,keepdims=True)
def rr(a,b):
    _,_,o,x=align_pwm(ict(a),ict(b),max_shift=12,consider_revcomp=True); return float(x),bool(o)
REF=onehot("AGGCTGAG")
print("loading MoE-base ...",flush=True)
m,_=load_model(CK,force_retrieval=False)
print(f"\nreference AGGCTGAG  GC {gc(REF):.2f}  LC {lc(REF):.2f}\n")
print(f"{'crop':24s} {'window':>10s} {'msk aa':>7s} {'%HD':>5s} | {'raw 20-col':22s} {'gate core':14s} {'w':>2s} {'GC':>5s} {'LC':>5s} {'r':>5s}")
print("-"*118)
rows=[]
for lab,s,e,masks,fid in CROPS:
    s=max(1,s); e=min(len(seq),e)
    if e-s+1>MAXW: s=e-MAXW+1
    win=seq[s-1:e]; mk=[False]*len(win)
    nzf=nhd=0
    for a,b in masks:
        for i in range(a,b+1):
            j=i-s
            if 0<=j<len(win):
                mk[j]=True
                if any(a>=x and b<=y for x,y in [(78,133)]): nzf+=1
                else: nhd+=1
    tot=sum(mk); pct=100*nhd/max(tot,1)
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in win]],dtype=torch.long,device=device)
    mask=torch.tensor([mk],dtype=torch.bool,device=device)
    g,pp,_=infer(m,tok,mask,fid,ret=None)
    core=pp[:,active_cols(g,0.5)]
    r,_=rr(core,REF)
    rows.append((lab,pp,core,gc(core),lc(core),r,tot,pct))
    print(f"{lab:24s} {f'{s}-{e}':>10s} {tot:7d} {pct:4.0f}% | {cons(pp):22s} {cons(core):14s} {core.shape[1]:2d} {gc(core):5.2f} {lc(core):5.2f} {r:5.2f}",flush=True)
print(f"\n{'reference':24s} {'':10s} {'':7s} {'':5s} | {'':22s} {'AGGCTGAG':14s} {8:2d} {gc(REF):5.2f} {lc(REF):5.2f}")
print("\nGC of the prediction is the honesty check: r is meaningless when GC(pred) != GC(ref).")

def logo(ax,p,t,c="black"):
    q=np.clip(p,1e-8,1);ic=np.maximum(2+(q*np.log2(q)).sum(0),0)
    logomaker.Logo(pd.DataFrame((p*ic).T,columns=list("ACGT")),ax=ax,color_scheme="classic",show_spines=False,vpad=0.02)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_ylim(0,2);ax.set_title(t,fontsize=8,color=c,loc="left",pad=2)
fig,axes=plt.subplots(6,2,figsize=(11,9.5))
for i,(lab,pp,core,g_,l_,r_,tot,pct) in enumerate(rows):
    col="#c0392b" if g_<0.35 else ("#0072B2" if g_>=0.50 else "#888")
    logo(axes[i,0],pp,f"{lab} | raw 20-col  (GC {gc(pp):.2f})","#555")
    logo(axes[i,1],core,f"{lab} | gate core  {cons(core)}   GC {g_:.2f}  LC {l_:.2f}  r={r_:.2f}",col)
logo(axes[5,0],REF,"reference AGGCTGAG (Zhang 2023 Nat Commun; ChIP-seq + EMSA)","#c0392b")
axes[5,1].axis("off")
axes[5,1].text(0.02,0.5,"red = AT-rich prediction (GC<0.35): homeodomain prior\nblue = GC-matched to reference\n\n"
    "Domain swamping: pooled DBD mean is dominated by\nthe largest masked domain. Only the ZF-only crop\n"
    "recovers the reference's GC composition.",fontsize=9,va="center",transform=axes[5,1].transAxes)
fig.suptitle("ZHX2 crop comparison (MoE-base): the zinc-finger signal is erased when homeodomains are added",
             fontsize=12,fontweight="bold",y=1.0)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(f"figures/figure_zhx2_crops/zhx2_crop_compare.{e}",dpi=200,bbox_inches="tight")
print("\nsaved figures/figure_zhx2_crops/zhx2_crop_compare.png")
