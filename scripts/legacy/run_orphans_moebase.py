#!/usr/bin/env python
"""Orphan-TF analysis, MoE-BASE only. For each of the 6 orphans, predict from
(1) the FULL sequence and (2) the ALL-DBD-spanning window (all DBDs masked), and
compare to any experimental reference. Figures -> figures/figure_orphans_moebase/.

Experimental refs (from literature/DB search):
  ZGLP1  = JASPAR MA2557.1 (CORE, PBM)          GATA-like
  ADNP   = JASPAR UN0305.1 (UNVALIDATED ChIP)   + Nature GCCCCCTGGAG
  SOHLH1 = E-box CAGCTG (EMSA; no formal PWM)
  ADNP2, ZHX2, ZHX3 = none
"""
import os, sys, json
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

CKPT="/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42/ckpt_best.pt"
ORDER=["SOHLH1","ADNP","ADNP2","ZHX2","ZHX3","ZGLP1"]
FAMTXT={"SOHLH1":"bHLH","ADNP":"Homeo+ZF","ADNP2":"Homeo+ZF","ZHX2":"ZF+4HD","ZHX3":"ZF+5HD","ZGLP1":"GATA"}
FID={"SOHLH1":3,"ADNP":4,"ADNP2":4,"ZHX2":4,"ZHX3":4,"ZGLP1":9}
FEAT=json.load(open("results/orphan_multidomain/dbd_features.json"))
OUTD="results/orphan_multidomain"; FIGD="figures/figure_orphans_moebase"; os.makedirs(FIGD,exist_ok=True)
GTH=0.5; MAXW=1000

IUP={"A":"A","C":"C","G":"G","T":"T","R":"AG","Y":"CT","W":"AT","S":"GC","M":"AC","K":"GT","N":"ACGT"}
def onehot(c):
    P=np.full((4,len(c)),0.02,np.float32)
    for j,ch in enumerate(c):
        bs=IUP.get(ch,ch)
        for b in bs: P["ACGT".index(b),j]=0.94/len(bs)
    return P/P.sum(0,keepdims=True)
def ict(p,t=0.25):
    q=np.clip(p,1e-8,1);ic=2+(q*np.log2(q)).sum(0);i=np.where(ic>=t)[0]
    return p if len(i)==0 else p[:,i[0]:i[-1]+1]
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))
def gc(p): return float(p[[1,2]].sum(0).mean())
def r(a,b):
    _,_,_,x=align_pwm(ict(a),ict(b),max_shift=12,consider_revcomp=True); return float(x)
# references — primary-literature / validated experimental motifs (NO JASPAR UNVALIDATED)
REF={}
# SOHLH1: NO EMSA / no direct monomer DNA-binding assay exists. The only specific E-box reported
# for a SOHLH1-bound promoter is CACGTG (Sohlh1 autoregulation, Suzuki 2014 PLOS One e101681,
# reporter + ChIP -- NOT EMSA), and SOHLH1 alone does not effectively transactivate: it acts as a
# SOHLH2/SOHLH1 heterodimer (+SP1). Treat this reference as WEAK; SOHLH1 is arguably not a valid
# monomer test case for TFScope at all. (The earlier "CAGCTG (EMSA)" reference was UNSOURCED.)
REF["SOHLH1"]=("CACGTG (reporter+ChIP, PLOS One 2014; heterodimer)",onehot("CACGTG"))
REF["ADNP"]=("GCCCCCTGGAG (Nature2018)",onehot("GCCCCCTGGAG"))          # dropped UN0305.1 (UNVALIDATED)
REF["ADNP2"]=("TGGGTTCCT (zebrafish, putative)",onehot("TGGGTTCCT"))    # Wang 2026 Development
REF["ZHX2"]=("AGGCTGAG (EMSA, NatComm2023)",onehot("AGGCTGAG"))         # Zhang 2023
REF["ZHX3"]=("MTTTATR CdxA-like (EMSA, JBC2006)",onehot("MTTTATR"))     # Liu 2006
if os.path.exists("/tmp/ma2557_pwm.npy"): REF["ZGLP1"]=("MA2557.1 CORE PBM",ict(np.load("/tmp/ma2557_pwm.npy")))

def full_input(g):
    seq=open(f"{OUTD}/{g}.seq.txt").read().strip()[:MAXW]
    return seq,[True]*len(seq)
def alldbd_input(g):
    seq=open(f"{OUTD}/{g}.seq.txt").read().strip()
    dbds=[(a,b) for a,b,_,_ in FEAT[g]["dbds"]]
    lastend=max(b for _,b in dbds); first=min(a for a,_ in dbds)
    end=min(len(seq),lastend+3); start=max(1,first-3)
    if end-start+1>MAXW: start=end-MAXW+1
    win=seq[start-1:end]; mask=[False]*len(win)
    for a,b in dbds:
        for i in range(a,b+1):
            j=i-start
            if 0<=j<len(win): mask[j]=True
    return win,mask
def run(model,seqc,mk,fid):
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in seqc]],dtype=torch.long,device=device)
    mask=torch.tensor([mk],dtype=torch.bool,device=device)
    g,p,_=infer(model,tok,mask,fid,ret=None); return ict(p[:,active_cols(g,GTH)])

print("loading MoE-base ...",flush=True)
model,_=load_model(CKPT,force_retrieval=False)
res={}
for g in ORDER:
    fs,fm=full_input(g); aw,am=alldbd_input(g)
    pf=run(model,fs,fm,FID[g]); pa=run(model,aw,am,FID[g])
    res[g]=dict(full=pf,alldbd=pa)
    line=f"{g:7s} full {cons(pf):16s}(GC{gc(pf):.2f})  all-DBD {cons(pa):16s}(GC{gc(pa):.2f})"
    if g in REF: line+=f"  | ref {REF[g][0]}: r_full {r(pf,REF[g][1]):.2f} r_allDBD {r(pa,REF[g][1]):.2f}"
    print(line,flush=True)
    np.save(f"{OUTD}/{g}_moebase_full.npy",pf); np.save(f"{OUTD}/{g}_moebase_alldbd.npy",pa)

def logo(ax,p,t,c="black"):
    q=np.clip(p,1e-8,1);ic=np.maximum(2+(q*np.log2(q)).sum(0),0)
    logomaker.Logo(pd.DataFrame((p*ic).T,columns=list("ACGT")),ax=ax,color_scheme="classic",show_spines=False,vpad=0.02)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_ylim(0,2);ax.set_title(t,fontsize=8,color=c,loc="left",pad=2)
ncol=3
fig,axes=plt.subplots(len(ORDER),ncol,figsize=(12,1.45*len(ORDER)))
for row,g in enumerate(ORDER):
    logo(axes[row,0],res[g]["full"],f"{g} ({FAMTXT[g]}) | FULL-seq  {cons(res[g]['full'])}","#009E73")
    logo(axes[row,1],res[g]["alldbd"],f"{g} | ALL-DBD  {cons(res[g]['alldbd'])}","#0072B2")
    if g in REF:
        rn,rp=REF[g]; logo(axes[row,2],rp,f"{g} | experimental ref\n{rn}","#c0392b")
    else:
        axes[row,2].text(0.5,0.5,"no experimental\nmotif",ha="center",va="center",fontsize=9,color="#888",transform=axes[row,2].transAxes)
        axes[row,2].set_xticks([]);axes[row,2].set_yticks([])
        for s in axes[row,2].spines.values(): s.set_visible(False)
fig.suptitle("Orphan TFs (MoE-base): full-seq vs all-DBD prediction, vs experimental reference",fontsize=12,fontweight="bold",y=1.0)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(f"{FIGD}/orphans_moebase.{e}",dpi=200,bbox_inches="tight")
print("\nsaved",f"{FIGD}/orphans_moebase.png",flush=True)
