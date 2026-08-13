#!/usr/bin/env python
"""Orphan TFs with the MoE + contact-bias model (v19_residue_moe_contactbias).
Reports, per TF: (1) raw 20-column PWM prediction, (2) the GATE CORE with NO IC-trim,
(3) the experimental reference motif. Input crop = all-DBD window (all DBDs masked).
Refs are primary-literature / JASPAR CORE only -- no JASPAR UNVALIDATED entries.
Out: figures/figure_orphans_moebias/ + results/orphan_multidomain/{TF}_moebias_*.npy
"""
import os, sys, json
os.environ["TORCH_HOME"]="/data1/leihuang/.cache/torch"; os.environ["HF_HOME"]="/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/case_study")
import numpy as np, torch, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import logomaker, pandas as pd
from cs_utils import load_model, active_cols, infer, device
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm

CKPT="/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe_contactbias/contactbias_seed42/ckpt_best.pt"
ORDER=["SOHLH1","ADNP","ADNP2","ZHX2","ZHX3","ZGLP1"]
FAMTXT={"SOHLH1":"bHLH","ADNP":"Homeo+ZF","ADNP2":"Homeo+ZF","ZHX2":"ZF+4HD","ZHX3":"ZF+5HD","ZGLP1":"GATA"}
FID={"SOHLH1":3,"ADNP":4,"ADNP2":4,"ZHX2":4,"ZHX3":4,"ZGLP1":9}
FEAT=json.load(open("results/orphan_multidomain/dbd_features.json"))
OUTD="results/orphan_multidomain"; FIGD="figures/figure_orphans_moebias"; os.makedirs(FIGD,exist_ok=True)
GTH=0.5; MAXW=1000

IUP={"A":"A","C":"C","G":"G","T":"T","R":"AG","Y":"CT","W":"AT","S":"GC","M":"AC","K":"GT","N":"ACGT"}
def onehot(c):
    P=np.full((4,len(c)),0.02,np.float32)
    for j,ch in enumerate(c):
        for b in IUP.get(ch,ch): P["ACGT".index(b),j]=0.94/len(IUP.get(ch,ch))
    return P/P.sum(0,keepdims=True)
def ict(p,t=0.25):
    q=np.clip(p,1e-8,1);ic=2+(q*np.log2(q)).sum(0);i=np.where(ic>=t)[0]
    return p if len(i)==0 else p[:,i[0]:i[-1]+1]
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))
def gc(p): return float(p[[1,2]].sum(0).mean())
def lc(p):
    c=cons(p); return max(c.count(b) for b in "ACGT")/len(c)
def rr(a,b):
    _,_,_,x=align_pwm(ict(a),ict(b),max_shift=12,consider_revcomp=True); return float(x)

REF={}
# SOHLH1: no EMSA exists; CACGTG is from reporter+ChIP (Suzuki 2014 PLOS One e101681) and SOHLH1
# binds only as a SOHLH2/SOHLH1 heterodimer. WEAK reference -- see run_orphans_moebase.py.
REF["SOHLH1"]=("CACGTG (reporter+ChIP, PLOS One 2014; heterodimer)",onehot("CACGTG"))
REF["ADNP"]=("GCCCCCTGGAG (Nature 2018)",onehot("GCCCCCTGGAG"))
REF["ADNP2"]=("TGGGTTCCT (zebrafish, putative)",onehot("TGGGTTCCT"))
REF["ZHX2"]=("AGGCTGAG (EMSA, NatComm 2023)",onehot("AGGCTGAG"))
REF["ZHX3"]=("MTTTATR CdxA-like (EMSA, JBC 2006)",onehot("MTTTATR"))
if os.path.exists("/tmp/ma2557_pwm.npy"): REF["ZGLP1"]=("MA2557.1 CORE PBM",ict(np.load("/tmp/ma2557_pwm.npy")))

def alldbd_input(g):
    seq=open(f"{OUTD}/{g}.seq.txt").read().strip()
    dbds=[(a,b) for a,b,_,_ in FEAT[g]["dbds"]]
    end=min(len(seq),max(b for _,b in dbds)+3); start=max(1,min(a for a,_ in dbds)-3)
    if end-start+1>MAXW: start=end-MAXW+1
    win=seq[start-1:end]; mask=[False]*len(win)
    for a,b in dbds:
        for i in range(a,b+1):
            j=i-start
            if 0<=j<len(win): mask[j]=True
    return win,mask

print("loading MoE + contact-bias ...",flush=True)
model,cfg=load_model(CKPT,force_retrieval=False)
print(f"  v18_contact_bias_scale = {getattr(cfg,'v18_contact_bias_scale','n/a')}",flush=True)

res={}
print(f"\n{'TF':8s} {'raw 20-col prediction':24s} {'GC':>5s} | {'gate core (NO IC-trim)':24s} {'w':>2s} {'GC':>5s} {'LC':>5s} | {'r vs ref':>8s}  reference")
print("-"*130)
for g in ORDER:
    win,mk=alldbd_input(g)
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in win]],dtype=torch.long,device=device)
    mask=torch.tensor([mk],dtype=torch.bool,device=device)
    gate,pp,_=infer(model,tok,mask,FID[g],ret=None)
    ac=active_cols(gate,GTH); core=pp[:,ac]          # gate core, NO IC-trim
    res[g]=dict(raw=pp,core=core,gate=gate,ac=ac)
    np.save(f"{OUTD}/{g}_moebias_raw20.npy",pp); np.save(f"{OUTD}/{g}_moebias_gatecore.npy",core)
    rtxt="   --   "; rname="(none)"
    if g in REF:
        rname=REF[g][0]; rtxt=f"{rr(core,REF[g][1]):8.2f}"
    print(f"{g:8s} {cons(pp):24s} {gc(pp):5.2f} | {cons(core):24s} {core.shape[1]:2d} {gc(core):5.2f} {lc(core):5.2f} | {rtxt}  {rname}")

print("\nreference GC (for the AT-vs-GC artifact check):")
for g in ORDER:
    if g in REF: print(f"  {g:7s} ref GC {gc(REF[g][1]):.2f}   pred core GC {gc(res[g]['core']):.2f}")

def logo(ax,p,t,c="black"):
    q=np.clip(p,1e-8,1);ic=np.maximum(2+(q*np.log2(q)).sum(0),0)
    logomaker.Logo(pd.DataFrame((p*ic).T,columns=list("ACGT")),ax=ax,color_scheme="classic",show_spines=False,vpad=0.02)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_ylim(0,2);ax.set_title(t,fontsize=8,color=c,loc="left",pad=2)
fig,axes=plt.subplots(len(ORDER),3,figsize=(13,1.5*len(ORDER)))
for row,g in enumerate(ORDER):
    R=res[g]
    logo(axes[row,0],R["raw"],f"{g} ({FAMTXT[g]}) | raw 20-col prediction","#009E73")
    # shade gate-off columns on the raw panel
    for j,on in enumerate(R["ac"]):
        if not on: axes[row,0].axvspan(j-0.5,j+0.5,color="0.85",zorder=0)
    logo(axes[row,1],R["core"],f"{g} | gate core, NO IC-trim ({R['core'].shape[1]} bp)  {cons(R['core'])}","#0072B2")
    if g in REF:
        logo(axes[row,2],REF[g][1],f"{g} | experimental ref\n{REF[g][0]}","#c0392b")
    else:
        axes[row,2].text(.5,.5,"no experimental motif",ha="center",va="center",fontsize=9,color="#888",transform=axes[row,2].transAxes)
        axes[row,2].set_xticks([]);axes[row,2].set_yticks([])
        for s in axes[row,2].spines.values(): s.set_visible(False)
fig.suptitle("Orphan TFs — TFScope (MoE + contact bias): raw prediction | gate core (no IC-trim) | experimental reference\n"
             "grey = columns the gate switched off",fontsize=11,fontweight="bold",y=1.005)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(f"{FIGD}/orphans_moebias.{e}",dpi=200,bbox_inches="tight")
print("\nsaved",f"{FIGD}/orphans_moebias.png",flush=True)
