#!/usr/bin/env python
"""Two standalone logo figures: TFScope (MoE-base) prediction vs experimental reference.

SOHLH1: crop 40-130, DBD mask 53-111, FID=3 (bHLH). ref CACGTG
        (Suzuki 2014 PLOS One e101681, PMC4086951; reporter+ChIP, SOHLH2/SOHLH1 heterodimer, no EMSA).
ADNP:   crop 447-686, ZF5-9 masked, FID=2 (C2H2_long). ref GCCCCCTGGAG
        (Ostapcuk 2018 Nature 557:739, ChAHP).
Motif = gate core, NO IC-trim.

r is reported over the MAXIMUM-COVERAGE ungapped alignment (all reference columns that can be
covered), NOT align_pwm's default -- align_pwm selects on r*(overlap/Lr) with min_overlap=2 and
for short references a PARTIAL overlap wins (SOHLH1: 4/6 cols r=.655 beats 6/6 r=.358;
ADNP: an RC 5/11 G-tail overlap r=.647 beats the poly-C alignment r=.550). Both numbers are shown.
Out: figures/figure_{sohlh1,adnp}_logo/
"""
import os,sys
os.environ["TORCH_HOME"]="/data1/leihuang/.cache/torch"; os.environ["HF_HOME"]="/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import logomaker, pandas as pd
from tfscope.models.alignment import align_pwm

OUT="results/sohlh1_adnp_case"
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))
def gc(p): return float(p[[1,2]].sum(0).mean())
def icv(p):
    q=np.clip(p,1e-8,1); return 2+(q*np.log2(q)).sum(0)
def meanic(p): return float(icv(p).mean())
def rcp(p): return p[::-1,::-1].copy()
def colr(a,b):
    if np.std(a)<1e-8 or np.std(b)<1e-8: return np.nan
    return float(np.corrcoef(a,b)[0,1])
def display_align(pred,ref):
    """FORWARD, offset 0: the biologically motivated, conservative placement.
    We do NOT let the aligner pick, because for these two cases the argmax over
    orientation x shift selects a partial / reverse-complement overlap (see docstring)."""
    Lp,Lr=pred.shape[1],ref.shape[1]; off=0
    i0,i1=max(0,-off),min(Lp,Lr-off)
    rs=[colr(pred[:,i],ref[:,i+off]) for i in range(i0,i1)]
    rs=[x for x in rs if not np.isnan(x)]
    return dict(orient="fwd",off=off,n=i1-i0,cov=(i1-i0)/Lr,r=float(np.mean(rs)),P=pred)

def best_any(pred,ref,min_ovl=4):
    Lp,Lr=pred.shape[1],ref.shape[1]; out=[]
    for tag,P in [("fwd",pred),("rc",rcp(pred))]:
        for off in range(-Lp+1,Lr):
            i0,i1=max(0,-off),min(Lp,Lr-off)
            if i1-i0<min_ovl: continue
            rs=[colr(P[:,i],ref[:,i+off]) for i in range(i0,i1)]
            rs=[x for x in rs if not np.isnan(x)]
            if rs: out.append((float(np.mean(rs)),tag,i1-i0))
    out.sort(reverse=True); return out[0] if out else (float("nan"),"-",0)

CASES={
 "SOHLH1": dict(fam="bHLH", crop="40–130 (DBD 53–111)", fid=3, ref="CACGTG",
   cite="Suzuki 2014, PLOS One e101681 · reporter + ChIP",
   note="SOHLH2/SOHLH1 heterodimer; no EMSA; CACGTG is palindromic"),
 "ADNP": dict(fam="C2H2 zinc fingers 5–9", crop="447–686 (ZF5–9 masked)", fid=2, ref="GCCCCCTGGAG",
   cite="Ostapcuk 2018, Nature 557:739 · ChAHP complex",
   note="ADNP binds via zinc fingers, not its atypical homeobox"),
}
for g,C in CASES.items():
    core=np.load(f"{OUT}/{g}_moebase_core.npy"); ref=np.load(f"{OUT}/{g}_ref.npy")
    A=display_align(core,ref)
    rb,tb,nb_=best_any(core,ref)
    _,_,o_ap,r_ap=align_pwm(core,ref,max_shift=12,consider_revcomp=True)
    Lr=ref.shape[1]; Lp=A["P"].shape[1]; off=A["off"]
    lo=min(0,off); hi=max(Lr,off+Lp); W=hi-lo
    predF=np.full((4,W),0.25,np.float32); refF=np.full((4,W),0.25,np.float32)
    predF[:,off-lo:off-lo+Lp]=A["P"]; refF[:,-lo:-lo+Lr]=ref
    pshade=[j for j in range(W) if not (0<=j+lo<Lr)]
    rshade=[j for j in range(W) if not (off<=j+lo<off+Lp)]

    def logo(ax,p,shade,title,sub,c):
        q=np.clip(p,1e-8,1);ic=np.maximum(2+(q*np.log2(q)).sum(0),0)
        # shade FIRST: logomaker glyphs share zorder with axvspan, so a later rectangle hides them
        for j in shade: ax.axvspan(j-0.5,j+0.5,color="0.93",zorder=-10,lw=0)
        ax.set_axisbelow(True)
        logomaker.Logo(pd.DataFrame((p*ic).T,columns=list("ACGT")),ax=ax,color_scheme="classic",
                       show_spines=False,vpad=0.02)
        ax.set_xticks(range(W)); ax.set_xticklabels([str(j+lo+1) for j in range(W)],fontsize=7)
        ax.set_yticks([0,1,2]); ax.set_ylabel("bits",fontsize=9); ax.set_ylim(0,2); ax.set_xlim(-0.6,W-0.4)
        ax.set_title(title,fontsize=11,fontweight="bold",color=c,loc="left",pad=15)
        ax.text(0.0,1.02,sub,transform=ax.transAxes,fontsize=8.3,color="#444",va="bottom")

    orient_txt=" (reverse complement)" if A["orient"]=="rc" else ""
    fig,axes=plt.subplots(2,1,figsize=(max(5.2,0.75*W+2.2),5.4))
    logo(axes[0],predF,pshade,f"TFScope prediction — {g} ({C['fam']})",
         f"MoE-base, sequence only · crop {C['crop']} · gate core, no IC-trim · {cons(core)}{orient_txt} · "
         f"GC {gc(core):.2f} · mean IC {meanic(core):.2f}","#0072B2")
    logo(axes[1],refF,rshade,f"Experimental reference — {C['ref']}",
         f"{C['cite']} · {C['note']} · GC {gc(ref):.2f}","#c0392b")
    cap=(f"Forward alignment, all {A['n']} predicted columns on {A['n']}/{Lr} reference columns "
         f"(coverage {A['cov']:.0%}):  r = {A['r']:.2f}.   Grey = not scored.\n"
         f"Caveat: the best-scoring alignment over orientation×shift is {tb} with {nb_} columns, "
         f"r = {rb:.2f} (align_pwm default reports r = {r_ap:.2f}); r does not discriminate.")
    fig.suptitle(f"{g}: sequence-only motif prediction vs experiment",fontsize=12,fontweight="bold",y=1.02)
    fig.text(0.5,-0.055,cap,ha="center",fontsize=8.2,color="#333")
    fig.tight_layout()
    d=f"figures/figure_{g.lower()}_logo"
    for ext in ["png","pdf"]: fig.savefig(f"{d}/{g.lower()}_logo.{ext}",dpi=220,bbox_inches="tight")
    print(f"{g:7s} DISPLAY fwd off=0 overlap={A['n']}/{Lr} r={A['r']:.3f} | best-any {tb} {nb_}col r={rb:.3f} "
          f"| align_pwm r={r_ap:.3f}  -> {d}/{g.lower()}_logo.png",flush=True)
