"""DeepPBS-style mutation-effect heatmaps (Nat Methods 41592_2024_2372 Fig 5 d/l/h/p) for the 4
designs, paired: EXPERIMENTAL (from the SELEX xls) on top, TFScope prediction below, same style.

Metric (both): relative binding vs WT base, log2 scale, WT=0=white.
  experimental: log2( value / WT_value )            value=Median PE/FITC(Normalized), lower=stronger
  TFScope:      log2( P_pred(WTbase) / P_pred(base) ) higher P=stronger
  NEGATIVE => stronger than WT => BLUE ; POSITIVE => weaker => RED.
Rows A,C,G,T (top->bottom). Full heatmap (TFScope uses full PWM placed at its gated-core register).
Out: figures/figure_dbp_heatmap/dbp_tfscope_heatmap.{png,pdf,svg}
"""
import os, sys, json, argparse
os.environ["TORCH_HOME"]="/data1/leihuang/.cache/torch"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
ap=argparse.ArgumentParser(); ap.add_argument("--tau",type=float,default=1.0); ap.add_argument("--vmax",type=float,default=2.5); a=ap.parse_args()
TAU=a.tau; VMAX=a.vmax; dev="cuda:0"
CKDIR="/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42"
OUT="figures_v24/figure_dbp_heatmap"; os.makedirs(OUT,exist_ok=True)
WT="GCAGATCTGCACAT"; L=len(WT); B="ACGT"; BA=np.array(list(B)); B2I={b:i for i,b in enumerate(B)}
ORDER=["DBP005","DBP009","DBP006","DBP035"]
XLS="case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
SHEET={"DBP005":"Extended_Data_Figure_1_C_DBP005","DBP009":"Extended_Data_Figure_1_E_DBP009",
       "DBP006":"Extended_Data_Figure_1_D_DBP006","DBP035":"Extended_Data_Figure_1_G_DBP035"}
VAL="Median PE/FITC (Normalized)"; WTOVERRIDE={"DBP006":0.1202}
e2={e["name"]:e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}
dff=pd.read_parquet("data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"); fn=dff["filename"].astype(str).str.upper()
def fidf(g): s=dff[fn.str.contains(g.upper())]; return int(s["family_id"].mode().iloc[0]) if len(s) else 4

def exp_rel(d):
    df=pd.ExcelFile(XLS).parse(SHEET[d]); wt=df[df.position.astype(str)=="WT"]
    wtv=WTOVERRIDE.get(d) or (float(wt[VAL].iloc[0]) if len(wt) and not pd.isna(wt[VAL].iloc[0]) else None)
    dd=df[df.position.astype(str).str.isdigit()].copy(); dd["position"]=dd.position.astype(int)
    R=np.zeros((4,L))
    for p in range(1,L+1):
        sub=dd[dd.position==p]
        wv=wtv if wtv is not None else float(sub[VAL].median())
        for _,r in sub.iterrows(): R[B2I[str(r.new_base)],p-1]=np.log2(float(r[VAL])/wv)
    return np.clip(R,-VMAX,VMAX)

cfg=TFScopeConfig()
for k,v in json.load(open(CKDIR+"/config.json")).items():
    if hasattr(cfg,k):
        try: setattr(cfg,k,type(getattr(cfg,k))(v))
        except: pass
cfg.use_retrieval=False
m=TFScopeModel(cfg).to(dev).eval(); m.load_state_dict(torch.load(CKDIR+"/ckpt_best.pt",map_location=dev,weights_only=False)["model"],strict=False)
@torch.no_grad()
def predict_full(seq,fid):
    t=torch.tensor([[AA_TO_TOKEN.get(c,4) for c in seq]],dtype=torch.long,device=dev)
    dm=torch.ones(1,len(seq),dtype=torch.bool,device=dev); fi=torch.tensor([fid],device=dev)
    gl,pl,_=m(t,dm,fi,retrieved_pwms=None,retrieved_masks=None,retrieved_sims=None,recog_prior=None)
    z=pl[0].cpu().numpy()/TAU; z=z-z.max(0,keepdims=True); P=np.exp(z); P/=P.sum(0,keepdims=True)
    return P, gl.sigmoid()[0].cpu().numpy()
def rc(M): return M[[3,2,1,0]][:,::-1]
def tf_rel(seq,fid):
    P,gate=predict_full(seq,fid); W=P.shape[1]; T=np.eye(4)[[B2I[c] for c in WT]].T
    gcols=np.where(gate>0.5)[0]
    if len(gcols)<4:
        ic=(P*np.log2(P+1e-9)).sum(0)+2; c=ic.argmax(); gcols=np.arange(max(0,c-4),min(W,c+5))
    lo,hi=gcols.min(),gcols.max()+1; klen=hi-lo
    best=(-1e9,None,None,None,None)
    for strand in ("+","-"):
        Q=P if strand=="+" else rc(P)
        clo = lo if strand=="+" else (W-hi)
        core=Q[:,clo:clo+klen]
        for coff in range(-(klen-1),L):
            sc=sum(float(core[:,j]@T[:,coff+j]) for j in range(klen) if 0<=coff+j<L)
            if sc>best[0]: best=(sc,strand,coff,Q,clo)
    _,strand,coff,Q,clo=best
    full=np.full((4,L),0.25)
    for p in range(L):
        c=clo+(p-coff)
        if 0<=c<W: full[:,p]=Q[:,c]
    R=np.zeros((4,L))
    for p in range(L):
        wb=B2I[WT[p]]; R[:,p]=np.log2((full[wb,p]+1e-6)/(full[:,p]+1e-6))
    return np.clip(R,-VMAX,VMAX), full, strand, coff

EXP={d:exp_rel(d) for d in ORDER}; TF={}; FULL={}
for d in ORDER:
    R,full,strand,coff=tf_rel(e2[d]["prot_seq"], fidf(str(e2[d].get("top_donor","POU2F1"))))
    TF[d]=R; FULL[d]=full
    print(f"{d}: strand {strand} coff {coff} pred={''.join(BA[full.argmax(0)])} | pos12(WT={WT[11]}): "
          + " ".join(f"{b}={'red' if R[B2I[b],11]>0 else 'blue'}" for b in B if b!=WT[11]))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
plt.rcParams.update({"font.size":8,"svg.fonttype":"none","pdf.fonttype":42,"axes.linewidth":0.6})
cmap=LinearSegmentedColormap.from_list("relative_binding",["#727DB7","#D9DBEC","#FFFFFF","#F7D9D7","#E96B68"])
norm=TwoSlopeNorm(vmin=-VMAX,vcenter=0,vmax=VMAX)
def draw(ax,M,title):
    im=ax.imshow(M,aspect="auto",cmap=cmap,norm=norm)
    ax.set_yticks(range(4)); ax.set_yticklabels(list(B),fontsize=7,fontweight="bold")
    ax.set_xticks(range(L)); ax.set_xticklabels(list(WT),fontsize=6.5)
    ax.set_xticks(np.arange(-.5,L,1),minor=True); ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which="minor",color="#CCCCCC",lw=0.6); ax.tick_params(which="minor",length=0); ax.tick_params(length=2)
    for sp in ax.spines.values(): sp.set_color("#888"); sp.set_linewidth(0.6)
    for p in range(L):
        ax.add_patch(plt.Rectangle((p-0.5,B2I[WT[p]]-0.5),1,1,fill=False,ec="black",lw=1.2))
        ax.text(p,B2I[WT[p]],WT[p],ha="center",va="center",fontsize=6,fontweight="bold",color="#222")
    ax.set_title(title,fontsize=8,loc="left",pad=2); return im
import logomaker
fig=plt.figure(figsize=(12,7.0)); gs=fig.add_gridspec(2,2,hspace=0.40,wspace=0.16); im=None
for i,d in enumerate(ORDER):
    sub=gs[i//2,i%2].subgridspec(2,1,height_ratios=[0.85,1.15],hspace=0.10)
    # TFScope predicted logo (placed on the 14 target positions, IC-scaled)
    axl=fig.add_subplot(sub[0]); P=np.clip(FULL[d],1e-9,1); ic=np.maximum(2+(P*np.log2(P)).sum(0),0)
    logomaker.Logo(pd.DataFrame((P*ic).T,columns=list(B)),ax=axl,color_scheme="classic",show_spines=False,vpad=0.02)
    axl.set_xlim(-0.5,L-0.5); axl.set_xticks([]); axl.set_yticks([0,2]); axl.set_ylim(0,2)
    axl.tick_params(length=2,labelsize=6); axl.set_ylabel("bits",fontsize=7)
    axl.set_title(f"{d}  — TFScope predicted (logo)",fontsize=9,fontweight="bold",loc="left",pad=2)
    # experimental mutation-effect heatmap
    axh=fig.add_subplot(sub[1]); im=draw(axh,EXP[d],"experimental (SELEX)")
    axh.set_xlim(-0.5,L-0.5); axh.set_xlabel("position (WT designed target)",fontsize=7)
cb=fig.colorbar(im,ax=fig.axes,fraction=0.018,pad=0.012,ticks=[-VMAX,0,VMAX])
cb.ax.set_yticklabels(["stronger\nthan WT","WT","weaker\nthan WT"],fontsize=6.5)
cb.set_label("Relative binding (experimental, log2 vs WT)",fontsize=7)
fig.suptitle(f"TFScope predicted logo vs experimental mutation-effect heatmap (τ={TAU})",fontsize=11,fontweight="bold",y=0.98)
out=f"{OUT}/dbp_tfscope_heatmap"
for e in ["pdf","svg"]: fig.savefig(f"{out}.{e}",bbox_inches="tight")
fig.savefig(f"{out}.png",dpi=300,bbox_inches="tight")
print(f"saved {out}.{{png,pdf,svg}}  (tau={TAU})")
