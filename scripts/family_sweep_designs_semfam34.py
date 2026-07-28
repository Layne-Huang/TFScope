"""Family-sweep for the SEMFAM34 (semantic rebin34) model on the 4 designs:
condition on each of the 34 semantic families; report predicted consensus + CAC
recovery + core-r. Tests whether the family choice is a STRONG lever for semfam34
(unlike combined, where it was weak) and which family drives the CAC core.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
from tfscope.data.dataset import AA_TO_TOKEN
from eval_full_metrics import aligned_cols, panel

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_semfam34_contact_fixed/rag_seed42/ckpt_best.pt"
FAM = ["C2H2_short","C2H2_medium","C2H2_long","bHLH","Homeodomain","bZIP","Nuclear_Receptor",
       "Forkhead","ETS","AP2/ERF","HMG/SOX","T-box","RHD/NFkB","E2F/DP","MADS/SRF","GATA","RFX",
       "STAT","p53","MYB/SANT","PAX","Runt","Grainyhead/CP2","DMRT","TEA/TEAD","NF-Y/CBF","IRF",
       "NDT80","MBD","GCM","THAP","HTH","ARID/SAND","Other"]
DESIGNS = ["DBP005","DBP006","DBP009","DBP035"]
SHEET = {"DBP005":"Extended_Data_Figure_1_C_DBP005","DBP006":"Extended_Data_Figure_1_D_DBP006",
         "DBP009":"Extended_Data_Figure_1_E_DBP009","DBP035":"Extended_Data_Figure_1_G_DBP035"}
XLS = "case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
B2I = {"A":0,"C":1,"G":2,"T":3}; BASES = np.array(list("ACGT"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
e2 = {e["name"]: e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}

def exp_pref(d):
    df = pd.ExcelFile(XLS).parse(SHEET[d]); vc = df.columns[-1]
    df = df[df.position.astype(str).str.fullmatch(r"\d+")].copy(); df["p"]=df.position.astype(int)
    L=int(df.p.max()); P=np.full((4,L),1e-6,np.float32)
    for p in range(1,L+1):
        sub=df[df.p==p]; ob=str(sub.original_base.iloc[0])
        if ob in B2I: P[B2I[ob],p-1]=1.0
        for _,r in sub.iterrows():
            nb=str(r["new_base"])
            if nb in B2I: P[B2I[nb],p-1]=max(0.0,float(r[vc]))
    return P/P.sum(0,keepdims=True)
prefs={d:exp_pref(d) for d in DESIGNS}

cfg=TFScopeConfig()
for k,v in json.load(open(os.path.join(os.path.dirname(CKPT),"config.json"))).items():
    if hasattr(cfg,k):
        try: setattr(cfg,k,type(getattr(cfg,k))(v))
        except: pass
cfg.use_retrieval=False
m=TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT,map_location=dev,weights_only=False)["model"],strict=False)

@torch.no_grad()
def predict(seq,fid):
    t=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in seq]],dtype=torch.long,device=dev)
    dm=torch.ones(1,len(seq),dtype=torch.bool,device=dev); fi=torch.tensor([fid],dtype=torch.long,device=dev)
    gl,pl,_=m(t,dm,fi,retrieved_pwms=None,retrieved_masks=None,retrieved_sims=None,recog_prior=None)
    gate=gl.sigmoid()[0].cpu().numpy(); pwm=F.softmax(pl,1)[0].cpu().numpy()
    return pwm[:,:max(4,int((gate>0.5).sum()))]

def assess(pwm,core):
    _,sh,ori,_=align_pwm(pwm,core,max_shift=10,consider_revcomp=True,min_overlap=4)
    p=revcomp_pwm_np(pwm) if ori=="rc" else pwm
    con="".join(BASES[p.argmax(0)])
    aligned,cols,_=aligned_cols(pwm,core); dd=panel(core,aligned,cols)
    cac=("CAC" in con) or ("GTG" in con) or ("CACA" in con) or ("TGTG" in con)
    return con,cac,(round(float(dd["r"]),3) if dd else None)

out={}; KEY={4,6,31}  # Homeodomain, NR, HTH -- highlight
for d in DESIGNS:
    print(f"\n=== {d} ===")
    seq=e2[d]["prot_seq"]; out[d]={}; rr=[]
    for fid in range(34):
        con,cac,r=assess(predict(seq,fid),prefs[d])
        out[d][FAM[fid]]={"consensus":con,"cac":bool(cac),"core_r":r}; rr.append(r if r else 0)
        if fid in KEY or cac:
            print(f"  fam {fid:2d} {FAM[fid]:16s}: {con:18s} core_r={r}  {'<== CAC' if cac else ''}")
    rr=np.array(rr); print(f"   core_r across 34 families: min {rr.min():.3f}  max {rr.max():.3f}  spread {rr.max()-rr.min():.3f}")
json.dump(out,open("results/design_case_study/family_sweep_semfam34.json","w"),indent=1)
print("\n=== families recovering CAC (count over 4 designs) ===")
cnt={FAM[f]:sum(out[d][FAM[f]]["cac"] for d in DESIGNS) for f in range(34)}
for k,v in sorted(cnt.items(),key=lambda x:-x[1])[:12]:
    if v>0: print(f"  {k:16s}: {v}/4")
print("saved results/design_case_study/family_sweep_semfam34.json")
