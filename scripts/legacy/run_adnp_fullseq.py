#!/usr/bin/env python
"""Naive FULL-LENGTH ADNP input (whole protein, no DBD cropping), combined + MoE versions.
OOD test (TFScope trained on DBD crops). Truncated to 1000 aa for the ESM2 length limit;
mask = all residues. Scored vs GCCCCCTGGAG (Nature 2018) and JASPAR UN0305.1.
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

SEQ=open("/tmp/adnp_seq.txt").read().strip()[:1000]   # truncate for ESM2 (keeps all ZFs 74-686 + homeobox 754-814)
MODELS={"combined":"/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt",
        "MoE(residue base)":"/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42/ckpt_best.pt",
        "MoE(deeptune)":"/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe_deeptune/deeptune_seed42/ckpt_best.pt"}
FID=4
def onehot(c):
    P=np.full((4,len(c)),0.02,np.float32)
    for j,ch in enumerate(c): P["ACGT".index(ch),j]=0.94
    return P
NAT=onehot("GCCCCCTGGAG"); UN=np.load("/tmp/un0305_pwm.npy")
def ict(p,t=0.25):
    p=np.clip(p,1e-8,1);ic=2+(p*np.log2(p)).sum(0);i=np.where(ic>=t)[0]
    return p if len(i)==0 else p[:,i[0]:i[-1]+1]
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))
def rc(s): return s[::-1].translate(str.maketrans("ACGT","TGCA"))
def gc(p): return float(p[[1,2]].sum(0).mean())
def r(a,b):
    _,_,_,x=align_pwm(ict(a),ict(b),max_shift=12,consider_revcomp=True); return float(x)
def run(model):
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in SEQ]],dtype=torch.long,device=device)
    mk=torch.ones(1,len(SEQ),dtype=torch.bool,device=device)
    g,p,_=infer(model,tok,mk,FID,ret=None); return ict(p[:,active_cols(g,0.5)])

print(f"FULL ADNP input: {len(SEQ)} aa (truncated), mask=all residues, FID={FID}\n")
print(f"{'model':20s} {'consensus':18s} {'RC':18s} {'GC':>5s} {'r_GCCCCCTGGAG':>14s} {'r_UN0305':>9s}")
for mn,ck in MODELS.items():
    model,_=load_model(ck,force_retrieval=False); p=run(model); c=cons(p)
    print(f"{mn:20s} {c:18s} {rc(c):18s} {gc(p):5.2f} {r(p,NAT):14.3f} {r(p,UN):9.3f}",flush=True)
    del model; torch.cuda.empty_cache()
print("\nexperimental GCCCCCTGGAG: GC=0.79 | UN0305.1 (ChIP-seq, AT/mixed)")
