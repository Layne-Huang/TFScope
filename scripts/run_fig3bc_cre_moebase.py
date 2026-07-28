#!/usr/bin/env python
"""Fig 3b-c CRE-enrichment experiment, re-run with MoE-BASE motifs.
Composition-controlled: scan each orphan motif (MOODS, p<1e-4, both strands) in ENCODE
cCRE promoter/enhancer sequences vs N dinucleotide-shuffles of those same sequences.
Motif per TF = MoE-base all-DBD prediction, scan-window = highest-IC contiguous span (>=8 bp).
Out: results/genome_cre_scan/cre_shuffle_moebase.json + figures/figure3bc_cre_moebase/.
"""
import os, sys, json
os.environ["TORCH_HOME"]="/data1/leihuang/.cache/torch"; os.environ["HF_HOME"]="/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0,"src"); sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/case_study")
import numpy as np, torch
import warnings; warnings.filterwarnings("ignore")
import MOODS.scan, MOODS.tools
from cs_utils import load_model, active_cols, infer, device
from tfscope.data.dataset import AA_TO_TOKEN

CKPT="/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42/ckpt_best.pt"
ORDER=["SOHLH1","ADNP","ADNP2","ZHX2","ZHX3","ZGLP1"]
FID={"SOHLH1":3,"ADNP":4,"ADNP2":4,"ZHX2":4,"ZHX3":4,"ZGLP1":9}
FEAT=json.load(open("results/orphan_multidomain/dbd_features.json"))
OUTD="results/orphan_multidomain"; RES="results/genome_cre_scan"
FIGD="figures/figure3bc_cre_moebase"; os.makedirs(FIGD,exist_ok=True)
PROM="/data1/leihuang/cre_scan/prom.fa"; ENH="/data1/leihuang/cre_scan/enh.fa"
PVAL=1e-4; NSHUF=100; MINW=8; rng=np.random.default_rng(0)

def read_fa(p): return [l.strip().upper() for l in open(p) if not l.startswith(">") and l.strip()]
def scan_window(pwm):   # highest-IC contiguous window, >= MINW (pad if shorter)
    q=np.clip(pwm,1e-8,1); ic=2+(q*np.log2(q)).sum(0); L=pwm.shape[1]
    w=max(MINW,int((ic>=0.25).sum())); w=min(w,L)
    best=(0,-1)
    for s in range(0,L-w+1):
        tot=ic[s:s+w].sum()
        if tot>best[1]: best=(s,tot)
    s=best[0]; return pwm[:,s:s+w]
def dinuc_shuffle(seq):  # Altschul-Erikson eulerian dinucleotide-preserving shuffle
    s=[c for c in seq if c in "ACGT"]
    if len(s)<4: return seq
    edges={};
    for a,b in zip(s[:-1],s[1:]): edges.setdefault(a,[]).append(b)
    for a in edges: rng.shuffle(edges[a])
    # ensure last-edge-to-terminal spanning tree (simple retry)
    last=s[-1]; out=[s[0]]; cur=s[0]; e={k:list(v) for k,v in edges.items()}
    ok=True
    for _ in range(len(s)-1):
        if not e.get(cur): ok=False; break
        nxt=e[cur].pop(); out.append(nxt); cur=nxt
    return "".join(out) if ok and len(out)==len(s) else "".join(rng.permutation(s))

print("loading MoE-base ...",flush=True)
model,_=load_model(CKPT,force_retrieval=False)
def predict(g):
    seq=open(f"{OUTD}/{g}.seq.txt").read().strip()
    dbds=[(a,b) for a,b,_,_ in FEAT[g]["dbds"]]; le=max(b for _,b in dbds); fs=min(a for a,_ in dbds)
    end=min(len(seq),le+3); start=max(1,fs-3)
    if end-start+1>1000: start=end-1000+1
    win=seq[start-1:end]; mk=[False]*len(win)
    for a,b in dbds:
        for i in range(a,b+1):
            j=i-start
            if 0<=j<len(win): mk[j]=True
    tok=torch.tensor([[AA_TO_TOKEN.get(a,4) for a in win]],dtype=torch.long,device=device)
    mask=torch.tensor([mk],dtype=torch.bool,device=device)
    gg,pp,_=infer(model,tok,mask,FID[g],ret=None)
    return scan_window(pp[:,active_cols(gg,0.5)] if active_cols(gg,0.5).sum()>=MINW else pp)
MOTIF={g:predict(g) for g in ORDER}
for g in ORDER: print(f"  {g:7s} scan-motif width {MOTIF[g].shape[1]}  {''.join('ACGT'[i] for i in MOTIF[g].argmax(0))}",flush=True)

BG=[0.25,0.25,0.25,0.25]
def n_hits(seqs,pwm):
    lo=MOODS.tools.log_odds(pwm.tolist(),BG,0.1)
    thr=MOODS.tools.threshold_from_p(lo,BG,PVAL)
    rc=MOODS.tools.reverse_complement(lo)
    big=("N"*20).join(seqs)
    res=MOODS.scan.scan_dna(big,[lo,rc],BG,[thr,thr],7)
    return sum(len(r) for r in res)

def enrich_set(seqs,name):
    out={}
    obs={g:n_hits(seqs,MOTIF[g]) for g in ORDER}
    null={g:[] for g in ORDER}
    for k in range(NSHUF):
        sh=[dinuc_shuffle(s) for s in seqs]
        for g in ORDER: null[g].append(n_hits(sh,MOTIF[g]))
        if k%25==0: print(f"    {name} shuffle {k}/{NSHUF}",flush=True)
    for g in ORDER:
        nd=np.array(null[g],float); mu=nd.mean(); sd=nd.std()+1e-9
        out[g]=dict(obs=int(obs[g]),exp=round(float(mu),2),
                    enrich=float(obs[g]/mu) if mu>0 else float('nan'),
                    z=float((obs[g]-mu)/sd),
                    emp_p=float((1+(nd>=obs[g]).sum())/(NSHUF+1)))
    return out

prom=read_fa(PROM); enh=read_fa(ENH)
print(f"scanning {len(prom)} promoters + {len(enh)} enhancers, {NSHUF} shuffles ...",flush=True)
result={"promoter":enrich_set(prom,"prom"),"enhancer":enrich_set(enh,"enh")}
json.dump(result,open(f"{RES}/cre_shuffle_moebase.json","w"),indent=1)
print("\n=== composition-controlled enrichment (MoE-base) ===")
for g in ORDER:
    p=result["promoter"][g]; e=result["enhancer"][g]
    print(f"  {g:7s} prom {p['enrich']:.2f}x (z={p['z']:+.1f}) | enh {e['enrich']:.2f}x (z={e['z']:+.1f})",flush=True)

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x=np.arange(len(ORDER)); w=0.38
pm=[np.log2(max(result["promoter"][g]["enrich"],1e-3)) for g in ORDER]
en=[np.log2(max(result["enhancer"][g]["enrich"],1e-3)) for g in ORDER]
fig,ax=plt.subplots(figsize=(9,4.2))
ax.bar(x-w/2,pm,w,color="#4575b4",label="promoter"); ax.bar(x+w/2,en,w,color="#d98c4a",label="enhancer")
for i,g in enumerate(ORDER):
    if result["promoter"][g]["z"]>2: ax.text(i-w/2,pm[i]+0.05,"*",ha="center",color="#c00",fontsize=13,fontweight="bold")
    if result["enhancer"][g]["z"]>2: ax.text(i+w/2,en[i]+0.05,"*",ha="center",color="#c00",fontsize=13,fontweight="bold")
ax.axhline(0,color="k",lw=0.7); ax.set_xticks(x); ax.set_xticklabels(ORDER,fontsize=9)
ax.set_ylabel("log2 enrichment vs dinucleotide-shuffle"); ax.legend(fontsize=8,frameon=False)
ax.set_title("Fig 3b-c (MoE-base): orphan motifs vs composition-matched cCREs",fontsize=11,fontweight="bold")
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(f"{FIGD}/fig3bc_cre_moebase.{e}",dpi=200,bbox_inches="tight")
print("\nsaved",f"{FIGD}/fig3bc_cre_moebase.png",flush=True)
