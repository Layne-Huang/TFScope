#!/usr/bin/env python
"""Fig 3b-c CRE enrichment (MoE-base) with PROPER controls:
  (1) low-complexity masking of cCRE sequences (homopolymer runs + low-entropy windows -> N),
      applied to BOTH observed and dinucleotide-shuffled scans, so homopolymer/poly-A tracts
      cannot drive spurious enrichment;
  (2) a per-motif low-complexity FLAG (max single-base fraction) so we mark motifs that are
      themselves poly-A/low-complexity and whose enrichment is not interpretable.
Motif per TF = MoE-base all-DBD IC-trimmed gate core (the real prediction).
Out: results/genome_cre_scan/cre_shuffle_moebase_masked.json + figures/figure3bc_cre_moebase/.
"""
import os, sys, json
sys.path.insert(0,"src")
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import MOODS.scan, MOODS.tools

ORDER=["SOHLH1","ADNP","ADNP2","ZHX2","ZHX3","ZGLP1"]
OUTD="results/orphan_multidomain"; RES="results/genome_cre_scan"
FIGD="figures/figure3bc_cre_moebase"; os.makedirs(FIGD,exist_ok=True)
PROM="/data1/leihuang/cre_scan/prom.fa"; ENH="/data1/leihuang/cre_scan/enh.fa"
PVAL=1e-4; NSHUF=100; rng=np.random.default_rng(0)

def read_fa(p): return [l.strip().upper() for l in open(p) if not l.startswith(">") and l.strip()]
def lowcomplexity_mask(seq, homrun=5, win=12, dom1=0.72):
    s=list(seq); n=len(s)
    # homopolymer runs >= homrun -> N
    i=0
    while i<n:
        j=i
        while j<n and s[j]==s[i]: j+=1
        if j-i>=homrun:
            for k in range(i,j): s[k]="N"
        i=j
    # low-entropy sliding windows (one base dominates) -> N center
    for c in range(0,n-win+1):
        w=[b for b in seq[c:c+win] if b in "ACGT"]
        if not w: continue
        top=max("ACGT",key=lambda b:w.count(b))
        if w.count(top)/len(w)>=dom1:
            for k in range(c,c+win): s[k]="N"
    return "".join(s)
def motif_lowcomplexity(pwm):  # max single-base fraction over consensus
    cons="".join("ACGT"[i] for i in pwm.argmax(0))
    return max(cons.count(b) for b in "ACGT")/len(cons)
def dinuc_shuffle(seq):
    s=[c for c in seq if c in "ACGT"]
    if len(s)<4: return seq
    e={}
    for a,b in zip(s[:-1],s[1:]): e.setdefault(a,[]).append(b)
    for a in e: rng.shuffle(e[a])
    out=[s[0]]; cur=s[0]
    ok=True
    for _ in range(len(s)-1):
        if not e.get(cur): ok=False; break
        nxt=e[cur].pop(); out.append(nxt); cur=nxt
    return "".join(out) if ok and len(out)==len(s) else "".join(rng.permutation(s))

BG=[0.25]*4
def n_hits(seqs,pwm):
    L=pwm.shape[1]; lo=MOODS.tools.log_odds(pwm.tolist(),BG,0.1)
    thr=MOODS.tools.threshold_from_p(lo,BG,PVAL); rc=MOODS.tools.reverse_complement(lo)
    big=("N"*20).join(seqs)
    res=MOODS.scan.scan_dna(big,[lo,rc],BG,[thr,thr],min(7,L))
    return sum(len(r) for r in res)
def enrich(seqs,pwm,name,g):
    obs=n_hits(seqs,pwm); null=[]
    for k in range(NSHUF):
        null.append(n_hits([dinuc_shuffle(s) for s in seqs],pwm))
        if k%25==0: print(f"    {g}/{name} shuffle {k}/{NSHUF}",flush=True)
    nd=np.array(null,float); mu=nd.mean(); sd=nd.std()+1e-9
    return dict(obs=int(obs),exp=round(float(mu),2),enrich=float(obs/mu) if mu>0 else float('nan'),
                z=float((obs-mu)/sd),emp_p=float((1+(nd>=obs).sum())/(NSHUF+1)))

print("masking low-complexity regions in cCREs ...",flush=True)
prom=[lowcomplexity_mask(s) for s in read_fa(PROM)]
enh =[lowcomplexity_mask(s) for s in read_fa(ENH)]
def frac_N(ss):
    t=sum(len(s) for s in ss); m=sum(s.count("N") for s in ss); return m/t
print(f"  masked fraction: prom {frac_N(prom):.1%}, enh {frac_N(enh):.1%}",flush=True)

MOT={g:np.load(f"{OUTD}/{g}_moebase_alldbd.npy") for g in ORDER}
res={"promoter":{},"enhancer":{}}
flag={}
for g in ORDER:
    lc=motif_lowcomplexity(MOT[g]); flag[g]=lc
    print(f"{g}: motif lowcomplexity={lc:.2f} {'(LOW-COMPLEXITY, unreliable)' if lc>=0.6 else ''}",flush=True)
    res["promoter"][g]=enrich(prom,MOT[g],"prom",g)
    res["enhancer"][g]=enrich(enh,MOT[g],"enh",g)
res["motif_lowcomplexity"]=flag
json.dump(res,open(f"{RES}/cre_shuffle_moebase_masked.json","w"),indent=1)

print("\n=== MASKED + composition-controlled (MoE-base) ===")
for g in ORDER:
    p=res["promoter"][g]; e=res["enhancer"][g]
    tag=" [LOW-COMPLEXITY motif]" if flag[g]>=0.6 else ""
    print(f"  {g:7s} prom {p['enrich']:.2f}x(z{p['z']:+.1f}) enh {e['enrich']:.2f}x(z{e['z']:+.1f}){tag}",flush=True)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x=np.arange(len(ORDER)); w=0.38
pm=[np.log2(max(res["promoter"][g]["enrich"],1e-3)) for g in ORDER]
en=[np.log2(max(res["enhancer"][g]["enrich"],1e-3)) for g in ORDER]
fig,ax=plt.subplots(figsize=(9.5,4.4))
ax.bar(x-w/2,pm,w,color="#4575b4",label="promoter"); ax.bar(x+w/2,en,w,color="#d98c4a",label="enhancer")
for i,g in enumerate(ORDER):
    if res["promoter"][g]["z"]>2: ax.text(i-w/2,pm[i]+0.03,"*",ha="center",color="#c00",fontsize=13,fontweight="bold")
    if res["enhancer"][g]["z"]>2: ax.text(i+w/2,en[i]+0.03,"*",ha="center",color="#c00",fontsize=13,fontweight="bold")
    if flag[g]>=0.6: ax.text(i,min(pm[i],en[i],0)-0.15,"LC",ha="center",color="#888",fontsize=8)
ax.axhline(0,color="k",lw=0.7); ax.set_xticks(x); ax.set_xticklabels(ORDER,fontsize=9)
ax.set_ylabel("log2 enrichment vs dinuc-shuffle\n(low-complexity-masked cCREs)")
ax.legend(fontsize=8,frameon=False)
ax.set_title("Fig 3b-c (MoE-base) — low-complexity-masked; LC = poly-A/low-complexity motif (unreliable)",fontsize=10,fontweight="bold")
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(f"{FIGD}/fig3bc_cre_masked.{e}",dpi=200,bbox_inches="tight")
print("\nsaved",f"{FIGD}/fig3bc_cre_masked.png",flush=True)
