#!/usr/bin/env python
"""cCRE enrichment for the SOHLH1 and ADNP MoE-base motifs, two baselines (as in Fig 3b-c):
  (b) vs WHOLE GENOME  : hits/Mb in ENCODE cCREs / hits/Mb across hg38 (naive, GC-confounded)
  (c) vs DINUC SHUFFLE : hits in cCREs / mean hits in 100 dinucleotide-shuffles of the same cCREs

Motifs = gate cores, no IC-trim: SOHLH1 CGCGGTG (7 bp), ADNP ATCCCC (6 bp).
THRESHOLD: p<1e-3, not 1e-4. A 6-bp motif has best attainable p = 4^-6 = 2.4e-4 > 1e-4, so
threshold_from_p(1e-4) is unreachable and MOODS clamps to the max score (exact matches only),
which would put the two motifs at different effective stringencies. p=1e-3 is reachable for both.
Out: results/sohlh1_adnp_case/cre_enrichment.json + figures/figure_sohlh1_adnp_cre/
"""
import os,sys,json,time
import numpy as np, warnings; warnings.filterwarnings("ignore")
import MOODS.scan, MOODS.tools
OUT="results/sohlh1_adnp_case"; FIGD="figures/figure_sohlh1_adnp_cre"; os.makedirs(FIGD,exist_ok=True)
GEN="/data1/leihuang/WholeGenomeFasta/genome.fa"
PROM="/data1/leihuang/cre_scan/prom.fa"; ENH="/data1/leihuang/cre_scan/enh.fa"
PVAL=1e-3; NSHUF=100; BG=[0.25]*4; rng=np.random.default_rng(0)
TFS=["SOHLH1","ADNP"]
MOT={g:np.load(f"{OUT}/{g}_moebase_core.npy") for g in TFS}
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))
for g in TFS: print(f"  {g}: {cons(MOT[g])} (w={MOT[g].shape[1]})",flush=True)

def matrices(pwm):
    lo=MOODS.tools.log_odds(pwm.tolist(),BG,0.1)
    thr=MOODS.tools.threshold_from_p(lo,BG,PVAL)
    return [lo,MOODS.tools.reverse_complement(lo)],[thr,thr]
MATS={g:matrices(MOT[g]) for g in TFS}
def n_hits(seq,g):
    mats,thr=MATS[g]
    res=MOODS.scan.scan_dna(seq,mats,BG,thr,7)
    return sum(len(r) for r in res)
def read_fa(p): return [l.strip().upper() for l in open(p) if not l.startswith(">") and l.strip()]
def dinuc_shuffle(s):
    s=[c for c in s if c in "ACGT"]
    if len(s)<4: return ""
    e={}
    for a,b in zip(s[:-1],s[1:]): e.setdefault(a,[]).append(b)
    for a in e: rng.shuffle(e[a])
    out=[s[0]]; cur=s[0]; ok=True
    for _ in range(len(s)-1):
        if not e.get(cur): ok=False; break
        nx=e[cur].pop(); out.append(nx); cur=nx
    return "".join(out) if ok and len(out)==len(s) else "".join(rng.permutation(s))

# ---------- 1. genome-wide ----------
print("\n[1/3] scanning hg38 ...",flush=True)
gh={g:0 for g in TFS}; gbp=0; t0=time.time()
CH={f"chr{c}" for c in list(range(1,23))+["X","Y"]}
name=None; buf=[]
def flush(nm,parts):
    global gbp
    if nm not in CH or not parts: return
    s="".join(parts).upper(); gbp+=sum(1 for c in s if c in "ACGT")
    STEP=10_000_000
    for g in TFS:
        L=MOT[g].shape[1]
        for st in range(0,len(s),STEP):
            sub=s[st:st+STEP+L-1]
            if len(sub)>=L: gh[g]+=n_hits(sub,g)
    print(f"    {nm}: {len(s)/1e6:.1f} Mb  cum hits {[gh[g] for g in TFS]}  ({time.time()-t0:.0f}s)",flush=True)
for line in open(GEN):
    if line.startswith(">"):
        flush(name,buf); name=line[1:].split()[0]; buf=[]
    else: buf.append(line.strip())
flush(name,buf)
gmb={g:gh[g]/(gbp/1e6) for g in TFS}
print(f"  genome: {gbp/1e6:.1f} Mb non-N; hits/Mb {[round(gmb[g],1) for g in TFS]}",flush=True)

# ---------- 2. cCREs ----------
print("\n[2/3] scanning cCREs ...",flush=True)
sets={"promoter":read_fa(PROM),"enhancer":read_fa(ENH)}
res={"pval":PVAL,"genome_per_Mb":gmb,"motif":{g:cons(MOT[g]) for g in TFS},
     "genome_bp":gbp,"vs_genome":{},"vs_shuffle":{}}
obs={}; nbp={}
for k,seqs in sets.items():
    nbp[k]=sum(len(s) for s in seqs)
    obs[k]={g:n_hits(("N"*20).join(seqs),g) for g in TFS}
    res["vs_genome"][k]={g:dict(obs=obs[k][g],per_Mb=obs[k][g]/(nbp[k]/1e6),
                                enrich=(obs[k][g]/(nbp[k]/1e6))/gmb[g]) for g in TFS}
    print(f"  {k}: {nbp[k]/1e6:.2f} Mb, hits {obs[k]}",flush=True)

# ---------- 3. dinucleotide shuffles ----------
print(f"\n[3/3] {NSHUF} dinucleotide shuffles ...",flush=True)
for k,seqs in sets.items():
    null={g:[] for g in TFS}
    for i in range(NSHUF):
        big=("N"*20).join(dinuc_shuffle(s) for s in seqs)
        for g in TFS: null[g].append(n_hits(big,g))
        if i%20==0: print(f"    {k} shuffle {i}/{NSHUF} ({time.time()-t0:.0f}s)",flush=True)
    res["vs_shuffle"][k]={}
    for g in TFS:
        nd=np.array(null[g],float); mu=nd.mean(); sd=nd.std()+1e-9
        res["vs_shuffle"][k][g]=dict(obs=obs[k][g],exp=round(float(mu),2),
            enrich=float(obs[k][g]/mu) if mu>0 else float("nan"),
            z=float((obs[k][g]-mu)/sd), emp_p=float((1+(nd>=obs[k][g]).sum())/(NSHUF+1)))
json.dump(res,open(f"{OUT}/cre_enrichment.json","w"),indent=1)
print("\n=== RESULT ===")
for k in sets:
    for g in TFS:
        a=res["vs_genome"][k][g]; b=res["vs_shuffle"][k][g]
        print(f"  {g:7s} {k:9s} vs genome {a['enrich']:5.2f}x | vs shuffle {b['enrich']:5.2f}x (z {b['z']:+6.1f}, p {b['emp_p']:.3f})")

# ---------- figure ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11.2,4.4))
x=np.arange(len(TFS)); w=0.36
for ax,key,title,sub in [
    (axes[0],"vs_genome","b   Naive: cCREs vs whole genome",
     "hit density in cCREs / hit density across hg38 — GC-confounded"),
    (axes[1],"vs_shuffle","c   Composition-controlled: cCREs vs dinucleotide shuffle",
     f"observed / mean of {NSHUF} dinucleotide-preserving shuffles of the same cCREs")]:
    pm=[np.log2(max(res[key]["promoter"][g]["enrich"],1e-3)) for g in TFS]
    en=[np.log2(max(res[key]["enhancer"][g]["enrich"],1e-3)) for g in TFS]
    ax.bar(x-w/2,pm,w,color="#4575b4",label="promoter")
    ax.bar(x+w/2,en,w,color="#d98c4a",label="enhancer")
    if key=="vs_shuffle":
        for i,g in enumerate(TFS):
            for xoff,vals in [(-w/2,pm),(w/2,en)]:
                z=res[key]["promoter" if xoff<0 else "enhancer"][g]["z"]
                if abs(z)>2: ax.text(i+xoff,vals[i]+(0.04 if vals[i]>=0 else -0.16),"*",ha="center",
                                     color="#c00",fontsize=14,fontweight="bold")
    ax.axhline(0,color="k",lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{g}\n{cons(MOT[g])}" for g in TFS],fontsize=9)
    ax.set_ylabel("log2 enrichment"); ax.legend(fontsize=8,frameon=False)
    ax.set_title(title,fontsize=10.5,fontweight="bold",loc="left",pad=20)
    ax.text(0.0,1.02,sub,transform=ax.transAxes,fontsize=8.2,color="#444",va="bottom")
fig.suptitle(f"cCRE localisation of the TFScope (MoE-base) motifs   ·   MOODS p<{PVAL:g}, both strands",
             fontsize=11.5,fontweight="bold",y=1.03)
fig.text(0.5,-0.06,"* = |z|>2 vs the dinucleotide-shuffle null.  ADNP's ATCCCC is C-rich "
   "(low-complexity 0.67); poly-C tracts are themselves enriched in promoters, so panel c should be "
   "read with that caveat.",ha="center",fontsize=8,color="#333")
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(f"{FIGD}/sohlh1_adnp_cre_enrichment.{e}",dpi=200,bbox_inches="tight")
print(f"\nsaved {FIGD}/sohlh1_adnp_cre_enrichment.png",flush=True)
