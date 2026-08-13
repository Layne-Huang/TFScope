#!/usr/bin/env python
"""Exact-consensus (k-mer) version of the cCRE test — no PWM, no MOODS threshold.
Counts overlapping exact matches of a string and its reverse complement.

Scans FOUR strings:
  TFScope predicted consensus : SOHLH1 CGCGGTG, ADNP ATCCCC
  Experimental reference      : SOHLH1 CACGTG,  ADNP GCCCCCTGGAG   <- positive control
Baselines identical to the PWM run: (b) whole genome, (c) 100 exact Altschul-Erikson dinuc shuffles.
"""
import os,sys,json,re,time
sys.path.insert(0,"scripts")
import numpy as np
from lib_dinuc_shuffle import dinuc_shuffle
OUT="results/sohlh1_adnp_case"
GEN="/data1/leihuang/WholeGenomeFasta/genome.fa"
PROM="/data1/leihuang/cre_scan/prom.fa"; ENH="/data1/leihuang/cre_scan/enh.fa"
NSHUF=100; rng=np.random.default_rng(0)
rc=lambda s:s[::-1].translate(str.maketrans("ACGT","TGCA"))
WORDS={"SOHLH1_pred":"CGCGGTG","SOHLH1_ref":"CACGTG","ADNP_pred":"ATCCCC","ADNP_ref":"GCCCCCTGGAG"}
PAT={}
for k,w in WORDS.items():
    ws={w,rc(w)}                                  # palindromes collapse to one pattern
    PAT[k]=re.compile("|".join(f"(?=({x}))" for x in sorted(ws)))
    print(f"  {k:12s} {w:12s} rc={rc(w):12s} {'PALINDROMIC' if rc(w)==w else ''} strands={len(ws)}")
def count(seq,k): return sum(1 for _ in PAT[k].finditer(seq))
def read_fa(p): return [l.strip().upper() for l in open(p) if not l.startswith(">") and l.strip()]

print("\n[1/3] genome ...",flush=True); t0=time.time()
CH={f"chr{c}" for c in list(range(1,23))+["X","Y"]}
gc_={k:0 for k in WORDS}; gbp=0; name=None; buf=[]
def flush(nm,parts):
    global gbp
    if nm not in CH or not parts: return
    s="".join(parts).upper(); gbp+=sum(1 for c in s if c in "ACGT")
    for k in WORDS: gc_[k]+=count(s,k)
for line in open(GEN):
    if line.startswith(">"): flush(name,buf); name=line[1:].split()[0]; buf=[]
    else: buf.append(line.strip())
flush(name,buf)
gmb={k:gc_[k]/(gbp/1e6) for k in WORDS}
print(f"  {gbp/1e6:.0f} Mb non-N ({time.time()-t0:.0f}s)")
for k in WORDS: print(f"    {k:12s} {gc_[k]:9d} sites   {gmb[k]:9.2f} /Mb",flush=True)

print("\n[2/3] cCREs ...",flush=True)
sets={"promoter":read_fa(PROM),"enhancer":read_fa(ENH)}
res={"words":WORDS,"genome_per_Mb":gmb,"genome_sites":gc_,"vs_genome":{},"vs_shuffle":{}}
obs={}; nbp={}
for s_,seqs in sets.items():
    nbp[s_]=sum(len(x) for x in seqs); big="N"*20; joined=big.join(seqs)
    obs[s_]={k:count(joined,k) for k in WORDS}
    res["vs_genome"][s_]={k:dict(obs=obs[s_][k],per_Mb=obs[s_][k]/(nbp[s_]/1e6),
                                 enrich=(obs[s_][k]/(nbp[s_]/1e6))/gmb[k] if gmb[k]>0 else float("nan")) for k in WORDS}
    print(f"  {s_}: {nbp[s_]/1e6:.2f} Mb  {obs[s_]}",flush=True)

print(f"\n[3/3] {NSHUF} exact dinucleotide shuffles ...",flush=True)
for s_,seqs in sets.items():
    null={k:[] for k in WORDS}
    for i in range(NSHUF):
        j=("N"*20).join(dinuc_shuffle(x,rng) for x in seqs)
        for k in WORDS: null[k].append(count(j,k))
        if i%25==0: print(f"   {s_} {i}/{NSHUF} ({time.time()-t0:.0f}s)",flush=True)
    res["vs_shuffle"][s_]={}
    for k in WORDS:
        nd=np.array(null[k],float); mu=nd.mean(); sd=nd.std()+1e-9
        res["vs_shuffle"][s_][k]=dict(obs=obs[s_][k],exp=round(float(mu),2),
            enrich=float(obs[s_][k]/mu) if mu>0 else float("nan"),
            z=float((obs[s_][k]-mu)/sd), emp_p=float((1+(nd>=obs[s_][k]).sum())/(NSHUF+1)))
json.dump(res,open(f"{OUT}/consensus_cre.json","w"),indent=1)
print("\n=== EXACT-CONSENSUS RESULT ===")
print(f"{'string':26s} {'set':9s} {'obs':>7s} {'vs genome':>10s} {'vs shuffle':>11s} {'z':>8s}")
for s_ in sets:
    for k in WORDS:
        a=res["vs_genome"][s_][k]; b=res["vs_shuffle"][s_][k]
        print(f"{k+' '+WORDS[k]:26s} {s_:9s} {a['obs']:7d} {a['enrich']:10.2f} {b['enrich']:11.2f} {b['z']:+8.1f}")
