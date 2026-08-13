#!/usr/bin/env python
"""Re-run ONLY the dinucleotide-shuffle null with the correct Altschul-Erikson shuffle.
The first run used a naive Eulerian walk that dead-ended 75% of the time and silently fell back
to a MONONUCLEOTIDE permutation, which over-produces CpG by ~28% and therefore exaggerated the
apparent depletion of the CpG-rich SOHLH1 motif. Genome-wide hits are unaffected and are reused.
"""
import os,sys,json,time
sys.path.insert(0,"scripts")
import numpy as np, warnings; warnings.filterwarnings("ignore")
import MOODS.scan, MOODS.tools
from lib_dinuc_shuffle import dinuc_shuffle, dinuc_counts
OUT="results/sohlh1_adnp_case"
PROM="/data1/leihuang/cre_scan/prom.fa"; ENH="/data1/leihuang/cre_scan/enh.fa"
old=json.load(open(f"{OUT}/cre_enrichment.json"))
PVAL=old["pval"]; NSHUF=100; BG=[0.25]*4; rng=np.random.default_rng(0)
TFS=["SOHLH1","ADNP"]
MOT={g:np.load(f"{OUT}/{g}_moebase_core.npy") for g in TFS}
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))
def matrices(pwm):
    lo=MOODS.tools.log_odds(pwm.tolist(),BG,0.1)
    return [lo,MOODS.tools.reverse_complement(lo)],[MOODS.tools.threshold_from_p(lo,BG,PVAL)]*2
MATS={g:matrices(MOT[g]) for g in TFS}
def n_hits(seq,g):
    mats,thr=MATS[g]
    return sum(len(r) for r in MOODS.scan.scan_dna(seq,mats,BG,thr,7))
def read_fa(p): return [l.strip().upper() for l in open(p) if not l.startswith(">") and l.strip()]
sets={"promoter":read_fa(PROM),"enhancer":read_fa(ENH)}
res=dict(old); res["vs_shuffle"]={}; res["nshuf"]=NSHUF; res["shuffle"]="Altschul-Erikson (exact dinucleotide)"
t0=time.time()
for k,seqs in sets.items():
    obs={g:n_hits(("N"*20).join(seqs),g) for g in TFS}
    # verify the null preserves dinucleotide composition in aggregate
    o=sum(dinuc_counts(s).get("CG",0) for s in seqs)
    sh=[dinuc_shuffle(s,rng) for s in seqs]
    n=sum(dinuc_counts(s).get("CG",0) for s in sh)
    print(f"{k}: obs {obs} | CpG original {o} vs one shuffle {n} (must match)",flush=True)
    null={g:[] for g in TFS}
    for i in range(NSHUF):
        big=("N"*20).join(dinuc_shuffle(s,rng) for s in seqs)
        for g in TFS: null[g].append(n_hits(big,g))
        if i%25==0: print(f"   {k} shuffle {i}/{NSHUF} ({time.time()-t0:.0f}s)",flush=True)
    res["vs_shuffle"][k]={}
    for g in TFS:
        nd=np.array(null[g],float); mu=nd.mean(); sd=nd.std()+1e-9
        res["vs_shuffle"][k][g]=dict(obs=int(obs[g]),exp=round(float(mu),2),
            enrich=float(obs[g]/mu), z=float((obs[g]-mu)/sd),
            emp_p=float((1+(nd>=obs[g]).sum())/(NSHUF+1)))
json.dump(res,open(f"{OUT}/cre_enrichment.json","w"),indent=1)
print("\n=== CORRECTED (exact dinucleotide null) ===")
for k in sets:
    for g in TFS:
        a=res["vs_genome"][k][g]; b=res["vs_shuffle"][k][g]; ob=old["vs_shuffle"][k][g]
        print(f"  {g:7s} {k:9s} vs genome {a['enrich']:5.2f}x | vs shuffle {b['enrich']:5.2f}x "
              f"(z {b['z']:+6.1f}, p {b['emp_p']:.3f})   [broken null gave {ob['enrich']:.2f}x, z {ob['z']:+.1f}]")
