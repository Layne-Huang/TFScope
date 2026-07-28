#!/usr/bin/env python
"""Exact-consensus cCRE enrichment: TFScope prediction vs EXPERIMENTAL REFERENCE (positive control).
Reads results/sohlh1_adnp_case/consensus_cre.json (scripts/run_consensus_cre.py).
Panel b = naive vs whole genome (GC-confounded). Panel c = vs 100 exact Altschul-Erikson
dinucleotide shuffles. ADNP's reference occurs 6/2 times -> greyed out, no statistical power.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
R=json.load(open("results/sohlh1_adnp_case/consensus_cre.json")); W=R["words"]
ORDER=[("SOHLH1_ref","SOHLH1","reference"),("SOHLH1_pred","SOHLH1","TFScope"),
       ("ADNP_ref","ADNP","reference"),("ADNP_pred","ADNP","TFScope")]
MINOBS=20                                  # below this, no power
PROM,ENH="#3b6fb0","#d98c4a"
fig,axes=plt.subplots(1,2,figsize=(13.4,5.6),gridspec_kw=dict(wspace=0.28))
x=np.arange(len(ORDER)); w=0.36
for ax,key,title,sub in [
 (axes[0],"vs_genome","b   Naive: cCREs vs whole genome",
  "hit density in cCREs / hit density across hg38 — confounded by GC content"),
 (axes[1],"vs_shuffle","c   Composition-controlled: cCREs vs dinucleotide shuffle",
  "observed / mean of 100 exact Altschul–Erikson dinucleotide shuffles of the same cCREs")]:
    for si,(setn,base,off) in enumerate([("promoter",PROM,-w/2),("enhancer",ENH,+w/2)]):
        vals=[np.log2(max(R[key][setn][k]["enrich"],1e-3)) for k,_,_ in ORDER]
        obs =[R[key][setn][k]["obs"] for k,_,_ in ORDER]
        low =[o<MINOBS for o in obs]
        cols=[("0.80" if lo else base) for lo in low]
        bars=ax.bar(x+off,vals,w,color=cols,edgecolor="0.25",linewidth=0.6,
                    hatch=["" if t=="reference" else "//" for _,_,t in ORDER])
        rng=max(abs(np.array(vals)).max(),0.2)
        for i,(v,o,lo) in enumerate(zip(vals,obs,low)):
            if key=="vs_shuffle":
                z=R[key][setn][ORDER[i][0]]["z"]; e=R[key][setn][ORDER[i][0]]["enrich"]
                t=f"{e:.2f}×\nz={z:+.1f}" if not lo else f"n={o}\nno power"
                c="#999" if lo else ("#c0392b" if z>2 else ("#2c6fbb" if z<-2 else "#666"))
            else:
                e=R[key][setn][ORDER[i][0]]["enrich"]
                t=f"{e:.2f}×" if not lo else f"n={o}"
                c="#999" if lo else "#333"
            ax.text(x[i]+off, v+(0.045*rng if v>=0 else -0.045*rng), t, ha="center",
                    va="bottom" if v>=0 else "top", fontsize=7.4, color=c)
    ax.axhline(0,color="k",lw=0.9); ax.axvline(1.5,color="0.7",lw=1,ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{W[k]}\n{tf} {t}" for k,tf,t in ORDER],fontsize=8.4)
    ax.set_ylabel("log2 enrichment")
    ax.set_title(title,fontsize=10.8,fontweight="bold",loc="left",pad=22)
    ax.text(0,1.02,sub,transform=ax.transAxes,fontsize=8.2,color="#444",va="bottom")
    ax.margins(y=0.34)
axes[1].legend(handles=[Patch(fc=PROM,ec="0.25",label="promoter"),Patch(fc=ENH,ec="0.25",label="enhancer"),
    Patch(fc="w",ec="0.25",label="experimental reference"),Patch(fc="w",ec="0.25",hatch="//",label="TFScope prediction"),
    Patch(fc="0.80",ec="0.25",label=f"<{MINOBS} sites — no power")],fontsize=7.8,frameon=False,
    loc="upper right",ncol=1)
fig.suptitle("Exact-consensus cCRE enrichment, with the experimental motif as a positive control",
             fontsize=12.5,fontweight="bold",y=1.015)
fig.text(0.5,-0.14,
 "The validated E-box CACGTG is strongly enriched (2.31× promoter, 3.32× enhancer) — the assay detects a real motif.\n"
 "TFScope's CGCGGTG is enriched in the same direction but at lower amplitude (1.34× / 2.49×): a blurred CACGTG, not a random GC word.\n"
 "ADNP's reference GCCCCCTGGAG occurs only 6× (promoter) and 2× (enhancer), so the positive control has no power and ADNP cannot be\n"
 "tested by this assay; its predicted ATCCCC is a generic 6-mer with 1,024,921 genomic sites. Empirical p floors at 1/101 → z is the discriminator.",
 ha="center",fontsize=8.2,color="#333")
fig.tight_layout()
for e in ["png","pdf"]:
    fig.savefig(f"figures/figure_consensus_cre/consensus_cre_enrichment.{e}",dpi=200,bbox_inches="tight")
print("saved figures/figure_consensus_cre/consensus_cre_enrichment.png")
