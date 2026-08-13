#!/usr/bin/env python
"""Replot the SOHLH1/ADNP cCRE enrichment panel from results/sohlh1_adnp_case/cre_enrichment.json.
Each bar is labelled with its enrichment and z. Enrichment (z>+2) = red, depletion (z<-2) = blue,
n.s. = grey — because here the composition control produces DEPLETIONS, not enrichments."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=json.load(open("results/sohlh1_adnp_case/cre_enrichment.json"))
TFS=["SOHLH1","ADNP"]; MOT=R["motif"]; FIGD="figures/figure_sohlh1_adnp_cre"
fig,axes=plt.subplots(1,2,figsize=(11.6,4.8))
x=np.arange(len(TFS)); w=0.36
def lab(ax,xs,ys,txts,cols):
    rng=max(abs(np.array(ys)).max(),0.05)
    for xi,yi,t,c in zip(xs,ys,txts,cols):
        va="bottom" if yi>=0 else "top"
        ax.text(xi,yi+(0.035*rng if yi>=0 else -0.035*rng),t,ha="center",va=va,fontsize=7.6,color=c)
# panel b
ax=axes[0]
pm=[np.log2(R["vs_genome"]["promoter"][g]["enrich"]) for g in TFS]
en=[np.log2(R["vs_genome"]["enhancer"][g]["enrich"]) for g in TFS]
ax.bar(x-w/2,pm,w,color="#4575b4",label="promoter"); ax.bar(x+w/2,en,w,color="#d98c4a",label="enhancer")
lab(ax,list(x-w/2)+list(x+w/2),pm+en,
    [f"{R['vs_genome']['promoter'][g]['enrich']:.2f}×" for g in TFS]+
    [f"{R['vs_genome']['enhancer'][g]['enrich']:.2f}×" for g in TFS],["#333"]*4)
ax.set_title("b   Naive: cCREs vs whole genome",fontsize=10.5,fontweight="bold",loc="left",pad=20)
ax.text(0,1.02,"hit density in cCREs / hit density across hg38 — confounded by GC content",
        transform=ax.transAxes,fontsize=8.2,color="#444",va="bottom")
# panel c
ax=axes[1]
def col(z): return "#c0392b" if z>2 else ("#2c6fbb" if z<-2 else "#999")
pmz=[R["vs_shuffle"]["promoter"][g]["z"] for g in TFS]; enz=[R["vs_shuffle"]["enhancer"][g]["z"] for g in TFS]
pm2=[np.log2(R["vs_shuffle"]["promoter"][g]["enrich"]) for g in TFS]
en2=[np.log2(R["vs_shuffle"]["enhancer"][g]["enrich"]) for g in TFS]
ax.bar(x-w/2,pm2,w,color="#4575b4",label="promoter",edgecolor=[col(z) for z in pmz],linewidth=2)
ax.bar(x+w/2,en2,w,color="#d98c4a",label="enhancer",edgecolor=[col(z) for z in enz],linewidth=2)
lab(ax,list(x-w/2)+list(x+w/2),pm2+en2,
    [f"{R['vs_shuffle']['promoter'][g]['enrich']:.2f}×\nz={pmz[i]:+.1f}" for i,g in enumerate(TFS)]+
    [f"{R['vs_shuffle']['enhancer'][g]['enrich']:.2f}×\nz={enz[i]:+.1f}" for i,g in enumerate(TFS)],
    [col(z) for z in pmz]+[col(z) for z in enz])
ax.set_title("c   Composition-controlled: cCREs vs dinucleotide shuffle",fontsize=10.5,fontweight="bold",loc="left",pad=20)
ax.text(0,1.02,f"observed / mean of {R.get('nshuf',100)} dinucleotide-preserving shuffles of the same cCREs",
        transform=ax.transAxes,fontsize=8.2,color="#444",va="bottom")
for ax in axes:
    ax.axhline(0,color="k",lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{g}\n{MOT[g]}" for g in TFS],fontsize=9.5)
    ax.set_ylabel("log2 enrichment"); ax.legend(fontsize=8,frameon=False,loc="best")
    ax.margins(y=0.30)
fig.suptitle(f"cCRE localisation of the TFScope (MoE-base) motifs   ·   MOODS p<{R['pval']:g}, both strands",
             fontsize=11.5,fontweight="bold",y=1.02)
fig.text(0.5,-0.13,
 "Red outline = enriched (z>+2); blue outline = depleted (z<−2); grey = n.s.  Null = 100 exact Altschul–Erikson\n"
 "dinucleotide shuffles (empirical p floors at 1/101 = 0.010, so z is the discriminator).\n"
 "SOHLH1's 4.26× promoter signal in panel b is largely GC composition; controlling for it leaves a modest but highly\n"
 "significant 1.23× (promoter) and 1.35× (enhancer) enrichment. ADNP's ATCCCC (3.4 bits, C-rich) is not enriched.",
 ha="center",fontsize=8.1,color="#333")
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(f"{FIGD}/sohlh1_adnp_cre_enrichment.{e}",dpi=200,bbox_inches="tight")
print("saved",f"{FIGD}/sohlh1_adnp_cre_enrichment.png")
