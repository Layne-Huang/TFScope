"""Fig — leave-family-out transfer floor: per-family in-distribution vs held-out (LOFO) oracle-r.
Reads results/lofo/lofo_summary.json (per-family LOFO mean + in-dist r). Out: figures/figure1_lofo/.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="figures/figure1_lofo"; os.makedirs(OUT,exist_ok=True)
S=json.load(open("results/lofo/lofo_summary.json"))
fams=[r["family"] for r in S["rows"]]; indist=[r["c40"] for r in S["rows"]]; lofo=[r["lofo_mean"] for r in S["rows"]]
order=np.argsort(indist)[::-1]; fams=[fams[i] for i in order]; indist=[indist[i] for i in order]; lofo=[lofo[i] for i in order]
floor=S["macro_mean"]
plt.rcParams.update({"font.size":9,"svg.fonttype":"none","pdf.fonttype":42,"axes.linewidth":0.7})
fig,ax=plt.subplots(figsize=(7.2,3.8)); x=np.arange(len(fams)); w=0.38
ax.bar(x-w/2,indist,w,label="in-distribution (cluster40)",color="#7f8c9b",edgecolor="k",lw=0.4)
ax.bar(x+w/2,lofo,w,label="held-out family (LOFO)",color="#D95F4C",edgecolor="k",lw=0.4)
ax.axhline(floor,color="#1a7a3a",ls="--",lw=1.2); ax.text(len(fams)-0.5,floor+0.012,f"transfer floor {floor:.2f}",fontsize=7.5,color="#1a7a3a",ha="right")
for i in range(len(fams)):
    d=lofo[i]-indist[i]
    ax.text(x[i]+w/2,lofo[i]+0.01,f"{d:+.2f}",fontsize=6.3,ha="center",color="#c0392b" if d<-0.05 else "#555")
ax.set_xticks(x); ax.set_xticklabels(fams,rotation=30,ha="right",fontsize=8)
ax.set_ylabel("oracle per-column $r$"); ax.set_ylim(0,0.8)
ax.set_title(f"Leave-family-out exposes a sequence-only transfer floor (macro avg {floor:.3f}, n={S['n_families']} families)",
             fontsize=8.6,fontweight="bold",loc="left")
ax.legend(fontsize=7.5,frameon=False,loc="upper right")
for s in ["top","right"]: ax.spines[s].set_visible(False)
fig.tight_layout()
for e in ["pdf","png","svg"]: fig.savefig(f"{OUT}/figure1_lofo.{e}",dpi=300,bbox_inches="tight")
print(f"saved {OUT}/figure1_lofo.* | families={fams} floor={floor:.3f}")
