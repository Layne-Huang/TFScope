#!/usr/bin/env python
"""CS-MoE figure: family-conditioned experts specialize by structural class.

Reads results/moe_routing/routing.npz (expert x family counts), draws a
column-normalized routing heatmap (each family's top-2 routing distribution over
the 12 experts), and annotates the dominant per-family expert. Highlights that
the C2H2_long frontier family commands a dedicated expert cluster.

Out: figures/cs_moe_routing.pdf / .png
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ = "results/moe_routing/routing.npz"
SUM = "results/moe_routing/summary.json"
OUT = "figures/cs_moe_routing"

FAM = {0: "C2H2_short", 1: "C2H2_medium", 2: "C2H2_long", 3: "bHLH",
       4: "Homeodomain", 5: "bZIP", 6: "Nuclear_Receptor", 7: "Forkhead",
       8: "ETS", 9: "Other"}

d = np.load(NPZ)
M = d["expert_family_counts"].astype(float)        # (E, Fpresent)
fams_present = list(d["fams_present"])
E = M.shape[0]
summary = json.load(open(SUM))

# column-normalize: each family -> routing distribution over experts
Mn = M / np.clip(M.sum(0, keepdims=True), 1, None)

# order families to group C2H2 subfamilies together for visual block structure
order = ["bZIP", "Homeodomain", "Nuclear_Receptor", "Forkhead", "bHLH", "ETS",
         "Other", "C2H2_short", "C2H2_medium", "C2H2_long"]
name2col = {FAM[f]: i for i, f in enumerate(fams_present)}
cols = [name2col[n] for n in order if n in name2col]
labels = [n for n in order if n in name2col]
Mo = Mn[:, cols]

fig, ax = plt.subplots(figsize=(7.4, 6.0))
im = ax.imshow(Mo, aspect="auto", cmap="magma", vmin=0, vmax=Mo.max())

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(E))
ax.set_yticklabels([f"E{e}" for e in range(E)], fontsize=8)
ax.set_xlabel("TF family", fontsize=10)
ax.set_ylabel("MoE expert", fontsize=10)
ax.set_title("Family-conditioned experts specialize by structural class\n"
             f"(top-2 routing, {summary['n_tfs']} held-out cluster40 TFs; "
             f"mean per-family entropy {summary['mean_per_family_routing_entropy_bits']:.2f} / "
             f"{summary['max_entropy_bits']:.2f} bits)",
             fontsize=10.5, fontweight="bold")

# mark the dominant expert for each family with a cyan box
for j, name in enumerate(labels):
    e = int(np.argmax(Mo[:, j]))
    ax.add_patch(plt.Rectangle((j - 0.5, e - 0.5), 1, 1, fill=False,
                               edgecolor="#56B4E9", lw=2.0))

# annotate the C2H2_long dedicated cluster
if "C2H2_long" in labels:
    jx = labels.index("C2H2_long")
    ax.annotate("dedicated\nC2H2-long\nexperts", xy=(jx, 10), xytext=(jx + 0.1, 5.5),
                fontsize=8, color="#56B4E9", ha="center", fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("fraction of family's routes to expert", fontsize=9)

fig.tight_layout()
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
print("saved", OUT + ".pdf")

# quick text recap of the cleanest specialists (row-normalized)
Mr = M / np.clip(M.sum(1, keepdims=True), 1, None)
print("\nCleanest expert specialists (row-normalized, family share of expert):")
for e in range(E):
    fi = int(np.argmax(Mr[e]))
    sh = Mr[e, fi]
    if sh > 0.55:
        print(f"  E{e}: {FAM[fams_present[fi]]:<16} {sh*100:.1f}%")
