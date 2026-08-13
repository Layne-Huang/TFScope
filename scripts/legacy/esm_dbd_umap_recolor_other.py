#!/usr/bin/env python
"""Re-annotate the 'Other' catch-all into representative DBD sub-families and
re-plot the frozen-ESM-2 DBD UMAP, reusing the coordinates already computed by
esm_dbd_family_diagnostic.py (identical layout, recolored only).

The nine curated families keep their muted colors and circle markers; the
sub-families rescued from 'Other' are drawn as triangles in a distinct vibrant
palette; residual 'Other (misc)' is faint gray.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/esm_dbd_family_diagnostic")
SEQ_PARQUET = "data/processed/tf_sequences.parquet"

# Priority-ordered domain-keyword -> representative sub-family label.
RULES = [
    ("T-box",                       ["t-box", "brachyury"]),
    ("HMG-box / SOX",               ["high mobility group", "hmg-box", "sry-related hmg", "hmg box"]),
    ("p53 / Rel / STAT",            ["p53-like", "rel homology", "stat", "runt", "ndt80"]),
    ("Winged-helix (E2F/RFX)",      ["winged helix", "fork head", "forkhead-like"]),
    ("PHD / RING zinc finger",      ["phd-type", "phd-finger", "phd finger", "ring/fyve/phd", "fyve/phd"]),
    ("Myb / SANT",                  ["sant/myb", "myb domain", "myb-like", "sant domain"]),
]
MIN_N = 10  # sub-families with >= MIN_N members are shown; rest -> Other (misc)

# 9 curated families: muted (circles).
CURATED_COLORS = {
    "C2H2_long": "#332288", "Homeodomain": "#CC6677", "C2H2_medium": "#117733",
    "bHLH": "#DDCC77", "C2H2_short": "#88CCEE", "bZIP": "#AA4499",
    "Nuclear_Receptor": "#44AA99", "Forkhead": "#882255", "ETS": "#999933",
}
# Only the TOP 3 sub-families from 'Other' are shown, as circles alongside the
# curated families. Everything else in 'Other' collapses to gray "Other".
SUB_COLORS = {
    "Winged-helix (E2F/RFX)": "#EE7733",
    "HMG-box / SOX": "#DD3388",
    "p53 / Rel / STAT": "#000000",
}
MISC = "Other"


def classify(dj):
    try:
        d = json.loads(dj) if isinstance(dj, str) else dj
    except Exception:
        return MISC
    descs = []
    if isinstance(d, list):
        for dom in d:
            if isinstance(dom, dict):
                for k in ("type", "name", "description", "pfam"):
                    if dom.get(k):
                        descs.append(str(dom[k]).lower())
    blob = " | ".join(descs)
    for name, kws in RULES:
        if any(kw in blob for kw in kws):
            return name
    return MISC


def main():
    seq = pd.read_parquet(SEQ_PARQUET)[["uniprot_id", "family_name", "domains_json"]]
    seq = seq.rename(columns={"uniprot_id": "tf_id"})
    seq["fine"] = seq["family_name"]
    om = seq["family_name"] == "Other"
    seq.loc[om, "fine"] = seq.loc[om, "domains_json"].apply(classify)
    # Collapse rare sub-families into misc.
    keep = set(SUB_COLORS)
    seq.loc[om & ~seq["fine"].isin(keep), "fine"] = MISC

    umap = pd.read_csv(OUT / "esm_dbd_umap.csv")
    df = umap.merge(seq[["tf_id", "fine"]], on="tf_id", how="left")
    df.to_csv(OUT / "esm_dbd_umap_fine.csv", index=False)

    # --- kNN purity of rescued sub-families in embedding space ---
    from sklearn.neighbors import NearestNeighbors
    emb = np.load(OUT / "dbd_embeddings.npy")
    inp = pd.read_csv(OUT / "esm_dbd_input.csv")  # same row order as embeddings
    fine_by_id = dict(zip(df["tf_id"], df["fine"]))
    fine = np.array([fine_by_id[t] for t in inp["tf_id"]])
    nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(emb)
    _, idx = nn.kneighbors(emb)
    same = (fine[idx[:, 1:]] == fine[:, None]).mean(axis=1)
    sub_purity = {f: {"n": int((fine == f).sum()),
                      "knn_purity": round(float(same[fine == f].mean()), 3)}
                  for f in SUB_COLORS if (fine == f).any()}
    with open(OUT / "other_subfamily_purity.json", "w") as fh:
        json.dump(sub_purity, fh, indent=2)

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    coords = df[["umap1", "umap2"]].to_numpy()
    fam = df["fine"].to_numpy()

    def scatter(label, color, marker, size, alpha, zorder):
        m = fam == label
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=size, marker=marker,
                       linewidths=0, c=color, alpha=alpha, zorder=zorder,
                       rasterized=True)

    scatter(MISC, "#D0D0D0", "o", 5, 0.5, 1)                       # gray Other
    for f, c in CURATED_COLORS.items():                            # curated
        scatter(f, c, "o", 6, 0.85, 2)
    for f, c in SUB_COLORS.items():                                # top-3 from Other
        scatter(f, c, "o", 6, 0.85, 3)

    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("Frozen ESM-2 DBD embeddings", fontsize=9, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(False); ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    def counted(label):
        n = int((fam == label).sum())
        return f"{label} (n={n})"

    legend_items = list(CURATED_COLORS.items()) + list(SUB_COLORS.items())
    handles = [Line2D([], [], marker="o", ls="", mfc=c, mec="none", ms=5,
                      label=counted(f)) for f, c in legend_items]
    handles.append(Line2D([], [], marker="o", ls="", mfc="#D0D0D0", mec="none",
                          ms=5, label=counted(MISC)))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=6.5, handletextpad=0.3, labelspacing=0.34,
              borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(OUT / "esm_dbd_umap_fine.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "esm_dbd_umap_fine.pdf", bbox_inches="tight")
    plt.close(fig)

    print("saved esm_dbd_umap_fine.{png,pdf,csv} + other_subfamily_purity.json")
    print("\nRescued sub-family kNN(k=10) purity:")
    for f, v in sorted(sub_purity.items(), key=lambda x: -x[1]["knn_purity"]):
        print(f"  {f:26s} n={v['n']:3d}  purity={v['knn_purity']:.3f}")


if __name__ == "__main__":
    main()
