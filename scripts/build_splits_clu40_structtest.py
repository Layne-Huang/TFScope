#!/usr/bin/env python
"""40%-identity splits with the TEST set drawn from OUR structure set.

Why not DeepPBS's own blind list: it carries known leakage (see
[[benchmark-tf-leakage]]), and we want test cases we can also run DeepPBS on
ourselves for a like-for-like comparison -- which requires a structure.

Design
  * cluster BOTH datasets jointly at 40% identity (MMseqs2)
  * TEST      = structure rows from selected clusters
  * EXCLUDED  = every other row in those clusters (homologous to test)
  * VAL       = further whole clusters, disjoint from train/test
  * TRAIN     = the rest
  * gene-level integrity: all rows of a gene land in ONE split (a gene can
    span several clusters because its rows have different crop lengths --
    without this, 46 genes leaked across splits)

Test-cluster selection maximises structure rows per unit of collateral
exclusion (some clusters are ~100% structure and cost nothing; the C2H2
mega-cluster would cost 1154 excluded rows for 56 test rows) while covering
multiple DBD families.
"""
import json, os, subprocess
import numpy as np
import pandas as pd

TEST_TARGET = 200
VAL_FRAC = 0.12
SEED = 42
OUT = "data/processed/splits/clu40_structtest"


def main():
    # Atomic unit = connected component of the gene<->cluster bipartite graph.
    # Splitting on clusters alone leaked 46 genes across splits (a gene's rows
    # have different crop lengths and land in different clusters); patching that
    # afterwards by gene then broke cluster-disjointness (3 shared clusters).
    # Components satisfy BOTH constraints simultaneously.
    allr = pd.read_parquet("/tmp/clu40/allrows_comp.parquet")
    print(f"rows={len(allr)}  clusters={allr['_c'].nunique()}  components={allr['_comp'].nunique()}")

    stats = (allr.groupby("_comp")
             .agg(n=("_rid", "size"), n_str=("_set", lambda s: (s == "struct").sum()),
                  fam=("_fam", lambda s: s.mode().iat[0]))
             .query("n_str > 0"))
    stats["collateral"] = stats["n"] - stats["n_str"]
    stats["eff"] = stats["n_str"] / stats["n"]

    # --- pick test clusters: family-diverse, cheapest-first ---
    rng = np.random.default_rng(SEED)
    chosen, n_test = [], 0
    for fam, grp in stats.sort_values("eff", ascending=False).groupby("fam"):
        best = grp.sort_values(["eff", "n_str"], ascending=[False, False]).index[0]
        if stats.loc[best, "n_str"] >= 3:            # skip trivially tiny families
            chosen.append(best); n_test += int(stats.loc[best, "n_str"])
    for c in stats.sort_values(["eff", "n_str"], ascending=[False, False]).index:
        if n_test >= TEST_TARGET: break
        if c in chosen: continue
        if stats.loc[c, "collateral"] > 3 * stats.loc[c, "n_str"]:
            continue                                  # too expensive
        chosen.append(c); n_test += int(stats.loc[c, "n_str"])
    test_clusters = set(chosen)
    print(f"\nselected {len(test_clusters)} test clusters -> {n_test} structure rows")
    print(stats.loc[sorted(test_clusters)][["n", "n_str", "collateral", "eff", "fam"]]
          .sort_values("n_str", ascending=False).to_string())

    allr["split"] = "train"
    inc = allr["_comp"].isin(test_clusters)
    allr.loc[inc & (allr["_set"] == "struct"), "split"] = "test"
    allr.loc[inc & (allr["_set"] != "struct"), "split"] = "excluded"

    rest = allr[allr["split"] == "train"]
    cl = rest.groupby("_comp").size()
    order = rng.permutation(cl.index.to_numpy())
    target, acc, val_clu = int(len(rest) * VAL_FRAC), 0, []
    for c in order:
        if acc >= target: break
        val_clu.append(c); acc += int(cl[c])
    allr.loc[allr["_comp"].isin(val_clu), "split"] = "val"

    # components already guarantee gene- AND cluster-disjointness; no patch needed
    allr.loc[(allr["split"] == "test") & (allr["_set"] != "struct"), "split"] = "excluded"

    print("\n=== final splits ===")
    print(allr.groupby(["split", "_set"]).size().unstack(fill_value=0).to_string())
    for s in ["train", "val", "test", "excluded"]:
        sub = allr[allr["split"] == s]
        print(f"  {s:<9} rows={len(sub):>5} genes={sub['_g'].nunique():>4} clusters={sub["_comp"].nunique():>4}")
    print("\ntest families:", allr[allr.split == "test"]["_fam"].value_counts().to_dict())

    print("\n=== leakage audit (must be 0) ===")
    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
        ca, cb = set(allr[allr.split == a]["_c"]), set(allr[allr.split == b]["_c"])
        pa, pb = set(allr[allr.split == a]["_comp"]), set(allr[allr.split == b]["_comp"])
        ga, gb = set(allr[allr.split == a]["_g"]), set(allr[allr.split == b]["_g"])
        print(f"  {a}/{b}: shared components={len(pa & pb)}  shared clusters={len(ca & cb)}  shared genes={len(ga & gb)}")

    os.makedirs(OUT, exist_ok=True)
    allr[["_rid", "_set", "_g", "_c", "_comp", "_fam", "split"]].to_parquet(f"{OUT}/assignments.parquet")
    json.dump({s: {t: allr[(allr.split == s) & (allr._set == t)]["_rid"].tolist()
                   for t in ["seq", "struct"]} for s in ["train", "val", "test", "excluded"]},
              open(f"{OUT}/split.json", "w"))
    print(f"\nsaved {OUT}/")


if __name__ == "__main__":
    main()
