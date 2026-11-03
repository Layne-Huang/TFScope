#!/usr/bin/env python
"""One split to replace them all: sequence set + structure set, 40% identity, with the
DeepPBS 84-sample benchmark inside the test set.

Replaces the two mutually incompatible splits the project accumulated. `train_v22` cut
the v23 table by family (24 of 43 families ended up with zero test rows) and
`deeppbs_cluster40` cut a SEPARATE table by homology for a different model. Their test
sets share two genes. Scoring v24 on the DeepPBS one therefore leaked badly: 84% of its
rows had a near-identical PWM in v24's training data under another filename.

Construction
------------
1. UNIVERSE = the repaired v25 table (sequence + structure rows) UNION the DeepPBS
   table. The DeepPBS rows are kept as their own rows, not merged into look-alikes, so
   the 84 benchmark samples survive verbatim and the comparison runs on exactly them.

2. CLUSTER every unique protein at 40% identity with CD-HIT (-c 0.4 -n 2, short-word
   mode). Homologues cannot straddle the split.

3. GROUP by CD-HIT cluster only. Sequence identity is the sole isolation criterion.

   A gene-identity edge was tried and rejected: chaining clusters through genes that
   carry two differently-cropped sequences produced a 438-gene / 1493-row super-cluster,
   which a single pinned benchmark row then dragged wholesale into test (test 51%,
   train 39%). Without it the residual risk is that one gene's crops land in different
   clusters and therefore on opposite sides of the split; `--report-leak` measures
   exactly how much of that survives instead of assuming it away.

4. ASSIGN whole components. Components holding a DeepPBS benchmark row go to test first
   and unconditionally; the rest are filled to the target ratio largest-first, which
   keeps the big components from overshooting a small split.

Out: data/processed/splits/unified_v25/{split.json, components.csv, summary.json}
     data/processed/tf_pwm_unified_v25.parquet   (the merged universe the split indexes)

  python scripts/build_unified_split_v25.py [--val-frac 0.10] [--test-frac 0.15]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

# cd-hit ships inside the tfscope env; invoking python by absolute path leaves that
# env's bin off PATH, so resolve the binary rather than relying on the caller's shell
CDHIT = next((c for c in (
    os.path.join(os.path.dirname(sys.executable), "cd-hit"),
    "/data1/leihuang/miniconda3/envs/tfscope/bin/cd-hit") if os.path.exists(c)), "cd-hit")

sys.path.insert(0, "scripts")

V25 = "data/processed/tf_pwm_training_v25.parquet"
DPBS = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
DPBS_SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
OUTDIR = "data/processed/splits/unified_v25"
MERGED = "data/processed/tf_pwm_unified_v25.parquet"
SEED = 42


def cdhit(seqs: dict[str, str], identity=0.4, threads=8) -> dict[str, int]:
    """seq_id -> cluster index. -n 2 is required below 0.5 identity."""
    d = tempfile.mkdtemp(prefix="cdhit_")
    fa, out = os.path.join(d, "in.fa"), os.path.join(d, "out")
    with open(fa, "w") as f:
        for k, v in seqs.items():
            f.write(f">{k}\n{v}\n")
    cmd = [CDHIT, "-i", fa, "-o", out, "-c", str(identity), "-n", "2",
           "-M", "8000", "-T", str(threads), "-d", "0", "-g", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"cd-hit ({CDHIT}) failed:\n{r.stderr[-2000:]}")
    assign, cid = {}, -1
    for line in open(out + ".clstr"):
        line = line.strip()
        if line.startswith(">Cluster"):
            cid += 1
        elif ">" in line:
            assign[line.split(">")[1].split("...")[0].strip()] = cid
    return assign


class DSU:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--identity", type=float, default=0.40)
    ap.add_argument("--gene-edge", action="store_true",
                    help="also union clusters that share a gene symbol. Off by default: "
                         "it creates a 438-gene super-component on this data.")
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    rng = np.random.RandomState(SEED)

    # ── 1. universe ──────────────────────────────────────────────────────────
    v = pd.read_parquet(V25); v["filename"] = v.filename.astype(str)
    v["source_table"] = "v25"
    d = pd.read_parquet(DPBS); d["filename"] = d.filename.astype(str)
    d["source_table"] = "deeppbs"
    bench = set(json.load(open(DPBS_SPLIT))["test"])
    d["is_deeppbs_bench"] = d.filename.isin(bench)
    v["is_deeppbs_bench"] = False
    shared = [c for c in v.columns if c in d.columns]
    uni = pd.concat([v[shared], d[shared]], ignore_index=True)
    uni["g"] = uni.gene_symbol.astype(str).str.upper()
    print(f"universe: {len(uni)} rows "
          f"({(uni.source_table=='v25').sum()} v25 + {(uni.source_table=='deeppbs').sum()} deeppbs) "
          f"/ {uni.g.nunique()} genes / {uni.sequence.nunique()} unique sequences")
    print(f"  DeepPBS benchmark rows carried through: {int(uni.is_deeppbs_bench.sum())}")

    # ── 2. cluster ───────────────────────────────────────────────────────────
    useq = {f"s{i}": s for i, s in enumerate(sorted(uni.sequence.unique()))}
    sid = {s: k for k, s in useq.items()}
    print(f"cd-hit -c {a.identity} on {len(useq)} unique sequences ...")
    cl = cdhit(useq, a.identity)
    uni["cluster"] = uni.sequence.map(lambda s: cl.get(sid[s], -1))
    print(f"  {uni.cluster.nunique()} clusters")

    # ── 3. components: cd-hit clusters, optionally chained through shared genes ──
    if a.gene_edge:
        dsu = DSU()
        for c, s in uni.groupby("cluster").g.apply(list).items():
            for x in s:
                dsu.union(("C", c), ("G", x))
        for g, s in uni.groupby("g").cluster.apply(list).items():
            for c in s:
                dsu.union(("G", g), ("C", c))
        comp = [dsu.find(("G", g)) for g in uni.g]
    else:
        comp = list(uni.cluster)
    comp_ids = {c: i for i, c in enumerate(sorted(set(comp), key=str))}
    uni["comp"] = [comp_ids[c] for c in comp]
    sizes = uni.groupby("comp").size().sort_values(ascending=False)
    print(f"  {len(sizes)} components ({'cd-hit clusters chained through genes'
          if a.gene_edge else 'cd-hit clusters only'}); "
          f"largest {sizes.iloc[0]} rows = {sizes.iloc[0]/len(uni)*100:.1f}% of the table")

    # ── 4. assign whole components ───────────────────────────────────────────
    forced = sorted(uni.loc[uni.is_deeppbs_bench, "comp"].unique())
    n_forced = int(uni.comp.isin(forced).sum())
    print(f"components pinned to test by the benchmark: {len(forced)} "
          f"({n_forced} rows, {uni[uni.comp.isin(forced)].g.nunique()} genes)")

    target_test = int(len(uni) * a.test_frac)
    target_val = int(len(uni) * a.val_frac)
    assign = {c: "test" for c in forced}
    rest = [c for c in sizes.index if c not in assign]
    rng.shuffle(rest)
    rest = sorted(rest, key=lambda c: -sizes[c])      # largest first, ties broken randomly

    n_test = n_forced
    n_val = 0
    for c in rest:
        s = sizes[c]
        if n_test + s <= target_test:
            assign[c] = "test"; n_test += s
        elif n_val + s <= target_val:
            assign[c] = "val"; n_val += s
        else:
            assign[c] = "train"
    uni["split"] = uni.comp.map(assign)

    # ── verify ───────────────────────────────────────────────────────────────
    for key in ("comp", "cluster"):
        bad = uni.groupby(key).split.nunique()
        assert (bad == 1).all(), f"{key} straddles splits: {bad[bad>1].index[:5].tolist()}"
    assert uni.loc[uni.is_deeppbs_bench, "split"].eq("test").all(), "benchmark not in test"
    print("  checks: no cd-hit cluster / component straddles a split; "
          "all 84 benchmark rows are in test")

    # genes CAN straddle now -- report it rather than let it pass silently
    per_gene = uni.groupby("g").split.nunique()
    straddle = sorted(per_gene[per_gene > 1].index)
    summary_leak = {"genes_straddling_splits": len(straddle),
                    "genes_total": int(uni.g.nunique()),
                    "straddling_genes": straddle[:50]}
    print(f"  genes on more than one side: {len(straddle)} / {uni.g.nunique()}")

    os.makedirs(a.outdir, exist_ok=True)
    split = {k: sorted(uni.loc[uni.split == k, "filename"]) for k in ("train", "val", "test")}
    summary = {
        "note": "unified 40%-identity split over sequence+structure rows; the DeepPBS "
                "84-sample benchmark is inside test by construction",
        "universe_rows": int(len(uni)), "universe_genes": int(uni.g.nunique()),
        "identity": a.identity, "seed": SEED,
        "n_cdhit_clusters": int(uni.cluster.nunique()),
        "n_components_after_gene_closure": int(uni.comp.nunique()),
        "largest_component_rows": int(sizes.iloc[0]),
        "deeppbs_benchmark_rows": int(uni.is_deeppbs_bench.sum()),
        "gene_edge_used": bool(a.gene_edge),
        "leakage": summary_leak,
        "splits": {k: {"rows": len(vv),
                       "genes": int(uni[uni.split == k].g.nunique()),
                       "frac": round(len(vv) / len(uni), 4)}
                   for k, vv in split.items()},
    }
    split["metadata"] = summary
    json.dump(split, open(f"{a.outdir}/split.json", "w"))
    uni[["filename", "gene_symbol", "g", "source_table", "is_deeppbs_bench",
         "cluster", "comp", "split"]].to_csv(f"{a.outdir}/components.csv", index=False)
    json.dump(summary, open(f"{a.outdir}/summary.json", "w"), indent=1)
    uni.to_parquet(MERGED, index=False)

    print("\n=== split ===")
    for k, s in summary["splits"].items():
        print(f"  {k:<6} {s['rows']:5d} rows ({s['frac']*100:4.1f}%) / {s['genes']:4d} genes")
    print(f"\nsaved {a.outdir}/split.json, components.csv, summary.json")
    print(f"saved {MERGED}")


if __name__ == "__main__":
    main()
