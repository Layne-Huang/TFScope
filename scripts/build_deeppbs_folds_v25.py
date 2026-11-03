#!/usr/bin/env python
"""Project the frozen unified-v25 split onto DeepPBS's structure files.

Both models then see the SAME partition of the SAME structures, so the 84-sample
comparison is symmetric: neither has trained on anything the other is tested on.
Previous DeepPBS numbers could not claim that -- the released checkpoints saw all
20 benchmark genes, and the earlier `pdbdisjoint` retrain was cut on a different
criterion from TFScope's.

  universe   the 650 rows of the DeepPBS table that survive into unified_v25,
             each mapping 1:1 onto an assembly2024 npz
  test       whatever unified_v25 calls test (149 npz, containing all 84 benchmark)
  train/val  5-fold CV over the remaining 501, folded by CD-HIT COMPONENT rather
             than by row -- a random row-level split would put homologues on both
             sides and make DeepPBS's best-state selection optimistic

npz naming has two conventions that the parquet filename does not share:
HOCOMOCO `1a02_F_FOS_HUMAN.H11MO.0.A.npz` vs JASPAR `1a0a_A_MA0357.1.jaspar.npz`,
against parquet `1a02_F_FOS.FOS_HUMAN.H11MO.0.A.txt`. Matching is by
`<pdb>_<chain>` prefix plus motif-id suffix, and every one of the 650 resolves.

  python scripts/build_deeppbs_folds_v25.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

UNI = "data/processed/tf_pwm_unified_v25.parquet"
DATA_DIR = "/data1/leihuang/DeepPBS/deeppbsmar24/data/assembly2024"
OUTDIR = "/data1/leihuang/DeepPBS/deeppbsmar24/run/v25_folds"
N_FOLDS = 5
SEED = 42


def npz_index(data_dir: str) -> dict[str, list[str]]:
    by = defaultdict(list)
    for n in sorted(os.listdir(data_dir)):
        if n.endswith(".npz"):
            by["_".join(n[:-4].split("_")[:2])].append(n)
    return by


def match(fn: str, by: dict[str, list[str]]) -> str | None:
    s = fn[:-4] if fn.endswith(".txt") else fn
    p = s.split("_")
    rest = "_".join(p[2:])                      # GENE.motif...
    motif = rest.split(".", 1)[1] if "." in rest else rest
    cands = by.get("_".join(p[:2]), [])
    for n in cands:
        if n[:-4].endswith(motif) or n[:-4].endswith(motif + ".jaspar"):
            return n
    for n in cands:
        if motif in n:
            return n
    return None


def main():
    rng = np.random.RandomState(SEED)
    u = pd.read_parquet(UNI)
    d = u[u.source_table == "deeppbs"].copy()
    by = npz_index(DATA_DIR)
    d["npz"] = d.filename.map(lambda f: match(f, by))
    assert d.npz.notna().all(), f"{int(d.npz.isna().sum())} rows have no npz"
    assert d.npz.is_unique, "two rows collapsed onto one npz"
    print(f"universe {len(d)} npz / {d.gene_symbol.nunique()} genes")

    test = d[d.split == "test"]
    bench = test[test.is_deeppbs_bench]
    pool = d[d.split != "test"].copy()
    print(f"  test {len(test)} npz (benchmark {len(bench)}) | CV pool {len(pool)}")

    # fold whole components, largest first, into the currently-smallest fold
    sizes = pool.groupby("comp").size().sort_values(ascending=False)
    order = list(sizes.index)
    rng.shuffle(order)
    order = sorted(order, key=lambda c: -sizes[c])
    fold_of, load = {}, [0] * N_FOLDS
    for c in order:
        k = int(np.argmin(load))
        fold_of[c] = k
        load[k] += sizes[c]
    pool["fold"] = pool.comp.map(fold_of)

    os.makedirs(OUTDIR, exist_ok=True)
    write = lambda p, names: open(p, "w").write("".join(f"{n}\n" for n in sorted(names)))
    write(f"{OUTDIR}/test149.txt", test.npz)
    write(f"{OUTDIR}/test84_benchmark.txt", bench.npz)

    print(f"\n{'fold':<6}{'train':>7}{'valid':>7}   valid genes / components")
    for k in range(N_FOLDS):
        va = pool[pool.fold == k]
        tr = pool[pool.fold != k]
        write(f"{OUTDIR}/train{k}.txt", tr.npz)
        write(f"{OUTDIR}/valid{k}.txt", va.npz)
        assert not (set(tr.comp) & set(va.comp)), f"fold {k} shares a component"
        assert not (set(tr.npz) & set(test.npz)), f"fold {k} train touches test"
        print(f"{k:<6}{len(tr):>7}{len(va):>7}   {va.gene_symbol.nunique()} / {va.comp.nunique()}")

    cfg = json.load(open("/data1/leihuang/DeepPBS/deeppbsmar24/run/config.json"))
    cfg["data_dir"] = DATA_DIR
    json.dump(cfg, open(f"{OUTDIR}/config.json", "w"), indent=2)

    meta = {"source_split": "data/processed/splits/unified_v25/split.json",
            "n_universe": len(d), "n_test": len(test), "n_benchmark": len(bench),
            "n_pool": len(pool), "n_folds": N_FOLDS, "fold_unit": "cdhit_component",
            "seed": SEED}
    json.dump(meta, open(f"{OUTDIR}/manifest.json", "w"), indent=1)
    print(f"\nsaved {OUTDIR}/  (train/valid 0-{N_FOLDS-1}, test149, test84_benchmark, config, manifest)")


if __name__ == "__main__":
    main()
