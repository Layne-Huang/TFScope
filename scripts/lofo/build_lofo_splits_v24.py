#!/usr/bin/env python
"""Build leave-one-family-out (LOFO) splits for the v24 config, with a paired
family-SEEN control test set carved from the *retained* families.

Why this design
---------------
The canonical `train_v22` split cannot support a per-family evaluation: under a
corrected DBD taxonomy 24 of 43 families have ZERO test rows, including the two
largest (C2H2: 1811 train rows / 0 test; Homeodomain: 1191 / 0). So "how good is
v24 on family F" is unanswerable on held-out data for most F.

Fix: one global gene partition {ctrl, val, train}, drawn ONCE and reused by every
LOFO run. For held-out family F the run is

    test      = every row of family F            (family UNSEEN)
    ctrl_test = rows of the global ctrl genes, families != F   (family SEEN,
                                                                gene unseen)
    val       = rows of the global val genes,  families != F
    train     = everything else,                families != F

Because the ctrl gene set is global, family F's own ctrl genes appear in the
ctrl_test of all the OTHER runs. That yields, for the SAME genes, a matched pair

    seen_F   = mean over runs G != F of score(model_G on F's ctrl genes)
    unseen_F = score(model_F on F's ctrl genes)
    delta_F  = seen_F - unseen_F        <- the cost of never seeing family F

with no extra training runs and no leakage on either side.

Taxonomy: `scripts/reclassify_tf_families.py` (curated Lambert-2018 rules), with
C2H2_short/medium/long merged into a single C2H2 family.

Data universe: the canonical train u val u test rows (5743), i.e. exactly what
v24 trained/validated/tested on. The 269 `excluded` rows are never placed in any
run's train/val/ctrl; they are appended to their OWN family's LOFO test set only
(they are clean by construction -- gene-disjoint from train and val).

Out: data/processed/splits/lofo_v24/<Fam>.json        (test = family F)
     data/processed/splits/lofo_v24/<Fam>__ctrl.json  (test = ctrl genes)
     data/processed/splits/lofo_v24/_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from reclassify_tf_families import classify  # noqa: E402

DATA = "data/processed/tf_pwm_training_v23.parquet"
CANON_SPLIT = "data/processed/splits/train_v22/split.json"
OUTDIR = "data/processed/splits/lofo_v24"
SEED = 42

# 12 families to hold out (chosen to span 1927 -> 49 rows so the LOFO drop can be
# regressed against family size). Filesystem-safe tags replace "/" with "-".
LOFO_FAMILIES = [
    "C2H2", "Homeodomain", "bHLH", "bZIP", "Nuclear_Receptor", "HMG/SOX",
    "IRF", "T-box", "RHD/NFkB", "PAX", "E2F/DP", "GATA",
]

# per-family gene holdout sizes
CTRL_FRAC, CTRL_MIN, CTRL_MAX = 0.25, 3, 40
VAL_FRAC,  VAL_MIN,  VAL_MAX = 0.12, 1, 25


def tag(fam: str) -> str:
    return fam.replace("/", "-")


def add_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Attach `fam_lofo`: curated DBD family with C2H2_* merged."""
    df = df.copy()
    df["g"] = df.gene_symbol.fillna("").astype(str).str.upper()
    c2h2_seed = {
        r.g: r.family_name
        for r in df[df.family_name.isin(["C2H2_short", "C2H2_medium", "C2H2_long"])].itertuples()
    }
    fam = df.g.map(lambda x: classify(x, c2h2_seed))
    df["fam_lofo"] = fam.map(lambda f: "C2H2" if str(f).startswith("C2H2") else f)
    return df


def _n_hold(n_genes: int, frac: float, lo: int, hi: int) -> int:
    """How many genes of a family to hold out; never take the whole family."""
    if n_genes <= 1:
        return 0
    k = int(min(max(lo, round(frac * n_genes)), hi))
    return int(min(k, n_genes - 1))


def partition_genes(pool: pd.DataFrame, seed: int = SEED):
    """Global, family-stratified gene partition -> (ctrl_genes, val_genes)."""
    rng = np.random.RandomState(seed)
    ctrl, val = set(), set()
    for fam, sub in pool.groupby("fam_lofo"):
        genes = sorted(sub.g.unique())
        rng.shuffle(genes)
        n_c = _n_hold(len(genes), CTRL_FRAC, CTRL_MIN, CTRL_MAX)
        n_v = _n_hold(len(genes) - n_c, VAL_FRAC, VAL_MIN, VAL_MAX)
        ctrl.update(genes[:n_c])
        val.update(genes[n_c:n_c + n_v])
    return ctrl, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--families", nargs="*", default=LOFO_FAMILIES)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = add_taxonomy(pd.read_parquet(args.data))
    df["filename"] = df.filename.astype(str)
    canon = json.load(open(CANON_SPLIT))

    inuse_fns = set(canon["train"]) | set(canon["val"]) | set(canon["test"])
    excl_fns = set(canon["excluded"])
    pool = df[df.filename.isin(inuse_fns)].reset_index(drop=True)   # training universe
    excl = df[df.filename.isin(excl_fns)].reset_index(drop=True)    # clean bonus test rows

    # `excluded` must be gene-disjoint from the training universe, else appending it
    # to a LOFO test set would import genes the other runs trained on.
    bad = set(excl.g) & set(pool.g)
    assert not bad, f"excluded genes leak into the training universe: {sorted(bad)[:10]}"

    ctrl_genes, val_genes = partition_genes(pool, args.seed)
    train_genes = set(pool.g) - ctrl_genes - val_genes
    print(f"universe: {len(pool)} rows / {pool.g.nunique()} genes / "
          f"{pool.fam_lofo.nunique()} families  (+{len(excl)} clean excluded rows)")
    print(f"global gene partition: train={len(train_genes)} ctrl={len(ctrl_genes)} "
          f"val={len(val_genes)}")

    manifest = {
        "note": __doc__.strip().splitlines()[0],
        "data": args.data,
        "canonical_split": CANON_SPLIT,
        "seed": args.seed,
        "taxonomy": "scripts/reclassify_tf_families.py, C2H2_* merged -> C2H2",
        "universe_rows": int(len(pool)),
        "universe_genes": int(pool.g.nunique()),
        "global_ctrl_genes": sorted(ctrl_genes),
        "global_val_genes": sorted(val_genes),
        "families": {},
    }

    rows_out = []
    for fam in args.families:
        f_pool = pool[pool.fam_lofo == fam]
        if f_pool.empty:
            print(f"[skip] {fam}: no rows"); continue
        f_excl = excl[excl.fam_lofo == fam]

        keep = pool[pool.fam_lofo != fam]
        tr = keep[keep.g.isin(train_genes)]
        va = keep[keep.g.isin(val_genes)]
        ct = keep[keep.g.isin(ctrl_genes)]
        te = pd.concat([f_pool, f_excl], ignore_index=True)

        tr_f, va_f = sorted(tr.filename), sorted(va.filename)
        ct_f, te_f = sorted(ct.filename), sorted(te.filename)
        for a, b in [(tr_f, va_f), (tr_f, te_f), (tr_f, ct_f),
                     (va_f, te_f), (va_f, ct_f), (ct_f, te_f)]:
            assert not (set(a) & set(b)), f"{fam}: filename overlap between splits"
        for a, b in [(tr, va), (tr, te), (tr, ct), (va, te), (va, ct), (ct, te)]:
            assert not (set(a.g) & set(b.g)), f"{fam}: GENE overlap between splits"
        assert fam not in set(tr.fam_lofo) | set(va.fam_lofo) | set(ct.fam_lofo), \
            f"{fam}: held-out family leaked into train/val/ctrl"

        meta = {
            "held_out_family": fam,
            "method": "leave_one_family_out + global paired ctrl gene holdout",
            "n_train_rows": len(tr_f), "n_train_genes": int(tr.g.nunique()),
            "n_val_rows": len(va_f), "n_val_genes": int(va.g.nunique()),
            "n_ctrl_rows": len(ct_f), "n_ctrl_genes": int(ct.g.nunique()),
            "n_test_rows": len(te_f), "n_test_genes": int(te.g.nunique()),
            "n_test_rows_from_excluded": int(len(f_excl)),
            "ctrl_families": int(ct.fam_lofo.nunique()),
            # this family's own ctrl genes -> the paired comparison set
            "own_ctrl_genes": sorted(set(f_pool.g) & ctrl_genes),
            "own_ctrl_rows": sorted(f_pool[f_pool.g.isin(ctrl_genes)].filename),
            "seed": args.seed,
        }
        t = tag(fam)
        json.dump({"train": tr_f, "val": va_f, "test": te_f, "ctrl": ct_f,
                   "metadata": meta},
                  open(f"{args.outdir}/{t}.json", "w"))
        # TFDataset only understands train/val/test, so the ctrl set gets its own file
        json.dump({"train": tr_f, "val": va_f, "test": ct_f,
                   "metadata": {**meta, "test_set_is": "ctrl (family-SEEN control)"}},
                  open(f"{args.outdir}/{t}__ctrl.json", "w"))
        manifest["families"][fam] = meta
        rows_out.append((fam, len(tr_f), len(va_f), len(ct_f), len(te_f),
                         int(te.g.nunique()), len(meta["own_ctrl_genes"])))
        print(f"[{fam:<18}] train={len(tr_f):>4} val={len(va_f):>4} ctrl={len(ct_f):>4} "
              f"test={len(te_f):>4} ({te.g.nunique()} genes)  own_ctrl_genes="
              f"{len(meta['own_ctrl_genes'])}")

    json.dump(manifest, open(f"{args.outdir}/_manifest.json", "w"), indent=1)
    print("\n" + "=" * 88)
    print(f"{'Family':<18} {'train':>6} {'val':>5} {'ctrl':>5} {'test':>6} "
          f"{'test_genes':>11} {'own_ctrl_genes':>15}")
    print("-" * 88)
    for r in rows_out:
        print(f"{r[0]:<18} {r[1]:>6} {r[2]:>5} {r[3]:>5} {r[4]:>6} {r[5]:>11} {r[6]:>15}")
    print("=" * 88)
    print(f"saved {len(rows_out)} LOFO splits -> {args.outdir}/")


if __name__ == "__main__":
    main()
