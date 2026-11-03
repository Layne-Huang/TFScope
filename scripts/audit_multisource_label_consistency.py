#!/usr/bin/env python
"""Do the several motif databases disagree about the same TF, or are they replicates?

Half the genes in the v23 table (673 / 1335) carry motifs from two or more databases.
That is the direct cause of the label leakage found in the cluster40 benchmark: 84% of
its test rows have a near-identical PWM sitting in v24's training set under a different
filename and a different source.

Before deciding whether the model should be CONDITIONED on the source database (a
proposed `platform id` input), the prior question has to be answered: how much do the
databases actually disagree?

  cross-source pairs   two PWMs of the same gene from DIFFERENT databases
  within-source pairs  two PWMs of the same gene from the SAME database
                       -> the replicate-noise floor. Cross-source agreement can only
                          be judged against it; "r = 0.9" means nothing until you know
                          what two rows of the same database score.

If cross-source ~ within-source, the extra rows are replicate measurements: deduplicate
or average them and no source input is needed. If cross-source is clearly worse, the
databases encode different assays and the source matters -- though conditioning on it
still leaves the question of what to feed at inference for a novel protein.

Similarity is the same oracle shift + reverse-complement registration used everywhere
else, scored over the IC-trimmed cores.

  python scripts/audit_multisource_label_consistency.py
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "src")
from iclr.baselines import _decode_pwm                        # noqa: E402
from iclr.unified_eval import _aligned_cols, _overlap_r, trimmed_core  # noqa: E402

DATA = "data/processed/tf_pwm_training_v23.parquet"
OUT = "results/data_audit/multisource_label_consistency.json"


def base_db(s: str) -> str:
    """CISBP_v3.10 -> CISBP, HOCOMOCO_v14 -> HOCOMOCO. Version bumps of one database
    are the same database; keeping them apart would count a re-release as a second
    independent source."""
    return re.split(r"[_.]v?\d", str(s))[0].upper()


def sim(a, b):
    ca, cb = trimmed_core(a), trimmed_core(b)
    if ca is None or cb is None:
        return None
    al, cols = _aligned_cols(ca, cb)
    return _overlap_r(cb, al, cols) if len(cols) >= 2 else None


def main():
    df = pd.read_parquet(DATA)
    df["g"] = df.gene_symbol.astype(str).str.upper()
    df["db"] = df.motif_source.map(base_db)
    df["arr"] = df.pwm.map(_decode_pwm)
    print(f"{len(df)} rows / {df.g.nunique()} genes / databases: "
          f"{sorted(df.db.unique())}")

    cross, within = [], []
    for g, s in df.groupby("g"):
        if len(s) < 2:
            continue
        recs = list(s[["db", "arr"]].itertuples(index=False))
        for (d1, p1), (d2, p2) in itertools.combinations(recs, 2):
            v = sim(p1, p2)
            if v is None:
                continue
            (cross if d1 != d2 else within).append(
                {"gene": g, "db1": min(d1, d2), "db2": max(d1, d2), "r": v})
    cr = pd.DataFrame(cross); wi = pd.DataFrame(within)

    def desc(d, name):
        if d.empty:
            return {"n_pairs": 0}
        q = d.r.quantile([.05, .25, .5, .75, .95]).round(3).to_dict()
        out = {"n_pairs": len(d), "n_genes": int(d.gene.nunique()),
               "mean": round(float(d.r.mean()), 3), "median": round(float(d.r.median()), 3),
               "q05": q[.05], "q25": q[.25], "q75": q[.75], "q95": q[.95],
               "frac_ge_0.99": round(float((d.r >= .99).mean()), 3),
               "frac_ge_0.95": round(float((d.r >= .95).mean()), 3),
               "frac_ge_0.90": round(float((d.r >= .90).mean()), 3),
               "frac_lt_0.70": round(float((d.r < .70).mean()), 3)}
        print(f"\n=== {name} ===")
        for k, v in out.items():
            print(f"  {k:<14} {v}")
        return out

    res = {"data": DATA,
           "n_rows": int(len(df)), "n_genes": int(df.g.nunique()),
           "cross_source": desc(cr, "CROSS-source pairs (different databases)"),
           "within_source": desc(wi, "WITHIN-source pairs (same database, replicates)")}

    if not cr.empty:
        pair = (cr.groupby(["db1", "db2"]).r
                .agg(n="size", median="median", frac_ge_095=lambda x: (x >= .95).mean())
                .round(3).sort_values("n", ascending=False))
        print("\n=== by database pair ===")
        print(pair.to_string())
        res["by_database_pair"] = json.loads(
            pair.reset_index().to_json(orient="records"))

        # a gene is "consistent" if every cross-source pair it has agrees
        byg = cr.groupby("gene").r.min()
        res["genes_with_cross_source"] = int(len(byg))
        res["genes_all_pairs_ge_0.90"] = int((byg >= .90).sum())
        res["genes_any_pair_lt_0.70"] = int((byg < .70).sum())
        print(f"\ngenes with >=1 cross-source pair : {len(byg)}")
        print(f"  all their pairs agree at r>=0.90: {(byg >= .90).sum()} "
              f"({(byg >= .90).mean()*100:.0f}%)")
        print(f"  at least one pair below r<0.70  : {(byg < .70).sum()} "
              f"({(byg < .70).mean()*100:.0f}%)")
        worst = byg.sort_values().head(12)
        res["least_consistent_genes"] = {k: round(float(v), 3) for k, v in worst.items()}
        print(f"  least consistent: "
              f"{', '.join(f'{k} {v:.2f}' for k, v in worst.items())}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    pd.concat([cr.assign(kind="cross"), wi.assign(kind="within")]).round(4).to_csv(
        OUT.replace(".json", "_pairs.csv"), index=False)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
