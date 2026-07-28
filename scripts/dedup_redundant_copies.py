#!/usr/bin/env python
"""Collapse redundant crystallographic copies: rows with identical gene +
sequence + partner-gene-set are the same biological unit crystallized
multiple times (e.g. 5E8I's 4 identical FLI1-monomer-on-identical-DNA
copies) and should contribute ONE representative training row, not N
duplicates. Prefer, within each duplicate group, a row with gap_flag=False
over one with gap_flag=True (cleaner sequence), else keep the first.
"""
import pandas as pd

df = pd.read_parquet("data/processed/tf_pwm_deeppbs_v2.parquet")
print(f"before dedup: {len(df)} rows", flush=True)

def dedup_key(row):
    partner_genes = tuple(sorted(g for g in row["partner_genes"] if g))
    return (str(row["gene"]).upper(), row["sequence"], partner_genes)

df["_dedup_key"] = df.apply(dedup_key, axis=1)
groups = df.groupby("_dedup_key")

keep_idx = []
n_groups = groups.ngroups
print(f"distinct (gene, sequence, partner-gene-set) groups: {n_groups}", flush=True)

for i, (key, g) in enumerate(groups):
    if len(g) == 1:
        keep_idx.append(g.index[0])
    else:
        # prefer a row without a disorder gap; else just the first
        no_gap = g[g["gap_flag"] == False]
        chosen = no_gap.index[0] if len(no_gap) else g.index[0]
        keep_idx.append(chosen)
    if (i + 1) % 500 == 0 or (i + 1) == n_groups:
        print(f"  [{i+1}/{n_groups}] groups processed, {len(keep_idx)} rows kept so far", flush=True)

deduped = df.loc[keep_idx].drop(columns=["_dedup_key"]).reset_index(drop=True)
print(f"after dedup: {len(deduped)} rows ({len(df) - len(deduped)} redundant copies removed)", flush=True)

deduped.to_parquet("data/processed/tf_pwm_deeppbs_v2_deduped.parquet")
print("saved to data/processed/tf_pwm_deeppbs_v2_deduped.parquet", flush=True)
