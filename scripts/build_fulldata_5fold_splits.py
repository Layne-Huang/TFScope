#!/usr/bin/env python
"""Build gene-stratified 5-fold splits for the full-data canonical training set.

Pool: all JASPAR + HOCOMOCO + CIS-BP + DeepPBS entries AFTER leakage filtering:
  - Exclude entries whose gene_symbol matches any of the 130 DeepPBS test genes
  - Exclude extra (non-DeepPBS) entries with >=40% DBD identity to any test entry
  → 3,403 safe entries (3,254 train pool after reserving val)

5-fold split: gene-stratified, family-balanced (same gene always in same fold).
Test set: same 130 DeepPBS blind benchmark filenames.

Output: data/processed/splits/fulldata_5fold/fold_{0..4}.json
"""
import json, os, random
import numpy as np
import pandas as pd

PARQUET      = "data/processed/tf_pwm_aug_dbd_canon.parquet"
DPBS_PARQUET = "data/processed/tf_pwm_deeppbs_only.parquet"
TEST_SPLIT   = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
LEAKY_FNS    = "data/processed/splits/fulldata_canon/leaky_extra_fns.json"
OUT_DIR      = "data/processed/splits/fulldata_5fold"
N_FOLDS      = 5
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)

df       = pd.read_parquet(PARQUET)
df_dpbs  = pd.read_parquet(DPBS_PARQUET)
dpbs_fns = set(df_dpbs.filename)

test_split = json.load(open(TEST_SPLIT))
test_fns   = set(test_split["test"])
test_genes = set(df[df.filename.isin(test_fns)].gene_symbol)
leaky_fns  = set(json.load(open(LEAKY_FNS)))

# same two-stage leakage filter as build_fulldata_split.py
pool = df[
    ~df.gene_symbol.isin(test_genes) &
    (df.filename.isin(dpbs_fns) | ~df.filename.isin(leaky_fns))
].copy()
print(f"Clean pool: {len(pool)} entries, {pool.gene_symbol.nunique()} unique genes")

# gene → majority family
gene2fam = (
    pool.groupby("gene_symbol")["family_name"]
    .agg(lambda x: x.value_counts().index[0])
    .to_dict()
)
genes    = sorted(pool.gene_symbol.unique())
fam_list = sorted(set(gene2fam[g] for g in genes))

# assign genes to folds: within each family, shuffle then round-robin
gene2fold = {}
for fam in fam_list:
    fam_genes = [g for g in genes if gene2fam[g] == fam]
    random.shuffle(fam_genes)
    for i, g in enumerate(fam_genes):
        gene2fold[g] = i % N_FOLDS

os.makedirs(OUT_DIR, exist_ok=True)

pool_fns     = list(pool.filename)
test_fns_list = sorted(test_fns)

for fold in range(N_FOLDS):
    val_genes = {g for g, f in gene2fold.items() if f == fold}
    val_fns   = pool[pool.gene_symbol.isin(val_genes)].filename.tolist()
    train_fns = pool[~pool.gene_symbol.isin(val_genes)].filename.tolist()

    # source breakdown for metadata
    src = pool[~pool.gene_symbol.isin(val_genes)].source.value_counts().to_dict()

    split = {
        "train": sorted(train_fns),
        "val":   sorted(val_fns),
        "test":  test_fns_list,
        "metadata": {
            "fold": fold,
            "n_folds": N_FOLDS,
            "n_train": len(train_fns),
            "n_val":   len(val_fns),
            "n_test":  len(test_fns_list),
            "strategy": "gene-stratified-family-balanced, 40%-DBD-identity leakage filtered",
            "train_sources": src,
        }
    }
    path = os.path.join(OUT_DIR, f"fold_{fold}.json")
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"fold {fold}: train={len(train_fns)}, val={len(val_fns)}, test={len(test_fns_list)} | sources={src}")

print(f"\nWrote {N_FOLDS} splits to {OUT_DIR}/")
