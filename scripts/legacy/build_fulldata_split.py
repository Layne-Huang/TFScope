#!/usr/bin/env python
"""Build train/val split for full-data (JASPAR+HOCOMOCO+CIS-BP+DeepPBS) canonical training.

Leakage filters applied in two stages:
  1. Gene-level: exclude entries whose gene_symbol matches any of the 130 test genes
  2. Sequence-level: exclude extra (non-DeepPBS) entries whose DBD has >=40% identity
     to any test DBD — computed by pairwise global alignment + k-mer pre-filter.
     Leaky filenames saved in leaky_extra_fns.json by the leakage-check script.
     DeepPBS structure entries (520) are kept as-is (same leakage level as DeepPBS itself).

- Pool after filtering: ~3403 entries
- Val: ~5% gene-stratified holdout for early stopping
- Test: same 130 DeepPBS blind benchmark filenames

Output: data/processed/splits/fulldata_canon/train_val.json
"""
import json, os, random
import numpy as np
import pandas as pd

PARQUET      = "data/processed/tf_pwm_aug_dbd_canon.parquet"
TEST_SPLIT   = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
DPBS_PARQUET = "data/processed/tf_pwm_deeppbs_only.parquet"
LEAKY_FNS    = "data/processed/splits/fulldata_canon/leaky_extra_fns.json"
OUT_DIR      = "data/processed/splits/fulldata_canon"
VAL_FRAC     = 0.05
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)

df        = pd.read_parquet(PARQUET)
df_dpbs   = pd.read_parquet(DPBS_PARQUET)
dpbs_fns  = set(df_dpbs.filename)

test_split  = json.load(open(TEST_SPLIT))
test_fns    = set(test_split["test"])
test_genes  = set(df[df.filename.isin(test_fns)].gene_symbol)
leaky_fns   = set(json.load(open(LEAKY_FNS)))

# filter 1: gene-level
gene_clean = df[~df.gene_symbol.isin(test_genes)].copy()
# filter 2: remove leaky extra entries (DeepPBS structure entries are exempt)
safe = gene_clean[
    gene_clean.filename.isin(dpbs_fns) |        # DeepPBS entries: keep
    ~gene_clean.filename.isin(leaky_fns)         # extras: only if not leaky
].copy()

n_removed = len(gene_clean) - len(safe)
print(f"Gene-clean pool: {len(gene_clean)} | Removed leaky extra: {n_removed}")
print(f"Safe pool: {len(safe)} entries, {safe.gene_symbol.nunique()} unique genes")
print(f"Sources: {safe.source.value_counts().to_dict()}")

# gene-stratified val: hold out VAL_FRAC of genes (spread across families)
gene2fam = (
    safe.groupby("gene_symbol")["family_name"]
    .agg(lambda x: x.value_counts().index[0])
    .to_dict()
)
genes = sorted(safe.gene_symbol.unique())
fam_list = sorted(set(gene2fam[g] for g in genes))

val_genes = set()
for fam in fam_list:
    fam_genes = [g for g in genes if gene2fam[g] == fam]
    n_val = max(1, round(len(fam_genes) * VAL_FRAC))
    random.shuffle(fam_genes)
    val_genes.update(fam_genes[:n_val])

val_fns   = list(safe[safe.gene_symbol.isin(val_genes)]["filename"])
train_fns = list(safe[~safe.gene_symbol.isin(val_genes)]["filename"])

print(f"Train: {len(train_fns)} | Val: {len(val_fns)} | Test: {len(test_fns)}")
print(f"Val genes: {len(val_genes)} | Train genes: {safe[~safe.gene_symbol.isin(val_genes)].gene_symbol.nunique()}")

os.makedirs(OUT_DIR, exist_ok=True)
split = {
    "train": sorted(train_fns),
    "val":   sorted(val_fns),
    "test":  sorted(test_fns),
    "metadata": {
        "description": "Full-data canonical: JASPAR+HOCOMOCO+CIS-BP+DeepPBS train, DeepPBS blind test. Leakage-filtered: gene-level + 40% DBD identity filter on extra entries.",
        "n_train": len(train_fns),
        "n_val":   len(val_fns),
        "n_test":  len(test_fns),
        "val_frac": VAL_FRAC,
        "strategy": "gene-stratified-family-balanced val, test genes excluded from pool",
        "sources": safe.source.value_counts().to_dict(),
    }
}
out_path = os.path.join(OUT_DIR, "train_val.json")
with open(out_path, "w") as f:
    json.dump(split, f, indent=2)
print(f"Wrote {out_path}")
