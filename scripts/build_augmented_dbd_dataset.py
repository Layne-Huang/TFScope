#!/usr/bin/env python
"""Build the augmented training dataset for v11.

Sources:
  - DeepPBS-only train+val (520 entries, crystal-chain sequences ≈ DBD)
  - TFScope-original (3743 entries), filtered to:
      * exclude any row overlapping the blind test by gene_symbol or source_id stem
      * keep only rows with DBD-length ≥ 20  (good InterPro annotation)
      * truncate `sequence` to `sequence[dbd_start:dbd_end]`, then reset dbd_start=0,
        dbd_end=len  (so the model sees only DBD residues — consistent with the
        DeepPBS-only entries that are already DBD-only crystal chains).

Output:
  data/processed/tf_pwm_aug_dbd.parquet                — augmented training data
  data/processed/splits/deeppbs_aug_dbd/benchmark.json — split (test = 130 blind)
"""
import json, os, re
import numpy as np
import pandas as pd

DEEPPBS_PARQUET = "data/processed/tf_pwm_deeppbs_only.parquet"
ORIG_PARQUET    = "data/processed/tf_pwm.parquet"
SPLIT_SRC       = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
OUT_PARQUET     = "data/processed/tf_pwm_aug_dbd.parquet"
OUT_SPLIT_DIR   = "data/processed/splits/deeppbs_aug_dbd"
MIN_DBD_LEN     = 20


def src_key(s):
    s = str(s).upper()
    m = re.match(r"(MA\d+)", s)
    if m: return m.group(1)
    m = re.match(r"([A-Z0-9\-]+)\.H11MO", s)
    if m: return m.group(1) + ".H11MO"
    return s


def main():
    os.makedirs(OUT_SPLIT_DIR, exist_ok=True)

    df_dp  = pd.read_parquet(DEEPPBS_PARQUET)
    df_or  = pd.read_parquet(ORIG_PARQUET)
    split  = json.load(open(SPLIT_SRC))
    test_fns = set(split["test"])
    test_df  = df_dp[df_dp["filename"].isin(test_fns)]
    test_genes = set(test_df["gene_symbol"].str.upper())
    test_src   = set(test_df["source_id"].apply(src_key))

    # ── 1. DeepPBS-only train+val (already DBD-equivalent crystal chains) ──
    dp_keep_fns = set(split["train"]) | set(split["val"])
    df_dp_train = df_dp[df_dp["filename"].isin(dp_keep_fns)].copy()
    # Normalise DBD to span the full crystal chain (it already does, but be explicit)
    df_dp_train["dbd_start"] = 0
    df_dp_train["dbd_end"]   = df_dp_train["seq_length"].astype(int)
    df_dp_train["origin"]    = "deeppbs"
    print(f"DeepPBS-only train+val:                 {len(df_dp_train)} rows")

    # ── 2. TFScope-original, filtered + DBD-truncated ─────────────────────
    orig_genes = df_or["gene_symbol"].str.upper()
    orig_src   = df_or["source_id"].apply(src_key)
    keep = (~orig_genes.isin(test_genes)) & (~orig_src.isin(test_src))
    df_or_clean = df_or[keep].copy()
    print(f"TFScope-original after test-overlap drop: {len(df_or_clean)} rows  "
          f"(excluded {(~keep).sum()})")

    # Drop rows whose annotated DBD is too short (likely InterPro failure)
    dbd_len = (df_or_clean["dbd_end"] - df_or_clean["dbd_start"]).astype(int)
    df_or_clean = df_or_clean[dbd_len >= MIN_DBD_LEN].copy()
    print(f"TFScope-original after dbd_len >= {MIN_DBD_LEN}:    {len(df_or_clean)} rows")

    # DBD-truncate the sequence; reset dbd_start/end to span the new (DBD-only) sequence
    def trunc(row):
        seq = row["sequence"]
        ds, de = int(row["dbd_start"]), int(row["dbd_end"])
        ds = max(0, min(ds, len(seq)))
        de = max(ds + 1, min(de, len(seq)))
        return seq[ds:de]
    df_or_clean["sequence"]   = df_or_clean.apply(trunc, axis=1)
    df_or_clean["seq_length"] = df_or_clean["sequence"].str.len()
    df_or_clean["dbd_start"]  = 0
    df_or_clean["dbd_end"]    = df_or_clean["seq_length"]
    df_or_clean["origin"]     = "tfscope_orig"

    # Tag filenames so they're disjoint from DeepPBS-only and easy to identify
    df_or_clean["filename"] = "ORIG__" + df_or_clean["filename"].astype(str)

    # ── 3. Combine and align columns ─────────────────────────────────────
    cols = list(df_dp_train.columns)
    # Make sure every column the model needs exists in both DataFrames
    for c in cols:
        if c not in df_or_clean.columns and c != "origin":
            print(f"  WARN: column {c} missing in TFScope-original; filling with NaN/empty")
            df_or_clean[c] = ""
    df_or_clean = df_or_clean[cols]

    df_aug = pd.concat([df_dp_train, df_or_clean], ignore_index=True)
    print(f"\nAugmented dataset:                       {len(df_aug)} rows total")
    print(f"  from deeppbs:                          {(df_aug['origin']=='deeppbs').sum()}")
    print(f"  from tfscope_orig (DBD-only):          {(df_aug['origin']=='tfscope_orig').sum()}")
    print(f"  unique gene_symbols:                   {df_aug['gene_symbol'].nunique()}")

    # ── 4. Append test rows so evaluate.py can find them in the same parquet ─
    test_rows = df_dp[df_dp["filename"].isin(test_fns)].copy()
    test_rows["origin"] = "deeppbs_test"
    test_rows = test_rows.reindex(columns=cols)         # align column order/types
    df_full = pd.concat([df_aug, test_rows], ignore_index=True)
    df_full.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET}  ({len(df_aug)} train + {len(test_rows)} test = {len(df_full)} rows)")

    train_fns = df_aug["filename"].tolist()
    # Small placeholder val for early-stop compatibility (matches v8/v9/v10 single-fold setup)
    import random; random.seed(42); random.shuffle(train_fns)
    val_fns = train_fns[:16]
    train_fns_final = train_fns[16:]
    out_split = {
        "train": sorted(train_fns_final),
        "val":   sorted(val_fns),
        "test":  sorted(test_fns),
        "metadata": {
            "description": "Augmented DBD-only training: DeepPBS crystal chains + TFScope-orig DBD trunc; test=130 blind",
            "n_train": len(train_fns_final),
            "n_val":   len(val_fns),
            "n_test":  len(test_fns),
            "n_excluded_overlap": int((~keep).sum()),
        },
    }
    split_path = os.path.join(OUT_SPLIT_DIR, "benchmark.json")
    with open(split_path, "w") as f:
        json.dump(out_split, f, indent=2)
    print(f"Wrote {split_path}")
    print(f"  train={len(train_fns_final)}, val={len(val_fns)}, test={len(test_fns)}")


if __name__ == "__main__":
    main()
