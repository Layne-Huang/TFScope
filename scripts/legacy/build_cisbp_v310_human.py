#!/usr/bin/env python
"""Build a fresh human CIS-BP v3.10 PWM table to REPLACE our stale v1.94
CIS-BP records.

v1.94 -> v3.10 is three major versions; spot-check confirmed the upgrade
removes junk (e.g. PHA2 / M01568_1, a yeast prephenate dehydratase carrying a
degenerate 2bp "motif", is gone entirely in v3.10).

Keeps only: TF_Species == Homo_sapiens AND TF_Status == 'D' (direct
experimental evidence, not inferred-by-similarity). Also drops motifs whose
PWM file is missing/empty or shorter than MIN_MOTIF_LEN columns.
"""
import os, sys, json
import numpy as np
import pandas as pd

CISBP_DIR = "data/raw/cisbp_v3_10"
PWM_DIR = os.path.join(CISBP_DIR, "pwms")
MIN_MOTIF_LEN = 5          # a 2bp "motif" carries no usable specificity
OUT = "data/processed/cisbp_v310_human_pwms.parquet"


def load_pwm(motif_id):
    """Return (4, L) float32 ACGT matrix, or None if missing/degenerate."""
    path = os.path.join(PWM_DIR, f"{motif_id}.txt")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return None
    if df.empty or not {"A", "C", "G", "T"}.issubset(df.columns):
        return None
    mat = df[["A", "C", "G", "T"]].to_numpy(dtype=np.float32).T   # (4, L)
    if mat.shape[1] < MIN_MOTIF_LEN or not np.isfinite(mat).all():
        return None
    # renormalise columns defensively
    colsums = mat.sum(axis=0, keepdims=True)
    if (colsums <= 0).any():
        return None
    return mat / colsums


def main():
    info = pd.read_csv(os.path.join(CISBP_DIR, "TF_Information.txt"),
                        sep="\t", low_memory=False)
    human = info[(info["TF_Species"] == "Homo_sapiens") &
                  (info["TF_Status"] == "D")].copy()
    print(f"human direct-evidence rows: {len(human)}", flush=True)

    rows, n_missing, n_short = [], 0, 0
    for i, r in enumerate(human.itertuples(index=False)):
        mat = load_pwm(r.Motif_ID)
        if mat is None:
            n_missing += 1
            continue
        if mat.shape[1] < MIN_MOTIF_LEN:
            n_short += 1
            continue
        rows.append({
            "gene_symbol": str(r.TF_Name),
            "motif_id": r.Motif_ID,
            "tf_id": r.TF_ID,
            "dbid": getattr(r, "DBID", ""),
            "family_name_cisbp": r.Family_Name,
            "dbds_cisbp": r.DBDs,
            "dbd_count_cisbp": r.DBD_Count,
            "motif_type": r.Motif_Type,
            "msource": r.MSource_Identifier,
            "pmid": r.PMID,
            "motif_length": mat.shape[1],
            "pwm": mat.astype(np.float32).tobytes(),
            "source": "CISBP_v3.10",
        })
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(human)}] kept={len(rows)}", flush=True)

    out = pd.DataFrame(rows)
    print(f"\nkept {len(out)} motifs  (missing/unreadable PWM: {n_missing}, too short: {n_short})", flush=True)
    print(f"distinct genes: {out['gene_symbol'].str.upper().nunique()}", flush=True)
    out.to_parquet(OUT)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
