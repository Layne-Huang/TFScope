#!/usr/bin/env python
"""Build TFScope split JSONs that mirror DeepPBS's 5-fold benchmark.

Outputs:
  data/processed/splits/deeppbs/benchmark.json   — primary comparison (all 130 test TFs)
  data/processed/splits/deeppbs/fold{0-4}.json   — per-fold for variance analysis

In every split:
  test  = TFScope filenames for DeepPBS validation TFs
  train = ALL remaining TFScope filenames (DeepPBS train TFs + TFScope-only TFs)
  val   = 10% stratified sample of train (by family_id)

No DeepPBS benchmark TF appears in any train/val set → fair comparison.
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

FOLDS_DIR = "/n/home13/leihuang/project/DeepPBS/run/folds"
PARQUET    = "data/processed/tf_pwm.parquet"
OUT_DIR    = "data/processed/splits/deeppbs"

# H11MO uses non-standard gene abbreviations; map to UniProt gene symbols
H11MO_ALIAS = {
    "GCR":    "NR3C1",
    "TF65":   "RELA",
    "BMAL1":  "ARNTL",
    "PO2F1":  "POU2F1",
    "PO2F2":  "POU2F2",
    "PO2F3":  "POU2F3",
    "PO3F1":  "POU3F1",
    "PO3F2":  "POU3F2",
    "PO3F3":  "POU3F3",
    "PO4F1":  "POU4F1",
    "PO5F1":  "POU5F1",
    "NFAC1":  "NFATC1",
    "NFAC2":  "NFATC2",
    "NFAC3":  "NFATC3",
    "NFAC4":  "NFATC4",
    "KAISO":  "ZBTB33",
    "SUH":    "RBPJ",
    "HXA1":   "HOXA1",
    "HXA2":   "HOXA2",
    "HXA5":   "HOXA5",
    "HXA9":   "HOXA9",
    "HXA10":  "HOXA10",
    "HXA11":  "HOXA11",
    "HXA13":  "HOXA13",
    "HXB13":  "HOXB13",
    "HXB4":   "HOXB4",
    "HXB7":   "HOXB7",
    "HXC10":  "HOXC10",
    "HXD13":  "HOXD13",
    "NDF1":   "NEUROD1",
    "NDF2":   "NEUROD2",
    "PRGR":   "PGR",
    "ZBT7A":  "ZBTB7A",
    "ZBT7B":  "ZBTB7B",
    "TFE2":   "TCF4",
    "ITF2":   "TCF4",
    "BRAC":   "T",
    "STF1":   "NR5A1",
    "NKX25":  "NKX2-5",
    "NKX21":  "NKX2-1",
    "NKX22":  "NKX2-2",
    "COE1":   "EBF1",
    "COE2":   "EBF2",
    "COE3":   "EBF3",
    "PKNX1":  "PKNOX1",
    "PKNX2":  "PKNOX2",
    "PO6F1":  "POU6F1",
    "HESX1":  "HESX1",
}


def parse_tf_id(npz_name: str) -> tuple[str, str]:
    """Return (kind, identifier) from NPZ basename.

    kind = 'hocomoco' or 'jaspar'
    identifier = canonical gene symbol (hocomoco) or MA-prefix (jaspar)
    """
    base = npz_name.replace(".npz", "")
    # strip leading pdb_chain_ prefix: everything up to and including the last _ before TF id
    tf_id = re.sub(r"^[^_]+_[A-Z]_", "", base)

    if "jaspar" in tf_id.lower():
        # e.g. MA0039.2.jaspar → MA0039
        ma = re.match(r"(MA\d+)", tf_id, re.IGNORECASE)
        return ("jaspar", ma.group(1).upper()) if ma else ("unknown", tf_id)
    else:
        # e.g. FOS_HUMAN.H11MO.0.A → FOS
        gene = re.sub(r"_(HUMAN|MOUSE|RAT|DROME|YEAST|CAEEL).*", "", tf_id)
        gene = H11MO_ALIAS.get(gene, gene)
        return ("hocomoco", gene.upper())


def load_fold(fold_idx: int) -> tuple[list[str], list[str]]:
    train_file = os.path.join(FOLDS_DIR, f"train{fold_idx}.txt")
    valid_file  = os.path.join(FOLDS_DIR, f"valid{fold_idx}.txt")
    with open(train_file) as f:
        train = [l.strip() for l in f if l.strip()]
    with open(valid_file) as f:
        valid = [l.strip() for l in f if l.strip()]
    return train, valid


def npz_list_to_gene_sets(npz_list: list[str]) -> tuple[set[str], set[str]]:
    """Return (hocomoco_genes, jaspar_ma_ids) from a list of NPZ filenames."""
    hoco, jasp = set(), set()
    for name in npz_list:
        kind, ident = parse_tf_id(name)
        if kind == "hocomoco":
            hoco.add(ident)
        elif kind == "jaspar":
            jasp.add(ident)
    return hoco, jasp


def match_filenames(df: pd.DataFrame,
                    hoco_genes: set[str],
                    jaspar_ids: set[str]) -> list[str]:
    """Return ALL TFScope filenames for TFs matching the given sets.

    Matching is done by gene_symbol (HOCOMOCO) or JASPAR MA-prefix, then
    expanded to include every entry for those genes regardless of source.
    This prevents the same gene from leaking across train/test splits.
    """
    # HOCOMOCO match: by gene_symbol
    hoco_mask = df["gene_symbol"].str.upper().isin(hoco_genes)

    # JASPAR match: source_id prefix (MA number, ignore version)
    def ma_prefix(sid: str) -> str:
        m = re.match(r"(MA\d+)", str(sid), re.IGNORECASE)
        return m.group(1).upper() if m else ""

    df_jasp = df[df["source"] == "JASPAR"].copy()
    df_jasp["ma_prefix"] = df_jasp["source_id"].apply(ma_prefix)
    jasp_matched_idx = df_jasp[df_jasp["ma_prefix"].isin(jaspar_ids)].index
    jasp_genes = set(df.loc[jasp_matched_idx, "gene_symbol"].str.upper())

    # Expand to ALL entries for matched genes (any source)
    all_matched_genes = hoco_genes | jasp_genes
    matched = df[df["gene_symbol"].str.upper().isin(all_matched_genes)]["filename"].tolist()
    return matched


def stratified_val_split(filenames: list[str],
                         df: pd.DataFrame,
                         val_frac: float = 0.10,
                         seed: int = 42) -> tuple[list[str], list[str]]:
    """Split filenames into train/val with stratification by family_id."""
    rng = np.random.default_rng(seed)
    fn_set = set(filenames)
    sub = df[df["filename"].isin(fn_set)].copy()

    val_idx = []
    for fam, grp in sub.groupby("family_id"):
        n_val = max(1, int(len(grp) * val_frac))
        chosen = rng.choice(grp.index.tolist(), size=n_val, replace=False)
        val_idx.extend(chosen.tolist())

    val_set  = set(sub.loc[val_idx, "filename"].tolist())
    train_fn = [f for f in filenames if f not in val_set]
    val_fn   = [f for f in filenames if f in val_set]
    return train_fn, val_fn


def build_split(train_fns: list[str],
                val_fns: list[str],
                test_fns: list[str],
                metadata: dict) -> dict:
    return {
        "train": sorted(set(train_fns)),
        "val":   sorted(set(val_fns)),
        "test":  sorted(set(test_fns)),
        "metadata": metadata,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(PARQUET)
    all_fns = set(df["filename"].tolist())

    # Collect test TF sets per fold
    fold_test_fns  = []   # list of lists (one per fold)
    fold_hoco      = []
    fold_jasp      = []
    missing_report = []

    for i in range(5):
        train_npz, valid_npz = load_fold(i)
        t_hoco, t_jasp = npz_list_to_gene_sets(train_npz)
        v_hoco, v_jasp = npz_list_to_gene_sets(valid_npz)

        test_fns = match_filenames(df, v_hoco, v_jasp)

        # Report missing
        matched_genes = set(df[df["filename"].isin(test_fns)]["gene_symbol"].str.upper())
        matched_jasp  = set()
        if test_fns:
            sub = df[df["filename"].isin(test_fns)]
            sub_j = sub[sub["source"] == "JASPAR"]["source_id"].apply(
                lambda s: re.match(r"(MA\d+)", str(s), re.I).group(1).upper()
                if re.match(r"(MA\d+)", str(s), re.I) else "")
            matched_jasp = set(sub_j.tolist())

        missing_hoco = v_hoco - matched_genes
        missing_jasp = v_jasp - matched_jasp
        missing_report.append({
            "fold": i,
            "missing_hocomoco": sorted(missing_hoco),
            "missing_jaspar":   sorted(missing_jasp),
        })

        fold_test_fns.append(test_fns)
        fold_hoco.append((t_hoco, v_hoco))
        fold_jasp.append((t_jasp, v_jasp))

    # ── Per-fold splits ────────────────────────────────────────────────────────
    for i in range(5):
        test_fns  = fold_test_fns[i]
        test_set  = set(test_fns)
        train_all = [f for f in all_fns if f not in test_set]
        train_fns, val_fns = stratified_val_split(train_all, df)

        meta = {
            "fold":           i,
            "n_test_tfs":     len(set(df[df["filename"].isin(test_fns)]["gene_symbol"])),
            "n_test_samples": len(test_fns),
            "n_train_samples": len(train_fns),
            "n_val_samples":   len(val_fns),
            "missing_hocomoco": missing_report[i]["missing_hocomoco"],
            "missing_jaspar":   missing_report[i]["missing_jaspar"],
        }
        split = build_split(train_fns, val_fns, test_fns, meta)
        out_path = os.path.join(OUT_DIR, f"fold{i}.json")
        with open(out_path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"fold{i}: test={meta['n_test_samples']} samples "
              f"({meta['n_test_tfs']} TFs)  "
              f"train={meta['n_train_samples']}  val={meta['n_val_samples']}  "
              f"missing_hoco={len(missing_report[i]['missing_hocomoco'])}  "
              f"missing_jasp={len(missing_report[i]['missing_jaspar'])}")

    # ── Benchmark split (union of all 5 valid sets) ────────────────────────────
    all_test_fns = sorted(set(fn for fns in fold_test_fns for fn in fns))
    test_set     = set(all_test_fns)
    train_all    = [f for f in all_fns if f not in test_set]
    train_fns, val_fns = stratified_val_split(train_all, df)

    all_missing_hoco = sorted(set(g for r in missing_report for g in r["missing_hocomoco"]))
    all_missing_jasp = sorted(set(g for r in missing_report for g in r["missing_jaspar"]))

    meta = {
        "description":     "Union of all 5 DeepPBS validation folds — primary benchmark",
        "n_test_tfs":      len(set(df[df["filename"].isin(all_test_fns)]["gene_symbol"])),
        "n_test_samples":  len(all_test_fns),
        "n_train_samples": len(train_fns),
        "n_val_samples":   len(val_fns),
        "missing_hocomoco": all_missing_hoco,
        "missing_jaspar":   all_missing_jasp,
    }
    split = build_split(train_fns, val_fns, all_test_fns, meta)
    out_path = os.path.join(OUT_DIR, "benchmark.json")
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"\nbenchmark: test={meta['n_test_samples']} samples "
          f"({meta['n_test_tfs']} TFs)  "
          f"train={meta['n_train_samples']}  val={meta['n_val_samples']}")
    print(f"  missing HOCOMOCO ({len(all_missing_hoco)}): {all_missing_hoco}")
    print(f"  missing JASPAR   ({len(all_missing_jasp)}): {all_missing_jasp[:10]}{'...' if len(all_missing_jasp)>10 else ''}")


if __name__ == "__main__":
    main()
