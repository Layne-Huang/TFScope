#!/usr/bin/env python
"""Leakage-free train/val/test splits at 40% sequence identity (MMseqs2).

Design (as specified):
  TEST     = rows from DeepPBS's own blind test set (run/folds/id.txt)
  EXCLUDED = every OTHER row falling in the same 40%-identity cluster as any
             test row -- dropped entirely, so nothing homologous to test can
             appear in train
  VAL      = whole clusters held out (cluster-disjoint from train and test)
  TRAIN    = everything else

Clustering is done jointly over BOTH datasets (structure + sequence-only) on
the DBD crop, so a gene's structural and sequence rows can never straddle the
split boundary.
"""
import json, os, subprocess, sys
import numpy as np
import pandas as pd

AUG = "data/processed/tf_pwm_aug_dbd_canon_trim_v2.parquet"
STR = "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"
IDTXT = "/afs/csail.mit.edu/u/l/leihuang/project/DeepPBS/run/folds/id.txt"
WORK = "/tmp/mmseqs40"
MIN_ID, COV = 0.4, 0.8
VAL_FRAC = 0.12
SEED = 42


def main():
    os.makedirs(WORK, exist_ok=True)
    aug = pd.read_parquet(AUG); st = pd.read_parquet(STR)
    aug["_set"], st["_set"] = "seq", "struct"
    aug["_gene"] = aug["gene_symbol"].astype(str).str.upper()
    st["_gene"] = st["gene"].astype(str).str.upper()
    aug["_rid"] = ["seq_%d" % i for i in range(len(aug))]
    st["_rid"] = ["str_%d" % i for i in range(len(st))]

    # --- TEST membership: any row from a PDB in DeepPBS's blind test set ---
    ids = [l.strip() for l in open(IDTXT) if l.strip()]
    test_pdbs = {e.split("_")[0].upper() for e in ids}
    st["_is_test"] = st["pdb_id"].astype(str).str.upper().isin(test_pdbs)
    aug["_is_test"] = False
    print(f"DeepPBS blind-test PDBs: {len(test_pdbs)}   matching rows in our structure set: {int(st['_is_test'].sum())}")

    allrows = pd.concat([aug, st], ignore_index=True)

    # --- cluster jointly at 40% identity ---
    fa = os.path.join(WORK, "all.fasta")
    with open(fa, "w") as f:
        for rid, s in zip(allrows["_rid"], allrows["sequence"].astype(str)):
            f.write(f">{rid}\n{s}\n")
    print(f"clustering {len(allrows)} sequences at {int(MIN_ID*100)}% identity ...", flush=True)
    subprocess.run(["mmseqs", "easy-cluster", fa, os.path.join(WORK, "clu"),
                    os.path.join(WORK, "tmp"), "--min-seq-id", str(MIN_ID),
                    "-c", str(COV), "--cov-mode", "1", "-v", "1"],
                   check=True, capture_output=True)
    clu = pd.read_csv(os.path.join(WORK, "clu_cluster.tsv"), sep="\t",
                       names=["rep", "member"])
    rid2clu = dict(zip(clu["member"], clu["rep"]))
    allrows["_clu"] = allrows["_rid"].map(rid2clu)
    missing = allrows["_clu"].isna().sum()
    if missing:
        allrows.loc[allrows["_clu"].isna(), "_clu"] = allrows.loc[allrows["_clu"].isna(), "_rid"]
    print(f"clusters: {allrows['_clu'].nunique()}  (singleton-fallback for {missing} unclustered)")

    # --- assign splits ---
    test_clusters = set(allrows[allrows["_is_test"]]["_clu"])
    in_test_clu = allrows["_clu"].isin(test_clusters)
    allrows["split"] = "train"
    allrows.loc[in_test_clu & allrows["_is_test"], "split"] = "test"
    allrows.loc[in_test_clu & ~allrows["_is_test"], "split"] = "excluded"

    # val: whole clusters from what remains
    rest = allrows[allrows["split"] == "train"]
    rng = np.random.default_rng(SEED)
    cl = rest.groupby("_clu").size().sort_values(ascending=False)
    order = rng.permutation(cl.index.to_numpy())
    target, chosen, acc = int(len(rest) * VAL_FRAC), [], 0
    for c in order:
        if acc >= target: break
        chosen.append(c); acc += int(cl[c])
    allrows.loc[allrows["_clu"].isin(chosen), "split"] = "val"

    print("\n=== split summary ===")
    print(allrows.groupby(["split", "_set"]).size().unstack(fill_value=0).to_string())
    print(f"\nrows: {allrows['split'].value_counts().to_dict()}")
    for s in ["train", "val", "test"]:
        sub = allrows[allrows["split"] == s]
        print(f"  {s:<9} clusters={sub['_clu'].nunique():>5}  genes={sub['_gene'].nunique():>5}")

    # --- leakage audit ---
    print("\n=== leakage audit (must all be 0) ===")
    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
        ca = set(allrows[allrows["split"] == a]["_clu"])
        cb = set(allrows[allrows["split"] == b]["_clu"])
        ga = set(allrows[allrows["split"] == a]["_gene"])
        gb = set(allrows[allrows["split"] == b]["_gene"])
        print(f"  {a}/{b}: shared clusters={len(ca & cb)}  shared genes={len(ga & gb)}")

    out = {s: {"seq": allrows[(allrows.split == s) & (allrows._set == "seq")]["_rid"].tolist(),
               "struct": allrows[(allrows.split == s) & (allrows._set == "struct")]["_rid"].tolist()}
           for s in ["train", "val", "test", "excluded"]}
    os.makedirs("data/processed/splits/mmseqs40_deeppbstest", exist_ok=True)
    with open("data/processed/splits/mmseqs40_deeppbstest/split.json", "w") as f:
        json.dump(out, f)
    allrows[["_rid", "_set", "_gene", "_clu", "split"]].to_parquet(
        "data/processed/splits/mmseqs40_deeppbstest/assignments.parquet")
    print("\nsaved data/processed/splits/mmseqs40_deeppbstest/")


if __name__ == "__main__":
    main()
