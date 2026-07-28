#!/usr/bin/env python
"""Build a family-stratified, 40%-identity cluster-wise split of the DeepPBS set.

Both DeepPBS and TFScope train/eval on the SAME partition so the comparison is
fair. Redundancy is removed at the DNA-binding-domain level (MMseqs2 40% id):
no protein cluster spans train/val/test, so the test set is a genuine
sequence-level hold-out (unlike DeepPBS's original PDB-ID-only dedup).

Outputs:
  data/processed/splits/deeppbs_cluster40/split.json    (TFScope: filename lists)
  data/processed/splits/deeppbs_cluster40/folds/train.txt|valid.txt|test.txt
        (DeepPBS: <pdb>_<chain>_<motif>.npz names)
  data/processed/splits/deeppbs_cluster40/clusters.json (audit: cluster→split)
"""
import json, os
from collections import defaultdict
import numpy as np
import pandas as pd

PARQUET   = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
CLUSTER   = "data/processed/deeppbs_cluster/dbd_id0.4_cluster.tsv"
OUT_DIR   = "data/processed/splits/deeppbs_cluster40"
DEEPPBS_DATA = "/data1/leihuang/DeepPBS/deeppbsmar24/data/assembly2024"
TARGET    = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED      = 42


def npz_name_for(filename, df_row, available):
    """Map a TFScope parquet filename to its DeepPBS npz name.

    Parquet filename looks like '2evi_A_NDT80.MA0343.1.txt'; the npz is
    '2evi_A_MA0343.1.jaspar.npz' or '<pdb>_<chain>_<motif>.npz'. We resolve by
    matching pdb_chain prefix against the available npz files on disk.
    """
    pdb_chain = "_".join(filename.split("_")[:2])
    cands = [n for n in available if n.startswith(pdb_chain + "_")]
    if not cands:
        return None
    src = str(df_row.get("source_id", "")).upper()
    for n in cands:
        if src and src.replace(".", "") in n.upper().replace(".", ""):
            return n
    return cands[0]


def main():
    os.makedirs(os.path.join(OUT_DIR, "folds"), exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(PARQUET)
    fn2fam = dict(zip(df["filename"], df["family_name"]))

    # cluster → members
    rep2mem = defaultdict(list)
    for line in open(CLUSTER):
        rep, mem = line.strip().split("\t")
        rep2mem[rep].append(mem)
    clusters = list(rep2mem.items())

    # dominant family per cluster
    def dom_family(mem):
        c = defaultdict(int)
        for m in mem:
            c[fn2fam.get(m, "NA")] += 1
        return max(c, key=c.get)

    # group clusters by dominant family, then stratify each family's clusters
    fam2clusters = defaultdict(list)
    for rep, mem in clusters:
        fam2clusters[dom_family(mem)].append((rep, mem))

    total = sum(len(m) for _, m in clusters)
    assigned = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    # Stratify WITHIN each family: allocate that family's clusters to
    # train/val/test toward 70/15/15 of the family's own structures, so every
    # family with >=3 clusters appears in all three splits. Families with <3
    # clusters fall back to train+test (val skipped) to keep test coverage.
    for fam in sorted(fam2clusters):
        fc = sorted(fam2clusters[fam], key=lambda kv: -len(kv[1]))  # big->small
        fam_total = sum(len(m) for _, m in fc)
        local = {"train": 0, "val": 0, "test": 0}
        order = ["train", "val", "test"] if len(fc) >= 3 else ["train", "test"]
        loc_target = {s: TARGET[s] for s in order}
        norm = sum(loc_target.values())
        loc_target = {s: loc_target[s] / norm for s in order}
        for rep, mem in fc:
            deficit = {s: loc_target[s] * fam_total - local[s] for s in order}
            s = max(deficit, key=deficit.get)
            assigned[s].append(rep)
            local[s] += len(mem)
            counts[s] += len(mem)

    # expand to filename lists
    split_files = {s: [] for s in assigned}
    for s, reps in assigned.items():
        for rep in reps:
            split_files[s].extend(rep2mem[rep])

    # TFScope split.json
    split_json = {
        "train": sorted(split_files["train"]),
        "val":   sorted(split_files["val"]),
        "test":  sorted(split_files["test"]),
    }
    with open(os.path.join(OUT_DIR, "split.json"), "w") as fh:
        json.dump(split_json, fh, indent=2)

    # DeepPBS fold files (map to npz names)
    available = [n for n in os.listdir(DEEPPBS_DATA) if n.endswith(".npz")]
    df_idx = df.set_index("filename")
    n_mapped, n_missing = 0, 0
    for s in ("train", "val", "test"):
        names = []
        for fn in split_files[s]:
            row = df_idx.loc[fn] if fn in df_idx.index else {}
            npz = npz_name_for(fn, row, available)
            if npz:
                names.append(npz); n_mapped += 1
            else:
                n_missing += 1
        out = "valid" if s == "val" else s
        with open(os.path.join(OUT_DIR, "folds", f"{out}.txt"), "w") as fh:
            fh.write("\n".join(sorted(set(names))) + "\n")

    # audit
    with open(os.path.join(OUT_DIR, "clusters.json"), "w") as fh:
        json.dump({s: assigned[s] for s in assigned}, fh, indent=2)

    print(f"Clusters: {len(clusters)} | structures: {total}")
    print(f"  train: {counts['train']} structs / {len(assigned['train'])} clusters")
    print(f"  val:   {counts['val']} structs / {len(assigned['val'])} clusters")
    print(f"  test:  {counts['test']} structs / {len(assigned['test'])} clusters")
    print(f"npz mapped: {n_mapped}, missing: {n_missing}")

    # family coverage check
    print("\nFamily coverage:")
    for s in ("train", "val", "test"):
        fams = defaultdict(int)
        for fn in split_files[s]:
            fams[fn2fam.get(fn, "NA")] += 1
        print(f"  {s}: {dict(sorted(fams.items(), key=lambda x: -x[1]))}")

    # leakage assertion: no cluster spans splits
    seen = {}
    ok = True
    for s, reps in assigned.items():
        for rep in reps:
            if rep in seen:
                ok = False
            seen[rep] = s
    print(f"\nNo cluster spans splits: {ok}")


if __name__ == "__main__":
    main()
