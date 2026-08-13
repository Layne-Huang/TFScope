#!/usr/bin/env python
"""Build a cluster-based train/val split for the full augmented dataset.

Test set is fixed to the 130 DeepPBS benchmark TFs (benchmark_no_val.json).
The remaining 4117 TFs (1322 unique proteins) are clustered by 3-mer Jaccard
at ~30% identity threshold. Val = ~15% of clusters (family-stratified).
No cluster straddles the train/val boundary.

Output: data/processed/splits/fulldata_cluster_val/split.json
"""
import json, os, sys, time, logging
from collections import defaultdict
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATA       = "data/processed/tf_pwm_aug_dbd_canon.parquet"
TEST_SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
OUTDIR     = "data/processed/splits/fulldata_cluster_val"
CLUSTER_THRESH = 0.15   # 3-mer Jaccard ≈ 30% sequence identity
VAL_FRAC   = 0.15
SEED       = 42


def greedy_cluster(seqs, ids, thresh, k=3):
    reps = []
    assigned = {}
    for i, uid in enumerate(ids):
        s1 = set(seqs[uid][j:j+k] for j in range(len(seqs[uid])-k+1))
        found = False
        for ci, ri in enumerate(reps):
            s2 = set(seqs[ids[ri]][j:j+k] for j in range(len(seqs[ids[ri]])-k+1))
            inter = len(s1 & s2)
            jac = inter / (len(s1)+len(s2)-inter) if (len(s1)+len(s2)-inter) > 0 else 0.0
            if jac >= thresh:
                assigned[uid] = ci; found = True; break
        if not found:
            assigned[uid] = len(reps); reps.append(i)
    clusters = defaultdict(list)
    for uid, ci in assigned.items():
        clusters[ci].append(uid)
    return [clusters[ci] for ci in sorted(clusters)]


def main():
    df = pd.read_parquet(DATA)
    test_fns = set(json.load(open(TEST_SPLIT))["test"])
    pool = df[~df["filename"].isin(test_fns)].copy()
    log.info(f"Pool: {len(pool)} rows, {pool['uniprot_id'].nunique()} unique proteins | Test: {len(test_fns)}")

    prot_df = pool.drop_duplicates("uniprot_id")[["uniprot_id","sequence","family_id"]].reset_index(drop=True)
    seqs = dict(zip(prot_df["uniprot_id"], prot_df["sequence"]))
    fam  = dict(zip(prot_df["uniprot_id"], prot_df["family_id"]))
    ids  = list(prot_df["uniprot_id"])

    log.info(f"Clustering {len(ids)} unique proteins (Jaccard>={CLUSTER_THRESH}) ...")
    t0 = time.time()
    clusters = greedy_cluster(seqs, ids, CLUSTER_THRESH)
    log.info(f"  → {len(clusters)} clusters in {time.time()-t0:.1f}s")

    # Family-stratified: assign ~VAL_FRAC of clusters to val
    rng = np.random.RandomState(SEED)
    val_proteins, train_proteins = set(), set()

    fam_clusters = defaultdict(list)
    for c in clusters:
        families = [fam[u] for u in c if u in fam]
        dom = max(set(families), key=families.count) if families else 9
        fam_clusters[dom].append(c)

    for fid, fclust in fam_clusters.items():
        fclust = fclust.copy(); rng.shuffle(fclust)
        n_val = max(1, round(len(fclust) * VAL_FRAC))
        for i, c in enumerate(fclust):
            (val_proteins if i < n_val else train_proteins).update(c)

    def to_fns(proteins):
        return sorted(pool[pool["uniprot_id"].isin(proteins)]["filename"].tolist())

    train_fns = to_fns(train_proteins)
    val_fns   = to_fns(val_proteins)
    test_fns_list = sorted(test_fns)

    assert not (set(train_fns) & set(val_fns)), "train/val overlap!"
    assert not (set(train_fns) & set(test_fns)), "train/test overlap!"
    assert not (set(val_fns)   & set(test_fns)), "val/test overlap!"

    log.info(f"Split → train: {len(train_fns)}, val: {len(val_fns)}, test: {len(test_fns_list)}")

    fn2fname = dict(zip(df["filename"], df["family_name"]))
    from collections import Counter
    for name, fns in [("train", train_fns), ("val", val_fns)]:
        fc = Counter(fn2fname.get(f,"?") for f in fns)
        log.info(f"  {name}: " + ", ".join(f"{k}={v}" for k,v in sorted(fc.items(), key=lambda x:-x[1])[:6]))

    split = {
        "train": train_fns,
        "val":   val_fns,
        "test":  test_fns_list,
        "metadata": {
            "method": "cluster_val_fulldata",
            "data": DATA,
            "cluster_thresh_jaccard": CLUSTER_THRESH,
            "val_frac": VAL_FRAC,
            "n_clusters": len(clusters),
            "seed": SEED,
        }
    }
    os.makedirs(OUTDIR, exist_ok=True)
    out_path = os.path.join(OUTDIR, "split.json")
    json.dump(split, open(out_path, "w"), indent=2)
    log.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
