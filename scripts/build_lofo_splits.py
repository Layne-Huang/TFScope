#!/usr/bin/env python
"""Build leave-family-out (LFO) splits against the current trimmed aug parquet.

For each TF family F:
  test  = all PWMs whose family_name == F  (entirely held out)
  pool  = all PWMs whose family_name != F
  val   = CD-HIT-40% cluster-based 15% slice of the pool (family-stratified
          across the *remaining* families, so val is sequence-dissimilar to train)
  train = pool - val

This tests generalization to an unseen TF family. The model's semantic family
embedding (precomputed, 2304-d) still places the held-out family meaningfully,
so there is no cold-start on the family token — the test is pure
sequence + semantic-prior → PWM transfer to a novel family.

Output: data/processed/splits/lofo_v2/<Family>.json  (one per family)
"""
import argparse, json, os, subprocess, time, logging
from collections import defaultdict
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATA   = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
OUTDIR = "data/processed/splits/lofo_v2"
SEED   = 42
VAL_FRAC = 0.15


def run_cdhit(fasta, prefix, identity=0.4, threads=8):
    n = 2 if identity < 0.5 else 3
    cmd = ["cd-hit", "-i", fasta, "-o", prefix, "-c", str(identity),
           "-n", str(n), "-M", "4000", "-T", str(threads), "-d", "0", "-g", "1"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return prefix + ".clstr"


def parse_clstr(path):
    assign, cid = {}, -1
    for line in open(path):
        line = line.strip()
        if line.startswith(">Cluster"):
            cid += 1
        elif ">" in line:
            uid = line.split(">")[1].split("...")[0].strip()
            assign[uid] = cid
    return assign


def cluster_val(pool, outdir, fam_tag, identity=0.4, seed=SEED, threads=8):
    """Return set of uniprot_ids assigned to val (CD-HIT cluster-based, family-strat)."""
    prot = pool.drop_duplicates("uniprot_id")[["uniprot_id","sequence","family_id"]].reset_index(drop=True)
    fasta = os.path.join(outdir, f"_tmp_{fam_tag}.fasta")
    with open(fasta, "w") as f:
        for _, r in prot.iterrows():
            f.write(f">{r['uniprot_id']}\n{r['sequence']}\n")
    clstr = run_cdhit(fasta, os.path.join(outdir, f"_tmp_{fam_tag}"), identity, threads)
    uid2c = parse_clstr(clstr)
    prot["cluster_id"] = prot["uniprot_id"].map(uid2c)
    # any unassigned → unique clusters
    mx = int(prot["cluster_id"].max()) + 1 if prot["cluster_id"].notna().any() else 0
    for i, idx in enumerate(prot[prot["cluster_id"].isna()].index):
        prot.at[idx, "cluster_id"] = mx + i
    prot["cluster_id"] = prot["cluster_id"].astype(int)

    rng = np.random.RandomState(seed)
    fam_clusters = defaultdict(dict)
    for _, r in prot.iterrows():
        fam_clusters[r["family_id"]].setdefault(r["cluster_id"], []).append(r["uniprot_id"])

    val_uids = set()
    for fid, cl in fam_clusters.items():
        items = list(cl.items()); rng.shuffle(items)
        n_val = max(1, round(len(items) * VAL_FRAC))
        for i, (cid, uids) in enumerate(items):
            if i < n_val: val_uids.update(uids)
    # cleanup temp files
    for ext in ["", ".clstr"]:
        p = os.path.join(outdir, f"_tmp_{fam_tag}{ext}")
        if os.path.exists(p): os.remove(p)
    if os.path.exists(fasta): os.remove(fasta)
    return val_uids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--identity", type=float, default=0.4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--families", nargs="*", default=None,
                    help="subset of families to build; default all")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_parquet(args.data)
    families = args.families or sorted(df["family_name"].unique())
    log.info(f"Data: {len(df)} rows | families: {families}")

    summary = []
    for fam in families:
        test = df[df["family_name"] == fam]
        pool = df[df["family_name"] != fam].copy()
        if len(test) == 0:
            log.warning(f"[{fam}] no test rows; skip"); continue

        t0 = time.time()
        val_uids = cluster_val(pool, args.outdir, fam.replace("/","_"),
                               identity=args.identity, threads=args.threads)
        val  = pool[pool["uniprot_id"].isin(val_uids)]
        train = pool[~pool["uniprot_id"].isin(val_uids)]

        train_fns = sorted(train["filename"].tolist())
        val_fns   = sorted(val["filename"].tolist())
        test_fns  = sorted(test["filename"].tolist())

        assert not (set(train_fns) & set(val_fns))
        assert not (set(train_fns) & set(test_fns))
        assert not (set(val_fns)   & set(test_fns))

        split = {
            "train": train_fns, "val": val_fns, "test": test_fns,
            "metadata": {
                "method": "leave_family_out_cdhit40",
                "data": args.data,
                "held_out_family": fam,
                "identity_threshold": args.identity,
                "n_train_pwms": len(train_fns),
                "n_val_pwms":   len(val_fns),
                "n_test_pwms":  len(test_fns),
                "n_train_prot": train["uniprot_id"].nunique(),
                "n_val_prot":   val["uniprot_id"].nunique(),
                "n_test_prot":  test["uniprot_id"].nunique(),
                "seed": SEED,
            }
        }
        out_path = os.path.join(args.outdir, f"{fam.replace('/','_')}.json")
        json.dump(split, open(out_path, "w"), indent=2)
        log.info(f"[{fam:<16}] train={len(train_fns):>4} val={len(val_fns):>4} "
                 f"test={len(test_fns):>4} (test_prot={test['uniprot_id'].nunique()}) "
                 f"{time.time()-t0:.1f}s")
        summary.append((fam, len(train_fns), len(val_fns), len(test_fns),
                        test["uniprot_id"].nunique()))

    print("\n" + "="*70)
    print(f"{'Family':<18} {'train':>7} {'val':>6} {'test':>6} {'test_prot':>10}")
    print("-"*70)
    for fam, tr, v, te, tp in summary:
        print(f"{fam:<18} {tr:>7} {v:>6} {te:>6} {tp:>10}")
    print("="*70)
    log.info(f"Saved {len(summary)} LFO splits → {args.outdir}/")


if __name__ == "__main__":
    main()
