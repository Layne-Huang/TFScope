#!/usr/bin/env python
"""Precompute alignment-augmented retrieval inputs for the alignment-teacher RAG.

For every TF in the dataset, retrieve its top-K LSO neighbours and produce:
  - seed_aligned : (K,4,L)  each neighbour aligned to the v10 SEED PWM
                            (deployable — uses only inference-available info)
  - oracle_aligned : (K,4,L) each neighbour aligned to the TARGET PWM
                            (TEACHER — training-only supervision)
  - oracle_trust : (K,)     per-neighbour per-column Pearson r to target, in [-1,1]
                            (TEACHER — what the trust/selection head should predict)
  - sims : (K,)             retrieval cosine similarities
  - n_valid : int           number of valid neighbours

The model consumes `seed_aligned` + `sims` at inference; `oracle_aligned` and
`oracle_trust` supervise the alignment/selection heads during training only.

Output: data/processed/aligned_retrieval_K{K}.npz  (keyed arrays by filename)
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from tfscope.models.alignment import align_pwm

MAX_L = 20


def load_pwm(blob):
    if isinstance(blob, bytes):
        return np.frombuffer(blob, dtype=np.float32).reshape(4, -1)
    return None


def pad(pwm, L=MAX_L):
    out = np.full((4, L), 0.25, dtype=np.float32)
    Lc = min(pwm.shape[1], L)
    out[:, :Lc] = pwm[:, :Lc]
    m = np.zeros(L, dtype=np.float32); m[:Lc] = 1.0
    return out, m


def percol_r(a, b):
    L = min(a.shape[1], b.shape[1])
    rs = []
    for j in range(L):
        if a[:, j].std() < 1e-8 or b[:, j].std() < 1e-8:
            continue
        rs.append(np.corrcoef(a[:, j], b[:, j])[0, 1])
    return float(np.mean(rs)) if rs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default="data/processed/tf_pwm_aug_dbd.parquet")
    ap.add_argument("--index", default="data/processed/tf_nn_index_lso.json")
    ap.add_argument("--seeds", default="data/processed/v10_seed_all.npz")
    ap.add_argument("--split", default="data/processed/splits/deeppbs_only/benchmark_no_val.json")
    ap.add_argument("--k",     type=int, default=8)
    ap.add_argument("--max-shift", type=int, default=6)
    ap.add_argument("--out",   default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"data/processed/aligned_retrieval_K{args.k}.npz"

    df = pd.read_parquet(args.data)
    fn2pwm = {r["filename"]: load_pwm(r["pwm"]) for _, r in df.iterrows()}
    fn2pwm = {k: v for k, v in fn2pwm.items() if v is not None}

    index = json.load(open(args.index))

    seeds = np.load(args.seeds, allow_pickle=True)
    seed_fns = list(seeds["filenames"]); seed_pwm = seeds["pwms"]; seed_mask = seeds["masks"]
    fn2seed = {fn: (seed_pwm[i], seed_mask[i]) for i, fn in enumerate(seed_fns)}

    with open(args.split) as f:
        split = json.load(f)
    all_query_fns = split["train"] + split["val"] + split["test"]

    K = args.k
    out = {}
    n_done = 0
    for fn in all_query_fns:
        if fn not in fn2pwm:
            continue
        target = fn2pwm[fn]
        seed_p = fn2seed[fn][0] if fn in fn2seed else None
        sL = int(fn2seed[fn][1].sum()) if fn in fn2seed else target.shape[1]
        seed_ref = seed_p[:, :sL] if seed_p is not None else target  # fallback

        cands = index.get(fn, [])[:K]
        seed_al  = np.full((K, 4, MAX_L), 0.25, dtype=np.float32)
        oracle_al = np.full((K, 4, MAX_L), 0.25, dtype=np.float32)
        trust = np.zeros(K, dtype=np.float32)
        sims = np.zeros(K, dtype=np.float32)
        nv = 0
        for ki, c in enumerate(cands):
            nn = c["nn_filename"]
            if nn not in fn2pwm:
                continue
            nb = fn2pwm[nn]
            # seed-aligned (deployable): align neighbour to seed PWM
            sa, _, _, _ = align_pwm(nb, seed_ref, max_shift=args.max_shift)
            sa_p, _ = pad(sa)
            seed_al[ki] = sa_p
            # oracle-aligned (teacher): align neighbour to target PWM
            oa, _, _, _ = align_pwm(nb, target, max_shift=args.max_shift)
            oa_p, _ = pad(oa)
            oracle_al[ki] = oa_p
            trust[ki] = percol_r(oa, target)            # oracle quality of this neighbour
            sims[ki]  = float(c.get("cos_sim", 0.0))
            nv += 1

        out[fn + "::seed"]   = seed_al
        out[fn + "::oracle"] = oracle_al
        out[fn + "::trust"]  = trust
        out[fn + "::sims"]   = sims
        out[fn + "::nv"]     = np.array([nv], dtype=np.int32)
        n_done += 1
        if n_done % 200 == 0:
            print(f"  {n_done} TFs done", flush=True)

    np.savez_compressed(args.out, **out)
    print(f"Wrote {args.out}  ({n_done} TFs, K={K})", flush=True)

    # quick sanity on test split: mean per-column r of seed-aligned vs oracle top-1
    test = split["test"]
    s_top1, o_top1 = [], []
    for fn in test:
        if fn + "::seed" not in out or fn not in fn2pwm:
            continue
        t = fn2pwm[fn]
        s_top1.append(percol_r(out[fn + "::seed"][0][:, :t.shape[1]], t))
        o_top1.append(out[fn + "::trust"][0])
    print(f"Test seed-aligned top-1 r:   {np.nanmean(s_top1):.4f}  (expect ~0.48)")
    print(f"Test oracle-aligned top-1 r: {np.nanmean(o_top1):.4f}  (expect ~0.84)")


if __name__ == "__main__":
    main()
