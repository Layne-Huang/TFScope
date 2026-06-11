#!/usr/bin/env python
"""Step 3: pure-retrieval diagnostic (NO model, NO training).

Question: for cluster40 OOD *test* TFs, does adding the 6,626 DPAC proteins to
the retrieval donor pool yield a closer/better neighbour PWM than our current
train+val donor pool?

For each test TF we retrieve top-1 (and best-of-K=3) under two donor pools:
  A) REAL          : cluster40 train+val TFs (current RAG donors)
  B) REAL + DPAC   : plus DPAC pseudo-PWMs
and score the RETRIEVED donor's PWM against the target core via oracle align
(+/-10 offset + RC, per-col Pearson r over the trimmed IC>=0.25 core, >=4 pos).

If B does not beat A, DPAC brings no useful neighbours -> no retrain warranted.
"""
import os, sys, json, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from tfscope.models.alignment import align_pwm

SPLIT = "data/processed/splits/cluster40/split.json"
DATA = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
EMB_REAL = "data/processed/tf_dbd_embeddings_aug.npz"
EMB_DPAC = "data/processed/dpac_dbd_embeddings.npz"
PWM_DPAC = "data/processed/dpac_pseudo_pwms.npz"
K = 3
IC_THRESH, MIN_POS, MAX_SHIFT = 0.25, 4, 10


def decode_pwm(b, L):
    return np.frombuffer(b, np.float32).reshape(4, L).copy()

def ic_cols(p):
    p = np.clip(p, 1e-8, 1.0); return 2.0 + (p * np.log2(p)).sum(0)

def trim_core(p):
    inf = np.where(ic_cols(p) >= IC_THRESH)[0]
    if len(inf) == 0: return None
    c = p[:, inf[0]:inf[-1] + 1]
    return c / c.sum(0, keepdims=True)

def score(donor_pwm, target_core):
    dc = trim_core(donor_pwm)
    if dc is None or dc.shape[1] == 0: return -1.0
    _, _, _, r = align_pwm(dc, target_core, max_shift=MAX_SHIFT, consider_revcomp=True)
    return float(r)


def main():
    split = json.load(open(SPLIT))
    donor_set = set(split["train"]) | set(split["val"])
    test_set = split["test"]
    df = pd.read_parquet(DATA)
    fn_pwm = {r.filename: decode_pwm(r.pwm, int(r.motif_length)) for r in df.itertuples()}

    er = np.load(EMB_REAL); ed = np.load(EMB_DPAC); pd_ = np.load(PWM_DPAC)
    real_fns = [fn for fn in donor_set if fn in er.files]
    dpac_fns = list(ed.files)
    print(f"donors: real={len(real_fns)}  dpac={len(dpac_fns)}  test queries={len(test_set)}")

    def norm(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    R = norm(np.stack([er[fn] for fn in real_fns]).astype(np.float32))
    D = norm(np.stack([ed[fn] for fn in dpac_fns]).astype(np.float32))

    res = {"real_top1": [], "aug_top1": [], "real_bestK": [], "aug_bestK": [],
           "dpac_won_top1": 0, "n": 0}
    for fn in test_set:
        if fn not in er.files or fn not in fn_pwm: continue
        core = trim_core(fn_pwm[fn])
        if core is None or core.shape[1] < MIN_POS: continue
        q = er[fn].astype(np.float32); q = q / (np.linalg.norm(q) + 1e-8)
        sr = R @ q; sd = D @ q
        # exclude self if present in real donors
        if fn in real_fns: sr[real_fns.index(fn)] = -2.0

        # --- pool A: real ---
        ra = np.argsort(-sr)[:K]
        sa = [score(fn_pwm[real_fns[i]], core) for i in ra]
        # --- pool B: real + dpac ---
        topd = np.argsort(-sd)[:K]
        # merge candidate (sim, source, key)
        cand = [(sr[i], "real", real_fns[i]) for i in ra] + \
               [(sd[i], "dpac", dpac_fns[i]) for i in topd]
        cand.sort(key=lambda x: -x[0])
        topb = cand[:K]
        sb = [score(fn_pwm[k] if src == "real" else pd_[k], core) for _, src, k in topb]

        res["real_top1"].append(sa[0]); res["aug_top1"].append(sb[0])
        res["real_bestK"].append(max(sa)); res["aug_bestK"].append(max(sb))
        if topb[0][1] == "dpac": res["dpac_won_top1"] += 1
        res["n"] += 1

    def stat(v): v = np.array(v); return f"mean={v.mean():.3f} median={np.median(v):.3f}"
    print(f"\nn={res['n']} test TFs scored (>= {MIN_POS} pos)")
    print(f"  RETRIEVED top-1 oracle-r   REAL : {stat(res['real_top1'])}")
    print(f"  RETRIEVED top-1 oracle-r   +DPAC: {stat(res['aug_top1'])}")
    print(f"  RETRIEVED best-of-{K} r     REAL : {stat(res['real_bestK'])}")
    print(f"  RETRIEVED best-of-{K} r     +DPAC: {stat(res['aug_bestK'])}")
    print(f"  top-1 neighbour was a DPAC donor: {res['dpac_won_top1']}/{res['n']} "
          f"({100*res['dpac_won_top1']/max(res['n'],1):.1f}%)")
    d_top1 = np.mean(res["aug_top1"]) - np.mean(res["real_top1"])
    d_bestk = np.mean(res["aug_bestK"]) - np.mean(res["real_bestK"])
    print(f"  Δ top-1 = {d_top1:+.4f}   Δ best-of-{K} = {d_bestk:+.4f}")
    print("\nVERDICT:", "DPAC ADDS useful neighbours -> retrain worth trying"
          if d_bestk > 0.01 else "DPAC adds ~nothing -> NO retrain")
    json.dump({k: (v if not isinstance(v, list) else None) for k, v in res.items()} |
              {"delta_top1": float(d_top1), "delta_bestK": float(d_bestk),
               "real_top1_mean": float(np.mean(res["real_top1"])),
               "aug_top1_mean": float(np.mean(res["aug_top1"])),
               "real_bestK_mean": float(np.mean(res["real_bestK"])),
               "aug_bestK_mean": float(np.mean(res["aug_bestK"]))},
              open("results/dpac_retrieval_diagnostic.json", "w"), indent=2)


if __name__ == "__main__":
    main()
