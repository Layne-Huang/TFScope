#!/usr/bin/env python
"""Where is the floor on the 84-row benchmark?

"TFScope loses to DeepPBS" is only half an answer. A method is worth something if it
beats what you get for free, so this puts the same 84 rows through the training-free
baselines under the identical protocol (`unified_eval.panel_A` content_r, oracle shift +
reverse-complement registration, gene-balanced).

  uniform          0.25 everywhere -- the true floor
  random_train     a random train PWM, averaged over 20 draws so one lucky draw cannot
                   set the bar
  family_median    the left-anchored mean PWM of the row's family, from train only.
                   Reported for context, NOT as a baseline TFScope must clear: the model
                   was trained with --no-family-id and never sees this label
  nearest_seq      the train PWM whose DBD is most similar to the test DBD. The one that
                   matters -- it is what you would do with no model at all, and the split
                   caps it at 40% identity by construction

  PYTHONPATH=src python -m iclr.bench84_floor
"""
from __future__ import annotations

import json
import os
import sys
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

sys.path.insert(0, "."); sys.path.insert(0, "src")
from iclr.unified_eval import panel_A, trimmed_core  # noqa: E402

UNI = "data/processed/tf_pwm_unified_v25.parquet"
CMP = "results/v25_bench84/bench84_comparison.json"
OUT = "results/v25_bench84/bench84_floor.json"
N_DRAWS = 20
SEED = 0


def decode(b):
    return np.frombuffer(b, dtype=np.float32).reshape(4, -1)


def norm(a):
    a = np.clip(np.asarray(a, np.float32), 0, None)
    s = a.sum(0, keepdims=True)
    return np.divide(a, s, out=np.full_like(a, 0.25), where=s > 0)


def gene_mean(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r["gene"], []).append(r[key])
    return float(np.mean([np.mean(v) for v in by.values()]))


def main():
    rng = np.random.RandomState(SEED)
    uni = pd.read_parquet(UNI); uni["filename"] = uni.filename.astype(str)
    uni["pwm_arr"] = uni.pwm.map(decode)
    tr = uni[uni.split == "train"]
    b = uni[uni.is_deeppbs_bench].copy()

    tgt, meta = {}, {}
    for r in b.itertuples():
        core = trimmed_core(r.pwm_arr)
        if core is not None and core.shape[1] >= 4:
            tgt[r.filename] = core
            meta[r.filename] = (str(r.gene_symbol), int(r.family_id), r.sequence)
    print(f"{len(tgt)} rows / {len({v[0] for v in meta.values()})} genes\n")

    # family median (left-anchored mean per family, train only)
    fam = {}
    for fid, g in tr.groupby("family_id"):
        W = max(int(np.median([p.shape[1] for p in g.pwm_arr])), 1)
        acc, n = np.full((4, W), 0.25, np.float64), np.zeros(W)
        for p in g.pwm_arr:
            L = min(p.shape[1], W)
            acc[:, :L] += norm(p[:, :L]); n[:L] += 1
        fam[fid] = norm(np.where(n > 0, acc / np.maximum(n, 1), 0.25))

    # nearest train DBD by sequence similarity
    tr_seqs, tr_pwms = list(tr.sequence), list(tr.pwm_arr)
    nearest = {}
    sm = SequenceMatcher(autojunk=False)
    for i, (fn, (_, _, seq)) in enumerate(meta.items()):
        sm.set_seq2(seq)
        best, bp = -1.0, None
        for s, p in zip(tr_seqs, tr_pwms):
            sm.set_seq1(s)
            if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                continue
            r = sm.ratio()
            if r > best:
                best, bp = r, p
        nearest[fn] = (norm(bp), best)
        if (i + 1) % 20 == 0:
            print(f"  nearest-neighbour {i+1}/{len(meta)}")

    preds = {
        "uniform": {fn: np.full((4, 8), 0.25, np.float32) for fn in tgt},
        "family_median": {fn: fam[meta[fn][1]] for fn in tgt},
        "nearest_seq": {fn: nearest[fn][0] for fn in tgt},
    }

    res = {}
    for name, pr in preds.items():
        rows = [{"gene": meta[fn][0], "r": panel_A(pr[fn], tgt[fn])["content_r"]}
                for fn in tgt]
        rows = [r for r in rows if np.isfinite(r["r"])]
        res[name] = {"gene_content_r": gene_mean(rows, "r"), "n": len(rows)}

    # random train PWM: average over draws so a lucky draw cannot set the bar
    draws = []
    for _ in range(N_DRAWS):
        pick = rng.randint(0, len(tr_pwms), size=len(tgt))
        rows = [{"gene": meta[fn][0],
                 "r": panel_A(norm(tr_pwms[pick[i]]), tgt[fn])["content_r"]}
                for i, fn in enumerate(tgt)]
        rows = [r for r in rows if np.isfinite(r["r"])]
        draws.append(gene_mean(rows, "r"))
    res["random_train"] = {"gene_content_r": float(np.mean(draws)),
                           "sd_over_draws": float(np.std(draws)), "n_draws": N_DRAWS}

    cmp_ = json.load(open(CMP))["targets"]["tfscope_target"]
    res["TFScope_v25_nofam"] = {"gene_content_r": cmp_["tfscope_gene_content_r"]}
    res["DeepPBS_v25_ens5"] = {"gene_content_r": cmp_["deeppbs_gene_content_r"]}
    res["_nearest_seq_identity"] = {
        "median": float(np.median([v[1] for v in nearest.values()])),
        "max": float(max(v[1] for v in nearest.values()))}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    order = ["uniform", "random_train", "family_median", "nearest_seq",
             "TFScope_v25_nofam", "DeepPBS_v25_ens5"]
    print(f"\n{'method':<22}{'gene content_r':>15}")
    for k in order:
        print(f"{k:<22}{res[k]['gene_content_r']:>15.4f}")
    nn = res["_nearest_seq_identity"]
    print(f"\nnearest-train DBD similarity: median {nn['median']:.3f}, max {nn['max']:.3f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
