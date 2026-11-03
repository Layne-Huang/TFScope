#!/usr/bin/env python
"""Full metric panel for the 84-row benchmark: TFScope vs DeepPBS vs the training-free floor.

Same registration as `score_bench84_v25.py` (oracle shift + reverse-complement via
`unified_eval._aligned_cols`), same gene-balanced aggregation, same gene-level paired
bootstrap. Adds MAE / RMSE / AUROC / top-base accuracy / macro-F1 to the content_r
already reported.

MAE and RMSE are in **DeepPBS's convention** (error summed over the 4 bases inside a
column, then averaged over columns; MAE in [0, 2]), so they can sit next to the numbers
DeepPBS's own trainer prints. `iclr.compare_full_metrics._col_metrics` is the single
definition -- do not reintroduce a per-entry mean, which is exactly this / 4.

AUROC treats every base of every aligned column as one example: label = target > 0.25
(is this base enriched?), score = predicted probability. Computed per row then averaged,
so a row with a degenerate label set drops out rather than dominating a pooled curve.

TFScope's checkpoint is the one selected on VALIDATION content_r (epoch 150), never on
test -- see `ckpt_reselection.json` for why the shipped `ckpt_best` is not used.

  PYTHONPATH=src python -m iclr.bench84_metrics
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

sys.path.insert(0, "."); sys.path.insert(0, "src")
from iclr.compare_full_metrics import _col_metrics      # noqa: E402
from iclr.unified_eval import _aligned_cols, trimmed_core  # noqa: E402

UNI = "data/processed/tf_pwm_unified_v25.parquet"
NPZ_DIR = "/data1/leihuang/DeepPBS/deeppbsmar24/data/assembly2024"
DP_PREDS = "/data1/leihuang/TFScope_store/deeppbs_v25_bench84_preds.pkl"
TF_PREDS = "results/v25_bench84/preds_ckpt_epoch150.json"
LEAK = "/tmp/bench84_leaked.json"
OUT = "results/v25_bench84/bench84_metrics.json"

HIGHER_BETTER = {"pearson_r": True, "auroc": True, "topbase_acc": True, "macroF1": True,
                 "cosine": True, "mae": False, "rmse": False, "jsd_bits": False,
                 "ic_mae": False}
ORDER = ["pearson_r", "auroc", "mae", "rmse", "topbase_acc", "macroF1", "cosine",
         "jsd_bits", "ic_mae"]
N_BOOT = 5000


def norm(a):
    a = np.clip(np.asarray(a, np.float32), 0, None)
    s = a.sum(0, keepdims=True)
    return np.divide(a, s, out=np.full_like(a, 0.25), where=s > 0)


def row_metrics(pred, core):
    aligned, cols = _aligned_cols(norm(pred), core)
    if len(cols) < 2:
        return None
    from sklearn.metrics import f1_score, roc_auc_score
    P, T = aligned[:, cols], core[:, cols]
    m = _col_metrics(P, T)
    lab, sc = m.pop("_auc_label"), m.pop("_auc_score")
    pl, tl = m.pop("_pred_letter"), m.pop("_true_letter")
    try:
        m["auroc"] = float(roc_auc_score(lab, sc)) if len(set(lab)) > 1 else np.nan
    except ValueError:
        m["auroc"] = np.nan
    m["macroF1"] = float(f1_score(tl, pl, average="macro", labels=[0, 1, 2, 3],
                                  zero_division=0))
    return m


def gene_mean(per_row, metric):
    by = defaultdict(list)
    for g, m in per_row:
        if m is not None and np.isfinite(m.get(metric, np.nan)):
            by[g].append(m[metric])
    if not by:
        return np.nan, {}
    gm = {g: float(np.mean(v)) for g, v in by.items()}
    return float(np.mean(list(gm.values()))), gm


def paired_ci(gm_a, gm_b, rng, n=N_BOOT):
    genes = sorted(set(gm_a) & set(gm_b))
    a = np.array([gm_a[g] for g in genes]); b = np.array([gm_b[g] for g in genes])
    idx = rng.randint(0, len(genes), size=(n, len(genes)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), [float(x) for x in np.percentile(d, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    rng = np.random.RandomState(0)

    uni = pd.read_parquet(UNI); uni["filename"] = uni.filename.astype(str)
    uni["pwm_arr"] = uni.pwm.map(lambda b: np.frombuffer(b, np.float32).reshape(4, -1))
    tr = uni[uni.split == "train"]
    b = uni[uni.is_deeppbs_bench].copy()
    leak = set(json.load(open(LEAK)))

    bykey = defaultdict(list)
    for n in sorted(os.listdir(NPZ_DIR)):
        if n.endswith(".npz"):
            bykey["_".join(n[:-4].split("_")[:2])].append(n)

    def to_npz(fn):
        s = fn[:-4] if fn.endswith(".txt") else fn
        p = s.split("_"); rest = "_".join(p[2:])
        mo = rest.split(".", 1)[1] if "." in rest else rest
        for n in bykey.get("_".join(p[:2]), []):
            if n[:-4].endswith(mo) or n[:-4].endswith(mo + ".jaspar"):
                return n
        for n in bykey.get("_".join(p[:2]), []):
            if mo in n:
                return n
        return None

    b["npz"] = b.filename.map(to_npz)
    tf = {k: np.asarray(v["content"], np.float32)
          for k, v in json.load(open(TF_PREDS))["true"].items()}
    dp = pickle.load(open(DP_PREDS, "rb"))

    # family median + nearest-train-DBD floors, same as bench84_floor
    fam = {}
    for fid, g in tr.groupby("family_id"):
        W = max(int(np.median([p.shape[1] for p in g.pwm_arr])), 1)
        acc, cnt = np.full((4, W), 0.25, np.float64), np.zeros(W)
        for p in g.pwm_arr:
            L = min(p.shape[1], W); acc[:, :L] += norm(p[:, :L]); cnt[:L] += 1
        fam[fid] = norm(np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.25))

    tr_seqs, tr_pwms = list(tr.sequence), list(tr.pwm_arr)
    sm = SequenceMatcher(autojunk=False)
    near = {}
    for r in b.itertuples():
        sm.set_seq2(r.sequence); best, bp = -1.0, None
        for s, p in zip(tr_seqs, tr_pwms):
            sm.set_seq1(s)
            if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                continue
            v = sm.ratio()
            if v > best:
                best, bp = v, p
        near[r.filename] = norm(bp)
    rand_idx = rng.randint(0, len(tr_pwms), size=len(b))

    methods = {
        "random_train":  lambda r, i: norm(tr_pwms[rand_idx[i]]),
        "family_median": lambda r, i: fam[int(r.family_id)],
        "nearest_seq":   lambda r, i: near[r.filename],
        "TFScope_ep150": lambda r, i: tf.get(r.filename),
        "DeepPBS_ens5":  lambda r, i: dp.get(r.npz),
    }

    out = {"note": "MAE/RMSE in DeepPBS convention (summed over bases, range [0,2])",
           "tfscope_ckpt": TF_PREDS, "surfaces": {}}

    for sname, drop in (("all84", set()), ("clean76", leak)):
        rows = {k: [] for k in methods}
        for i, r in enumerate(b.itertuples()):
            if r.filename in drop:
                continue
            core = trimmed_core(r.pwm_arr)
            if core is None or core.shape[1] < 4:
                continue
            for k, f in methods.items():
                p = f(r, i)
                rows[k].append((str(r.gene_symbol),
                                None if p is None else row_metrics(p, core)))
        agg, gms = {}, {}
        for k in methods:
            agg[k] = {}; gms[k] = {}
            for met in ORDER:
                v, gm = gene_mean(rows[k], met)
                agg[k][met] = v; gms[k][met] = gm
        def compare(ref, ref_label):
            """TFScope vs `ref`, paired on genes. Verdict names the winner or 'tie'."""
            res = {}
            for met in ORDER:
                d, ci = paired_ci(gms["TFScope_ep150"][met], gms[ref][met], rng)
                better = (ci[0] > 0) if HIGHER_BETTER[met] else (ci[1] < 0)
                worse = (ci[1] < 0) if HIGHER_BETTER[met] else (ci[0] > 0)
                res[met] = {"delta_tf_minus_ref": d, "ci95": ci,
                            "verdict": "TFScope" if better else ref_label if worse else "tie"}
            return res

        n = len([1 for g, m in rows["TFScope_ep150"] if m is not None])
        out["surfaces"][sname] = {
            "n_rows": n, "aggregate": agg,
            "tf_vs_dp": compare("DeepPBS_ens5", "DeepPBS"),
            # the comparison that decides whether the model earns its keep at all
            "tf_vs_nearest": compare("nearest_seq", "nearest_seq"),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    for sname, s in out["surfaces"].items():
        print(f"\n=== {sname} (n={s['n_rows']}) — MAE/RMSE in DeepPBS convention ===")
        hdr = f"{'metric':<13}" + "".join(f"{k:>15}" for k in methods) + \
              f"{'Δ TF-DP':>10}   95% CI              verdict"
        print(hdr)
        for met in ORDER:
            arrow = "↑" if HIGHER_BETTER[met] else "↓"
            line = f"{met + ' ' + arrow:<13}"
            for k in methods:
                line += f"{s['aggregate'][k][met]:>15.4f}"
            d = s["tf_vs_dp"][met]
            nn = s["tf_vs_nearest"][met]
            line += (f"{d['delta_tf_minus_ref']:>+10.4f}   "
                     f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]   {d['verdict']:<10}"
                     f"{nn['delta_tf_minus_ref']:>+9.4f} {nn['verdict']}")
            print(line)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
