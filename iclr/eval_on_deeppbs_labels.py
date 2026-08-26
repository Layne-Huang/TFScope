#!/usr/bin/env python
"""Score every method against DeepPBS's OWN masked labels, on DeepPBS's home turf.

Why this exists
---------------
The `deeppbs20` surface scores everyone against TFScope's v23 HOCOMOCO targets. That
is fair in protocol (identical genes, identical oracle shift + RC registration, scored
only on overlapping columns) but it evaluates DeepPBS on a target it was never trained
against: its own labels have a median of 9 valid columns, ours a median of 12, and its
raw output spans the whole co-crystal DNA (median 13, up to 25). So `coverage` is
inflated for DeepPBS almost by definition.

This script removes that asymmetry by using DeepPBS's own ground truth:

    target = Y_pwm[strand 0][pwm_mask[0]]      (4, n_valid),  n_valid = 5..15

taken straight from the structure's .npz. Verified beforehand to be the same motif as
our v23 target for the same gene (median overlap r 0.967; >0.9 for 15/20 genes), so
this is a re-windowing of the same biology, not a different label.

Everything else is held fixed: same oracle shift + reverse-complement registration,
same metric panel, same rungs. One structure per gene, so n = 20 and gene-balanced
equals row-level.

Deliberately NOT IC-trimmed: the mask is DeepPBS's own definition of which columns
count, and re-trimming it would substitute our motif definition for theirs, which is
the very thing this analysis is meant to avoid.

  PYTHONPATH=src python -m iclr.eval_on_deeppbs_labels --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from iclr.baselines import _decode_pwm, _nearest_pwm                      # noqa: E402
from iclr.compare_full_metrics import _col_metrics                        # noqa: E402
from iclr.ladder_full_metrics import (DEEPPBS_PKL, ORDER, SEED42, ENS,    # noqa: E402
                                      _group_mean, _kl_bits, add_taxonomy)
from iclr.score_v24_ensemble import predict_ensemble                      # noqa: E402
from iclr.unified_eval import _aligned_cols, panel_B                      # noqa: E402

DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
NPZ_DIR = "/data1/leihuang/DeepPBS/deeppbsmar24/data/assembly2024"
FOLDS = "/data1/leihuang/DeepPBS/deeppbsmar24/run/iclr_folds_pdbdisjoint"
OUT = "results/baseline_ladder/eval_on_deeppbs_labels.json"

RUNGS = ["random_uniform", "random_train_pwm", "B0_global", "B0_family",
         "B1_nearest_pwm", "deeppbs", "v24_seed42", "v24_ens5"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gene_of(struct):
    m = re.match(r"^[^_]+_[^_]+_([A-Za-z0-9\-]+)_", struct)
    return m.group(1).upper() if m else None


def deeppbs_target(struct):
    """DeepPBS's own label for this structure: (4, n_valid), column-normalized."""
    z = np.load(os.path.join(NPZ_DIR, struct), allow_pickle=True)
    mask = z["pwm_mask"][0].astype(bool)
    Y = z["Y_pwm"][0][mask].T.astype(np.float64)
    return (Y / np.clip(Y.sum(0, keepdims=True), 1e-9, None)).astype(np.float32)


def score(preds, lens, targets):
    """Metric panel per structure, then plain mean (one structure per gene)."""
    from sklearn.metrics import f1_score, roc_auc_score
    rows = []
    for k, core in targets.items():
        pred = preds.get(k)
        if pred is None:
            continue
        aligned, cols = _aligned_cols(pred, core)
        if len(cols) < 2:
            continue
        P, T = aligned[:, cols], core[:, cols]
        m = _col_metrics(P, T)
        m["kl_bits"] = _kl_bits(P, T)
        # DeepPBS's own convention (deeppbs/nn/metrics/metrics.py:89): sum the absolute
        # error over the four bases, THEN average over columns -- range [0, 2]. Our
        # `_col_metrics` averages over bases and columns together, range [0, 0.5], so
        # ours is exactly DeepPBS's / 4 (and RMSE exactly / 2; verified to machine
        # precision). A constant factor changes no ranking, but reporting our number
        # under the bare name "MAE" would look 4x better than DeepPBS's published one.
        m["mae_dpbs"] = float(np.mean(np.sum(np.abs(P - T), axis=0)))
        m["rmse_dpbs"] = float(np.sqrt(np.mean(np.sum((P - T) ** 2, axis=0))))
        lab = m.pop("_auc_label"); sc = m.pop("_auc_score")
        pl = m.pop("_pred_letter"); tl = m.pop("_true_letter")
        try:
            m["auroc"] = float(roc_auc_score(lab, sc)) if len(set(lab)) > 1 else np.nan
        except ValueError:
            m["auroc"] = np.nan
        m["macroF1"] = float(f1_score(tl, pl, average="macro", labels=[0, 1, 2, 3],
                                      zero_division=0))
        b = panel_B(pred, core, lens.get(k))
        m.update(covR=b["covR"], coverage=b["coverage"],
                 len_mae=float(b["len_mae"]), len_bias=float(b["len_bias"]))
        m["key"] = k
        rows.append(m)
    return rows


def agg(rows, n_boot=10000, seed=0):
    keys = ORDER + ["mae_dpbs", "rmse_dpbs", "covR", "coverage", "len_mae", "len_bias"]
    out = {"n": len(rows)}
    rng = np.random.RandomState(seed)
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        boot = [np.nanmean(rng.choice(v, v.size, replace=True)) for _ in range(n_boot)]
        out[k] = round(float(np.nanmean(v)), 4)
        out[k + "_ci95"] = [round(float(np.percentile(boot, 2.5)), 4),
                            round(float(np.percentile(boot, 97.5)), 4)]
    return out


def paired(a_rows, b_rows, metric, up, n_boot=10000, seed=0):
    """b - a on shared structures, oriented so positive means `b` is better."""
    A = {r["key"]: r[metric] for r in a_rows}
    B = {r["key"]: r[metric] for r in b_rows}
    ks = sorted(set(A) & set(B))
    d = np.array([B[k] - A[k] for k in ks], float) * (1.0 if up else -1.0)
    d = d[~np.isnan(d)]
    rng = np.random.RandomState(seed)
    boot = [np.mean(rng.choice(d, d.size, replace=True)) for _ in range(n_boot)]
    return (round(float(np.mean(d)), 4),
            round(float(np.percentile(boot, 2.5)), 4),
            round(float(np.percentile(boot, 97.5)), 4), int(d.size))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    structs = [l.strip() for l in open(f"{FOLDS}/test20.txt") if l.strip()]
    targets = {s: deeppbs_target(s) for s in structs}
    g_of = {s: gene_of(s) for s in structs}
    tl = np.array([t.shape[1] for t in targets.values()])
    log(f"{len(structs)} structures / {len(set(g_of.values()))} genes | "
        f"DeepPBS label length min {tl.min()} median {int(np.median(tl))} max {tl.max()}")

    df = add_taxonomy(pd.read_parquet(DATA)); df["filename"] = df.filename.astype(str)
    sp = json.load(open(SPLIT))
    tr = df[df.filename.isin(set(sp["train"]))].copy()
    tr["pwm_arr"] = tr.pwm.map(_decode_pwm)

    # one representative v23 test row per gene -> one TFScope prediction per structure
    te = df[df.filename.isin(set(sp["test"]))]
    rep = {}
    for s in structs:
        sub = te[te.g == g_of[s]]
        if sub.empty:
            log(f"  WARNING no v23 test row for {g_of[s]}; skipping {s}"); continue
        rep[s] = sub.iloc[0].filename
    fn2s = {v: k for k, v in rep.items()}

    # trained models
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"test": sorted(rep.values())}, tmp); tmp.close()
    model_out = {}
    try:
        for tag, ck in [("v24_seed42", [SEED42]), ("v24_ens5", ENS)]:
            p, l = predict_ensemble(ck, DATA, tmp.name, a.device)
            model_out[tag] = ({fn2s[f]: v for f, v in p.items() if f in fn2s},
                              {fn2s[f]: v for f, v in l.items() if f in fn2s})
            log(f"  {tag}: {len(model_out[tag][0])} predictions")
    finally:
        os.unlink(tmp.name)

    # DeepPBS's own raw output
    dp = {g.upper(): np.asarray(v, np.float32)
          for g, v in pickle.load(open(DEEPPBS_PKL, "rb")).items()}
    dp_pred = {s: dp[g_of[s]] for s in structs if g_of[s] in dp}

    # training-free rungs
    global_prior = _group_mean(list(tr.pwm_arr))
    fam_prior = {f: _group_mean(list(g.pwm_arr)) for f, g in tr.groupby("fam_lofo")}
    fam_of = {s: df[df.g == g_of[s]].fam_lofo.iloc[0] for s in structs}
    tr_uniq = tr.drop_duplicates("sequence")[["sequence", "pwm_arr"]].reset_index(drop=True)
    seq_of = {s: te[te.g == g_of[s]].sequence.iloc[0] for s in structs if not te[te.g == g_of[s]].empty}
    rng = np.random.RandomState(0); tr_pwms = list(tr.pwm_arr)
    # the pre-registered baseline length here is the median DeepPBS label length, which
    # never reads any individual test target
    blen = {s: int(np.median(tl)) for s in structs}

    bank = {}
    bank["random_uniform"] = ({s: np.full((4, blen[s]), 0.25, np.float32) for s in structs}, blen)
    bank["B0_global"] = ({s: global_prior for s in structs}, blen)
    bank["B0_family"] = ({s: fam_prior.get(fam_of[s], global_prior) for s in structs}, blen)
    bank["B1_nearest_pwm"] = ({s: _nearest_pwm(seq_of[s], tr_uniq) for s in seq_of}, blen)
    bank["deeppbs"] = (dp_pred, {s: v.shape[1] for s, v in dp_pred.items()})
    for tag in ("v24_seed42", "v24_ens5"):
        bank[tag] = model_out[tag]

    results, per_rung_rows = {}, {}
    for tag in RUNGS:
        if tag == "random_train_pwm":
            # Average each STRUCTURE across the draws first, then bootstrap over
            # structures. Averaging the draws' summary statistics instead (and taking
            # one draw's CI) is not self-consistent: it put the point estimate outside
            # its own interval and produced a negative error bar.
            acc = {}
            for d in range(20):
                pr = {s: tr_pwms[rng.randint(len(tr_pwms))] for s in structs}
                for r in score(pr, blen, targets):
                    acc.setdefault(r["key"], []).append(r)
            merged = []
            for key, rs in acc.items():
                m = {k: float(np.nanmean([r[k] for r in rs]))
                     for k in rs[0] if k != "key"}
                m["key"] = key
                merged.append(m)
            per_rung_rows[tag] = merged
            results[tag] = agg(merged)
            results[tag]["n_draws"] = 20
            continue
        p, l = bank[tag]
        rows = score(p, l, targets)
        per_rung_rows[tag] = rows
        results[tag] = agg(rows)

    # paired comparisons against DeepPBS on its own labels
    cmp_rows = []
    cmp_metrics = ORDER + ["mae_dpbs", "rmse_dpbs", "covR", "coverage"]
    direction = {k: (k in ("pearson_r", "cosine", "topbase_acc", "auroc", "macroF1",
                           "covR", "coverage")) for k in cmp_metrics}
    for alt in ("v24_seed42", "v24_ens5"):
        for m in cmp_metrics:
            v, lo, hi, n = paired(per_rung_rows["deeppbs"], per_rung_rows[alt],
                                  m, direction[m])
            cmp_rows.append({"comparison": f"{alt} - deeppbs", "metric": m,
                             "direction": "higher_better" if direction[m] else "lower_better",
                             "delta_positive_is_tfscope_better": v,
                             "ci95_lo": lo, "ci95_hi": hi, "n": n,
                             "significant": bool(not (lo < 0 < hi))})

    payload = {"note": "every method scored against DeepPBS's own masked labels "
                       "(Y_pwm[pwm_mask]) on the 20 co-crystal test structures",
               "target": "Y_pwm[strand0][pwm_mask[0]], column-normalized, NOT IC-trimmed",
               "metric_conventions": {
                   "mae": "mean |p-t| over bases x columns, range [0, 0.5]",
                   "mae_dpbs": "DeepPBS convention: mean over columns of the per-column "
                               "sum of |p-t| over the 4 bases, range [0, 2]; == 4 x mae",
                   "rmse": "sqrt(mean (p-t)^2 over bases x columns)",
                   "rmse_dpbs": "DeepPBS convention (brier_multi with root): "
                                "sqrt(mean over columns of the per-column sum of "
                                "(p-t)^2); == 2 x rmse"},
               "label_length": {"min": int(tl.min()), "median": int(np.median(tl)),
                                "max": int(tl.max())},
               "n_structures": len(structs),
               "ladder": results, "paired_vs_deeppbs": cmp_rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(payload, open(a.out, "w"), indent=1)
    pd.DataFrame(cmp_rows).to_csv(a.out.replace(".json", "_paired.csv"), index=False)

    print(f"\n=== scored against DeepPBS's OWN labels ({len(structs)} structures) ===")
    hdr = f"{'rung':<18}" + "".join(f"{k:>12}" for k in ORDER)
    print(hdr); print("-" * len(hdr))
    for t in RUNGS:
        e = results[t]
        print(f"{t:<18}" + "".join(f"{e[k]:>12.4f}" for k in ORDER))
    print(f"\n{'rung':<18}{'covR':>10}{'coverage':>10}{'len_mae':>10}")
    for t in RUNGS:
        e = results[t]
        print(f"{t:<18}{e['covR']:>10.4f}{e['coverage']:>10.4f}{e['len_mae']:>10.2f}")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
