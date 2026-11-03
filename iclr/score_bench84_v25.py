#!/usr/bin/env python
"""TFScope vs DeepPBS on the 84-row benchmark, both trained on the frozen unified-v25 split.

This is the first symmetric comparison the project has had: one 40%-identity partition,
both models trained on its train side, both scored on its test side, so neither has seen
anything the other is tested on.

Scored under BOTH target definitions, because the choice moves the numbers more than the
method gap does:

  tfscope_target   the parquet PWM, IC-trimmed to its core (the usual protocol)
  deeppbs_target   Y_pwm[0][pwm_mask[0]] straight from the .npz, NOT re-trimmed --
                   the mask is DeepPBS's own definition of which columns count, and
                   re-trimming would substitute our motif definition for theirs

Everything else is held fixed across methods and targets: the same oracle shift +
reverse-complement registration from `unified_eval.panel_A`, the same gene-balanced
aggregation, and a gene-level paired bootstrap for the delta so the CI accounts for
several structures sharing a gene.

  PYTHONPATH=src python -m iclr.score_bench84_v25
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "."); sys.path.insert(0, "src")
from iclr.unified_eval import panel_A, trimmed_core  # noqa: E402

UNI = "data/processed/tf_pwm_unified_v25.parquet"
TF_PREDS = "results/v25_bench84/tfscope_preds.json"
DP_PREDS = "/data1/leihuang/TFScope_store/deeppbs_v25_bench84_preds.pkl"
NPZ_MAP = "/data1/leihuang/DeepPBS/deeppbsmar24/data/assembly2024"
OUT = "results/v25_bench84/bench84_comparison.json"
N_BOOT = 5000
SEED = 0


def norm(p):
    a = np.asarray(p, dtype=np.float32)
    if a.shape[0] != 4 and a.shape[1] == 4:
        a = a.T
    a = np.clip(a, 0, None)
    s = a.sum(0, keepdims=True)
    return np.divide(a, s, out=np.full_like(a, 0.25), where=s > 0)


def deeppbs_label(npz_path):
    """DeepPBS's own target: strand 0 of Y_pwm, restricted to its own mask."""
    d = np.load(npz_path, allow_pickle=True)
    y, m = d["Y_pwm"], d["pwm_mask"]
    return norm(y[0][m[0]].T)                      # (4, n_valid)


def gene_bootstrap(rows, rng, n_boot=N_BOOT):
    """Paired gene-level bootstrap of mean(a) - mean(b), gene-balanced at each draw."""
    by = {}
    for r in rows:
        by.setdefault(r["gene"], []).append((r["a"], r["b"]))
    genes = sorted(by)
    ga = np.array([np.mean([x for x, _ in by[g]]) for g in genes])
    gb = np.array([np.mean([y for _, y in by[g]]) for g in genes])
    obs = float(ga.mean() - gb.mean())
    idx = rng.randint(0, len(genes), size=(n_boot, len(genes)))
    draws = ga[idx].mean(1) - gb[idx].mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return obs, float(lo), float(hi), float(ga.mean()), float(gb.mean()), len(genes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    rng = np.random.RandomState(SEED)

    uni = pd.read_parquet(UNI); uni["filename"] = uni.filename.astype(str)
    b = uni[uni.is_deeppbs_bench].copy()
    assert len(b) == 84, len(b)

    # npz <-> parquet filename, same rule as build_deeppbs_folds_v25
    from collections import defaultdict
    bykey = defaultdict(list)
    for n in sorted(os.listdir(NPZ_MAP)):
        if n.endswith(".npz"):
            bykey["_".join(n[:-4].split("_")[:2])].append(n)

    def to_npz(fn):
        s = fn[:-4] if fn.endswith(".txt") else fn
        p = s.split("_"); rest = "_".join(p[2:])
        motif = rest.split(".", 1)[1] if "." in rest else rest
        for n in bykey.get("_".join(p[:2]), []):
            if n[:-4].endswith(motif) or n[:-4].endswith(motif + ".jaspar"):
                return n
        for n in bykey.get("_".join(p[:2]), []):
            if motif in n:
                return n
        return None

    b["npz"] = b.filename.map(to_npz)
    assert b.npz.notna().all()

    # dump_predictions stores {family_mode: {filename: {"content": (4,L), "len": int}}}
    tf_raw = json.load(open(TF_PREDS))
    content = lambda blob: {k: np.asarray(v["content"], dtype=np.float32)
                            for k, v in blob.items()}
    tf_pred = content(tf_raw["true"] if "true" in tf_raw else tf_raw)
    dp_pred = pickle.load(open(DP_PREDS, "rb"))

    # free sanity check: with --no-family-id, rolling the label must change nothing
    fam_note = "n/a"
    if "rolled" in tf_raw:
        rolled = content(tf_raw["rolled"])
        d = max(float(np.abs(tf_pred[k] - rolled[k]).max())
                for k in tf_pred if k in rolled and tf_pred[k].shape == rolled[k].shape)
        fam_note = f"max |pred(true family_id) - pred(rolled family_id)| = {d:.3e}"

    results = {"n_rows": len(b), "n_genes": int(b.gene_symbol.nunique()),
               "family_id_invariance": fam_note, "targets": {}}

    for tname in ("tfscope_target", "deeppbs_target"):
        rows = []
        for r in b.itertuples():
            fn = r.filename
            if fn not in tf_pred or r.npz not in dp_pred:
                continue
            if tname == "tfscope_target":
                gt = trimmed_core(np.frombuffer(r.pwm, dtype=np.float32).reshape(4, -1))
            else:
                gt = deeppbs_label(os.path.join(NPZ_MAP, r.npz))
            if gt is None or gt.shape[1] < 4:
                continue
            rows.append({
                "filename": fn, "gene": str(r.gene_symbol),
                "a": panel_A(norm(tf_pred[fn]), gt)["content_r"],
                "b": panel_A(norm(dp_pred[r.npz]), gt)["content_r"],
            })
        rows = [r for r in rows if np.isfinite(r["a"]) and np.isfinite(r["b"])]
        obs, lo, hi, ma, mb, ng = gene_bootstrap(rows, rng)
        results["targets"][tname] = {
            "n_scored": len(rows), "n_genes": ng,
            "tfscope_gene_content_r": ma, "deeppbs_gene_content_r": mb,
            "delta_tfscope_minus_deeppbs": obs, "delta_ci95": [lo, hi],
            "significant": bool(lo > 0 or hi < 0),
            "per_row": rows,
        }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)

    print(f"84-row benchmark | {results['n_genes']} genes | {fam_note}\n")
    print(f"{'target':<16}{'n':>4}{'genes':>7}{'TFScope':>10}{'DeepPBS':>10}"
          f"{'delta':>9}   95% CI            verdict")
    for t, d in results["targets"].items():
        v = "TFScope" if d["delta_ci95"][0] > 0 else \
            "DeepPBS" if d["delta_ci95"][1] < 0 else "tie (CI crosses 0)"
        print(f"{t:<16}{d['n_scored']:>4}{d['n_genes']:>7}"
              f"{d['tfscope_gene_content_r']:>10.4f}{d['deeppbs_gene_content_r']:>10.4f}"
              f"{d['delta_tfscope_minus_deeppbs']:>+9.4f}   "
              f"[{d['delta_ci95'][0]:+.4f}, {d['delta_ci95'][1]:+.4f}]   {v}")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
