"""Gate-swap + family_id shortcut + family-residual, from saved v24 predictions.

Consumes results/.../v24_predictions.json (true & rolled family_id) plus the
frozen exact-family prior. No model inference. Produces:
  1. 2x2 gate-swap: {v24,B0} content x {v24,B0} length, identical registration.
  2. family_id shortcut: how much v24's prediction changes when family_id is
     corrupted (rolled), and whether its covR drops.
  3. family-residual: does v24's deviation-from-family-mean match the true
     deviation (i.e. does v24 add within-family specificity over the prior)?
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "src")
from iclr.baselines import _decode_pwm
from iclr.unified_eval import (trimmed_core, panel_B, _aligned_cols, _overlap_r,
                               train_length_policy, baseline_pred_len)
from tfscope.models.alignment import align_pwm, revcomp_pwm_np

DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
ROOT = "results/iclr_phase1_apples_to_apples"


def _group_mean(pwms):
    W = int(np.median([p.shape[1] for p in pwms])) or 1
    acc = np.zeros((4, W)); n = np.zeros(W)
    for p in pwms:
        L = min(p.shape[1], W); acc[:, :L] += p[:, :L] / p[:, :L].sum(0, keepdims=True).clip(1e-8); n[:L] += 1
    avg = np.where(n > 0, acc / np.maximum(n, 1), 0.25)
    return (avg / avg.sum(0, keepdims=True).clip(1e-8)).astype(np.float32)


def _gene_mean(rows, key):
    by = {}
    for r in rows: by.setdefault(r["gene"], []).append(r[key])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")


def main():
    preds = json.load(open(f"{ROOT}/v24_predictions.json"))
    v_true = {k: (np.array(v["content"], np.float32), v["len"]) for k, v in preds["true"].items()}
    v_roll = {k: (np.array(v["content"], np.float32), v["len"]) for k, v in preds["rolled"].items()}

    df = pd.read_parquet(DATA); df["filename"] = df.filename.astype(str)
    sp = json.load(open(SPLIT))
    tr = df[df.filename.isin(set(sp["train"]))].copy(); te = df[df.filename.isin(set(sp["test"]))].copy()
    tr["pwm_arr"] = tr.pwm.map(_decode_pwm); te["pwm_arr"] = te.pwm.map(_decode_pwm)
    policy = train_length_policy(tr)
    exact = {k: _group_mean(list(g["pwm_arr"])) for k, g in tr.groupby("family_name")}
    exact_n = {k: len(g) for k, g in tr.groupby("family_name")}
    global_prior = _group_mean(list(tr["pwm_arr"]))

    meta = {r.filename: {"gene": r.gene_symbol, "family_id": int(r.family_id),
                         "family": r.family_name, "n_chains": int(getattr(r, "n_chains", 1))}
            for r in te.itertuples()}
    cores = {r.filename: trimmed_core(r.pwm_arr) for r in te.itertuples()}

    def b0_content(fn):
        fam = meta[fn]["family"]
        return exact[fam] if (fam in exact and exact_n.get(fam, 0) > 0) else global_prior

    # ---- 1. 2x2 gate-swap ------------------------------------------------------
    combos = {
        "v24content_v24len":  ("deployable (=v24 end-to-end)", lambda fn: (v_true[fn][0], v_true[fn][1])),
        "v24content_B0len":   ("diagnostic",                   lambda fn: (v_true[fn][0], baseline_pred_len(meta[fn]["family_id"], policy))),
        "B0content_v24len":   ("diagnostic",                   lambda fn: (b0_content(fn), v_true[fn][1])),
        "B0content_B0len":    ("deployable (=B0 end-to-end)",  lambda fn: (b0_content(fn), baseline_pred_len(meta[fn]["family_id"], policy))),
    }
    gate_swap = {}
    for name, (kind, fn_get) in combos.items():
        rows = []
        for fn, core in cores.items():
            if core is None or fn not in v_true: continue
            content, L = fn_get(fn)
            B = panel_B(content, core, L)
            rows.append({"gene": meta[fn]["gene"], "covR": B["covR"], "coverage": B["coverage"], "len_mae": B["len_mae"]})
        gate_swap[name] = {"kind": kind, "gene_covR": _gene_mean(rows, "covR"),
                           "mean_coverage": float(np.mean([r["coverage"] for r in rows])),
                           "mean_len_mae": float(np.mean([r["len_mae"] for r in rows])), "n": len(rows)}

    # ---- 2. family_id shortcut -------------------------------------------------
    dcorr, dl1 = [], []
    for fn in v_true:
        if fn not in v_roll: continue
        a, b = v_true[fn][0], v_roll[fn][0]
        L = min(a.shape[1], b.shape[1])
        if L < 2: continue
        aa, bb = a[:, :L].ravel(), b[:, :L].ravel()
        dcorr.append(np.corrcoef(aa, bb)[0, 1] if aa.std() and bb.std() else 1.0)
        dl1.append(float(np.abs(a[:, :L] - b[:, :L]).mean()))
    # rolled covR
    roll_rows = []
    for fn, core in cores.items():
        if core is None or fn not in v_roll: continue
        B = panel_B(v_roll[fn][0], core, v_roll[fn][1]); roll_rows.append({"gene": meta[fn]["gene"], "covR": B["covR"]})
    shortcut = {
        "pred_corr_true_vs_rolled_familyid_mean": float(np.mean(dcorr)),
        "pred_L1_true_vs_rolled_mean": float(np.mean(dl1)),
        "v24_gene_covR_true": gate_swap["v24content_v24len"]["gene_covR"],
        "v24_gene_covR_rolled_familyid": _gene_mean(roll_rows, "covR"),
        "interpretation": "corr~1 and covR unchanged => family_id is NOT used at inference "
                          "(v24 is effectively metadata-free); large change => metadata shortcut.",
    }

    # ---- 3. family-residual: v24 deviation vs true deviation from family mean --
    res_corr = []
    for fn, core in cores.items():
        if core is None or fn not in v_true: continue
        fam = b0_content(fn)
        # align both v24 pred and family mean to the GT core frame
        pv, cols_v = _aligned_cols(v_true[fn][0], core)
        pf, cols_f = _aligned_cols(fam, core)
        cols = np.intersect1d(cols_v, cols_f)
        if len(cols) < 3: continue
        t = core[:, cols]; pv_ = pv[:, cols]; pf_ = pf[:, cols]
        dv = (pv_ - pf_).ravel(); dt = (t - pf_).ravel()   # v24 dev vs true dev from prior
        if dv.std() > 1e-8 and dt.std() > 1e-8:
            res_corr.append(float(np.corrcoef(dv, dt)[0, 1]))
    residual = {"mean_residual_corr_v24_dev_vs_true_dev": float(np.mean(res_corr)) if res_corr else None,
                "n": len(res_corr),
                "interpretation": "positive => v24 captures within-family specificity beyond the "
                                  "family-mean prior; ~0 => v24 adds little over the prior."}

    out = {"gate_swap_2x2": gate_swap, "family_id_shortcut": shortcut, "family_residual": residual}
    json.dump(out, open(f"{ROOT}/analyze_dumped.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
