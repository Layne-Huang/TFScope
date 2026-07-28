"""Frozen, train-only family taxonomy and three priors (ICLR audit Task 1).

Builds three strictly train-only PWM priors, documents the family_id vs
family_name mismatch, and scores B0 under each through the unified evaluator so
the confounded `family_id` prior can be replaced by a clean exact-family prior.

  * global   : one prior = mean of ALL train PWMs (fallback for zero-shot).
  * coarse   : by `family_id` (the taxonomy v24 is conditioned on; p53->"Other"
               bin, POU->Homeodomain bin — the confounded current B0).
  * exact    : by biological `family_name`; families with 0 train rows
               (p53, POU) fall back to the GLOBAL prior, reported separately.

No test PWM ever enters a prototype. Left-anchored mean to the group's median
train motif length.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "src")
from iclr.baselines import _decode_pwm
from iclr.unified_eval import score_model, train_length_policy, baseline_pred_len

DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
ROOT = "results/iclr_phase1_apples_to_apples"


def _group_mean(pwms):
    W = int(np.median([p.shape[1] for p in pwms])) or 1
    acc = np.zeros((4, W)); n = np.zeros(W)
    for p in pwms:
        L = min(p.shape[1], W)
        acc[:, :L] += p[:, :L] / p[:, :L].sum(0, keepdims=True).clip(1e-8); n[:L] += 1
    avg = np.where(n > 0, acc / np.maximum(n, 1), 0.25)
    return (avg / avg.sum(0, keepdims=True).clip(1e-8)).astype(np.float32)


def build(train_df, col):
    priors, counts = {}, {}
    for key, g in train_df.groupby(col):
        priors[key] = _group_mean(list(g["pwm_arr"]))
        counts[str(key)] = {"n_rows": int(len(g)), "n_genes": int(g["gene_symbol"].nunique())}
    return priors, counts


def main():
    df = pd.read_parquet(DATA); df["filename"] = df.filename.astype(str)
    sp = json.load(open(SPLIT))
    tr = df[df.filename.isin(set(sp["train"]))].copy()
    te = df[df.filename.isin(set(sp["test"]))].copy()
    tr["pwm_arr"] = tr.pwm.map(_decode_pwm); te["pwm_arr"] = te.pwm.map(_decode_pwm)
    policy = train_length_policy(tr)

    global_prior = _group_mean(list(tr["pwm_arr"]))
    coarse, coarse_counts = build(tr, "family_id")     # == current B0 taxonomy
    exact, exact_counts = build(tr, "family_name")

    # document the id<->name mismatch for the zero-shot families
    mismatch = {}
    for fname in te["family_name"].unique():
        te_sub = te[te.family_name == fname]
        for fid in te_sub.family_id.unique():
            tr_names = tr[tr.family_id == fid]["family_name"].value_counts().to_dict()
            if str(fname) not in tr_names:      # zero-shot by name but id present in train
                mismatch[str(fname)] = {"family_id": int(fid),
                                        "train_names_sharing_this_id": tr_names,
                                        "n_test_rows": int(len(te_sub))}

    targets = {r.filename: r.pwm_arr for r in te.itertuples()}
    meta = {r.filename: {"gene": r.gene_symbol, "family_id": int(r.family_id),
                         "family": r.family_name, "n_chains": int(getattr(r, "n_chains", 1))}
            for r in te.itertuples()}
    length_fn = lambda fn, m: baseline_pred_len(m["family_id"], policy)

    def preds_for(level):
        out = {}; fell_back = []
        for r in te.itertuples():
            if level == "global":
                out[r.filename] = global_prior
            elif level == "coarse":
                out[r.filename] = coarse.get(int(r.family_id), global_prior)
            else:  # exact by family_name, global fallback for zero-shot
                if r.family_name in exact and exact_counts[str(r.family_name)]["n_rows"] > 0:
                    out[r.filename] = exact[r.family_name]
                else:
                    out[r.filename] = global_prior; fell_back.append(str(r.family_name))
        return out, sorted(set(fell_back))

    baselines = json.load(open(f"{ROOT}/unified_baselines.json")) if os.path.exists(f"{ROOT}/unified_baselines.json") else {}
    summary = {}
    for level, tag in [("global", "B0_global"), ("coarse", "B0_coarse_familyid"), ("exact", "B0_exact_family")]:
        preds, fb = preds_for(level)
        res = score_model(tag, preds, targets, meta, length_fn=length_fn)
        baselines[tag] = res
        A = res["panelA_oracle_content"]["gene_content_r"]; B = res["panelB_end_to_end"]["gene_covR"]
        summary[tag] = {"panelA_content_r": A, "panelB_covR": B, "zero_shot_fallback_families": fb}
        print(f"{tag:<22} A={A:.4f}  B={B:.4f}  fallback={fb}")
    json.dump(baselines, open(f"{ROOT}/unified_baselines.json", "w"), indent=2)

    frozen = {
        "note": "Frozen train-only family taxonomy + 3 priors. No test PWM used.",
        "global_train_prior": {"n_rows": int(len(tr)), "median_len": int(np.median([p.shape[1] for p in tr["pwm_arr"]]))},
        "family_id_vs_name_mismatch_zeroshot": mismatch,
        "coarse_family_id_counts": coarse_counts,
        "exact_family_name_counts": exact_counts,
        "test_family_name_counts": te["family_name"].value_counts().to_dict(),
        "B0_variants_scored": summary,
    }
    json.dump(frozen, open(f"{ROOT}/family_taxonomy_frozen.json", "w"), indent=2)
    print(f"\nmismatch (zero-shot name but id in train): {json.dumps(mismatch, indent=2)}")
    print(f"saved {ROOT}/family_taxonomy_frozen.json")


if __name__ == "__main__":
    main()
