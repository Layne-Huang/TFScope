"""Canonical-fixed (deployable, no-peeking) covR — cross-check for oracle covR.

The validation selector and the Panel-A/B numbers use *oracle* offset+RC
registration (peeks at the target to pick the best frame/strand). This module
scores the SAME predictions under a deterministic **canonical** registration
(trim low-IC flanks + canonical-strand rule, fixed left-anchor) — the honest
deployable frame, with no target-peeking. If the model ranking survives here,
the oracle-registration selection concern is moot.

Coverage-inclusive: prediction is left-anchored to the canonical target core and
scored per-column over ALL target columns (uncovered -> ~0), exactly like
eval_full_metrics.canon_fixed_r but with BOTH pred and target canonicalized.
Gene-balanced aggregation, identical to the panels.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from canonicalize_pwms import canonicalize
from iclr.baselines import _decode_pwm
from iclr.unified_eval import train_length_policy   # (unused length; kept for parity)

DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
ROOT = "results/iclr_phase1_apples_to_apples"


def canon_fixed_covR(pred, target_raw):
    """Deployable fixed-frame covR: canonicalize both, left-anchor, per-col r over
    the full canonical target core (uncovered columns penalised)."""
    tc = canonicalize(np.clip(np.asarray(target_raw, np.float32), 1e-8, 1.0))
    tc = tc / tc.sum(0, keepdims=True).clip(1e-8)
    cp = canonicalize(np.clip(np.asarray(pred, np.float32), 1e-8, 1.0))
    cp = cp / cp.sum(0, keepdims=True).clip(1e-8)
    Lt = tc.shape[1]
    if Lt == 0:
        return None
    pp = np.full((4, Lt), 0.25, np.float32)
    L = min(cp.shape[1], Lt); pp[:, :L] = cp[:, :L]
    rs = [0.0 if (tc[:, j].std() < 1e-8 or pp[:, j].std() < 1e-8)
          else np.corrcoef(tc[:, j], pp[:, j])[0, 1] for j in range(Lt)]
    return float(np.mean(rs))


def _group_mean(pwms):
    W = int(np.median([p.shape[1] for p in pwms])) or 1
    acc = np.zeros((4, W)); n = np.zeros(W)
    for p in pwms:
        L = min(p.shape[1], W); acc[:, :L] += p[:, :L] / p[:, :L].sum(0, keepdims=True).clip(1e-8); n[:L] += 1
    avg = np.where(n > 0, acc / np.maximum(n, 1), 0.25)
    return (avg / avg.sum(0, keepdims=True).clip(1e-8)).astype(np.float32)


def _gene_covR(preds, targets, meta):
    by_gene = {}
    for fn, pr in preds.items():
        if fn not in targets: continue
        v = canon_fixed_covR(pr, targets[fn])
        if v is None: continue
        by_gene.setdefault(meta[fn]["gene"], []).append(v)
    gm = {g: float(np.mean(v)) for g, v in by_gene.items()}
    return (float(np.mean(list(gm.values()))) if gm else float("nan")), len(by_gene)


def main():
    df = pd.read_parquet(DATA); df["filename"] = df.filename.astype(str)
    sp = json.load(open(SPLIT))
    tr = df[df.filename.isin(set(sp["train"]))].copy(); te = df[df.filename.isin(set(sp["test"]))].copy()
    tr["pwm_arr"] = tr.pwm.map(_decode_pwm)
    targets = {r.filename: _decode_pwm(r.pwm) for r in te.itertuples()}
    meta = {r.filename: {"gene": r.gene_symbol, "family_id": int(r.family_id),
                         "family": r.family_name} for r in te.itertuples()}
    exact = {k: _group_mean(list(g["pwm_arr"])) for k, g in tr.groupby("family_name")}
    exact_n = {k: len(g) for k, g in tr.groupby("family_name")}
    coarse = {int(k): _group_mean(list(g["pwm_arr"])) for k, g in tr.groupby("family_id")}
    gpri = _group_mean(list(tr["pwm_arr"]))

    out = {}
    # v24 from dumped gated content
    vp = json.load(open(f"{ROOT}/v24_predictions.json"))["true"]
    v24_preds = {k: np.array(v["content"], np.float32) for k, v in vp.items()}
    out["v24_seed42"] = _gene_covR(v24_preds, targets, meta)
    # baselines
    def bpred(kind):
        d = {}
        for r in te.itertuples():
            if kind == "exact":
                d[r.filename] = exact[r.family_name] if (r.family_name in exact and exact_n[r.family_name] > 0) else gpri
            elif kind == "coarse":
                d[r.filename] = coarse.get(int(r.family_id), gpri)
            else:
                d[r.filename] = gpri
        return d
    out["B0_exact_family"] = _gene_covR(bpred("exact"), targets, meta)
    out["B0_coarse_familyid"] = _gene_covR(bpred("coarse"), targets, meta)
    out["B0_global"] = _gene_covR(bpred("global"), targets, meta)
    # B1 nearest
    from iclr.baselines import _nearest_pwm
    b1 = {r.filename: _nearest_pwm(r.sequence, tr) for r in te.itertuples()}
    out["B1_nearest"] = _gene_covR(b1, targets, meta)
    # B2/B3/B4 if their dumps exist
    for tag in ["B2", "B3", "B4"]:
        f = f"{ROOT}/preds_{tag}.json"
        if os.path.exists(f):
            dd = json.load(open(f))
            for seedkey, pdict in dd.items():
                pr = {k: np.array(v["content"], np.float32) for k, v in pdict.items()}
                out[f"{tag}_{seedkey}"] = _gene_covR(pr, targets, meta)

    res = {k: {"canonical_fixed_gene_covR": v[0], "n_genes": v[1]} for k, v in out.items()}
    json.dump(res, open(f"{ROOT}/canonical_fixed.json", "w"), indent=2)
    print(f"{'model':<24}{'canonical-fixed gene_covR':>28}")
    for k, v in res.items():
        print(f"  {k:<22}{v['canonical_fixed_gene_covR']:>26.4f}")
    print(f"saved {ROOT}/canonical_fixed.json")


if __name__ == "__main__":
    main()
