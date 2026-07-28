"""Score training-free baselines (B0/B1) through the unified two-panel evaluator.

CPU-only, no GPU contention. Baselines use the pre-registered per-family training
length policy for Panel B (never reads the test target).
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "src")
from iclr.baselines import _decode_pwm, _family_average, _nearest_pwm
from iclr.unified_eval import score_model, train_length_policy, baseline_pred_len

DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
OUT = "results/iclr_phase1_apples_to_apples/unified_baselines.json"


def main(which):
    df = pd.read_parquet(DATA); df["filename"] = df.filename.astype(str)
    sp = json.load(open(SPLIT))
    tr = df[df.filename.isin(set(sp["train"]))].copy()
    te = df[df.filename.isin(set(sp["test"]))].copy()
    tr["pwm_arr"] = tr.pwm.map(_decode_pwm); te["pwm_arr"] = te.pwm.map(_decode_pwm)
    policy = train_length_policy(tr)
    fam_avg = _family_average(tr)

    targets = {r.filename: r.pwm_arr for r in te.itertuples()}
    meta = {r.filename: {"gene": r.gene_symbol, "family_id": int(r.family_id),
                         "family": r.family_name, "n_chains": int(getattr(r, "n_chains", 1))}
            for r in te.itertuples()}
    length_fn = lambda fn, m: baseline_pred_len(m["family_id"], policy)

    out = {}
    if "B0" in which:
        preds = {r.filename: fam_avg.get(int(r.family_id),
                 np.full((4, max(1, baseline_pred_len(int(r.family_id), policy))), 0.25, np.float32))
                 for r in te.itertuples()}
        out["B0"] = score_model("B0_family_avg", preds, targets, meta, length_fn=length_fn)
        print(f"[B0] panelA gene_content_r={out['B0']['panelA_oracle_content']['gene_content_r']:.4f}  "
              f"panelB gene_covR={out['B0']['panelB_end_to_end']['gene_covR']:.4f}  "
              f"cov={out['B0']['panelB_end_to_end']['mean_coverage']:.3f}  "
              f"len_mae={out['B0']['panelB_end_to_end']['gate_len_mae']:.2f}")
    if "B1" in which:
        preds = {r.filename: _nearest_pwm(r.sequence, tr) for r in te.itertuples()}
        out["B1"] = score_model("B1_nearest", preds, targets, meta, length_fn=length_fn)
        print(f"[B1] panelA gene_content_r={out['B1']['panelA_oracle_content']['gene_content_r']:.4f}  "
              f"panelB gene_covR={out['B1']['panelB_end_to_end']['gene_covR']:.4f}  "
              f"cov={out['B1']['panelB_end_to_end']['mean_coverage']:.3f}  "
              f"len_mae={out['B1']['panelB_end_to_end']['gate_len_mae']:.2f}")

    # merge into existing file if present
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev.update(out)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(prev, open(OUT, "w"), indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    which = sys.argv[1:] or ["B0", "B1"]
    main(which)
