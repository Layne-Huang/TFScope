#!/usr/bin/env python
"""Pure-retrieval baseline: predict the test PWM as the nearest training TF's PWM.

For each test TF, look up its top-K nearest training neighbours (from tf_nn_index.json),
average their PWMs (after aligning to common length), and compute metrics against the truth.

Reports:
  - Top-1 NN (best single neighbour)
  - Top-K=3 average
  - Top-K=5 weighted average (by cosine similarity)
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

MAX_LEN = 20


def per_position_pearson(pred, target):
    rs = []
    for i in range(pred.shape[1]):
        r = pearsonr(target[:, i], pred[:, i])[0]
        if not np.isnan(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def load_pwm(fn, df):
    row = df[df["filename"] == fn].iloc[0]
    pwm = np.frombuffer(row["pwm"], dtype=np.float32).reshape(4, -1)   # (4, L_native)
    return pwm


def align_pwm_to_length(pwm_src, target_L):
    """Truncate or pad (with uniform) to target_L. Both sources are left-aligned."""
    pwm_aligned = np.full((4, target_L), 0.25, dtype=np.float32)
    L = min(pwm_src.shape[1], target_L)
    pwm_aligned[:, :L] = pwm_src[:, :L]
    return pwm_aligned


def evaluate_set(test_fns, df, idx, k_mode="top1"):
    """k_mode: 'top1' | 'top3_mean' | 'top5_sim_weighted'."""
    r_list, mae_list = [], []
    coverage = 0
    for fn in test_fns:
        neighbours = idx.get(fn, [])
        if not neighbours:
            continue
        coverage += 1
        true_pwm = load_pwm(fn, df)        # (4, L_true)
        L_true   = true_pwm.shape[1]

        if k_mode == "top1":
            nn_pwm = load_pwm(neighbours[0]["nn_filename"], df)
            pred = align_pwm_to_length(nn_pwm, L_true)
        elif k_mode == "top3_mean":
            preds = []
            for nb in neighbours[:3]:
                nn_pwm = load_pwm(nb["nn_filename"], df)
                preds.append(align_pwm_to_length(nn_pwm, L_true))
            pred = np.mean(preds, axis=0)
        elif k_mode == "top5_sim_weighted":
            preds, weights = [], []
            for nb in neighbours[:5]:
                nn_pwm = load_pwm(nb["nn_filename"], df)
                preds.append(align_pwm_to_length(nn_pwm, L_true))
                weights.append(max(0.0, nb["cos_sim"]))
            preds   = np.stack(preds)                              # (k, 4, L)
            weights = np.array(weights) / (sum(weights) + 1e-8)    # (k,)
            pred    = (preds * weights[:, None, None]).sum(axis=0)
        else:
            raise ValueError(k_mode)

        # Re-normalize columns to sum to 1
        pred = np.clip(pred, 1e-8, None)
        pred = pred / pred.sum(axis=0, keepdims=True)

        r_list.append(per_position_pearson(pred, true_pwm))
        mae_list.append(float(np.abs(pred - true_pwm).mean()))

    return {
        "n":               coverage,
        "pearson_r_mean":   float(np.nanmean(r_list)),
        "pearson_r_median": float(np.nanmedian(r_list)),
        "mae_mean":         float(np.mean(mae_list)),
        "mae_median":       float(np.median(mae_list)),
        "mae_dp_x4":        float(np.mean(mae_list) * 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default="data/processed/tf_pwm_deeppbs_only.parquet")
    ap.add_argument("--split", default="data/processed/splits/deeppbs_only/benchmark_no_val.json")
    ap.add_argument("--index", default="data/processed/tf_nn_index.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    with open(args.split) as f:
        split = json.load(f)
    with open(args.index) as f:
        idx = json.load(f)

    test_fns = sorted(split["test"])
    print(f"Pure-retrieval baseline on {len(test_fns)} test TFs")
    print(f"Index file: {args.index}\n")

    for mode in ["top1", "top3_mean", "top5_sim_weighted"]:
        res = evaluate_set(test_fns, df, idx, mode)
        print(f"  {mode:<20}  n={res['n']:3d}  "
              f"Pearson r mean={res['pearson_r_mean']:.4f}  med={res['pearson_r_median']:.4f}  "
              f"MAE={res['mae_mean']:.4f}  MAE×4={res['mae_dp_x4']:.4f}")


if __name__ == "__main__":
    main()
