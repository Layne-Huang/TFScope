#!/usr/bin/env python
"""Evaluate pwm_rosetta crystal predictions against DeepPBS ground truth.

Ground truth: Y_pwm masked by pwm_mask from
  /n/holylabs/.../deeppbsmar24/data/assembly2024/<entry>.npz

For each test entry:
  1. Pick the matching DeepPBS NPZ by motif ID in the filename.
  2. true_ppm  = Y_pwm[0][pwm_mask[0]]  — binding-site PPM, shape (K, 4)
  3. pred_ppm  = sliding-window best alignment of scan PPM to true_ppm
  4. Compute Pearson r, MAE, IC-r, Top-1.

Usage:
    python scripts/eval_crystal_pwm_rosetta.py
    python scripts/eval_crystal_pwm_rosetta.py --out results/crystal_pwm_rosetta
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

CRYSTAL_RUN_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/crystal_test"
SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
DEEPPBS_NPZ_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/deeppbsmar24/data/assembly2024"
DEEPPBS_PER_SAMPLE = "results/deeppbs_blind_benchmark/per_sample.json"
BASES = ["A", "C", "G", "T"]
TAU = 1.5


# ── NPZ selection ─────────────────────────────────────────────────────────────

def pick_deeppbs_npz(tid, npz_dir):
    """Return the DeepPBS NPZ path that matches the motif ID in tid."""
    entry = tid.replace(".txt", "")
    parts = entry.split("_")
    prefix = parts[0] + "_" + parts[1] + "_"
    suffix = "_".join(parts[2:])                       # e.g. 'Egr1.MA0162.1'
    motif_id = suffix.split(".", 1)[1] if "." in suffix else suffix

    cands = [f for f in os.listdir(npz_dir) if f.startswith(prefix)]
    if not cands:
        return None
    if len(cands) == 1:
        return os.path.join(npz_dir, cands[0])

    for c in cands:
        body = c[len(prefix):].replace(".npz", "")
        if motif_id in body:
            return os.path.join(npz_dir, c)
    return os.path.join(npz_dir, cands[0])             # fallback


# ── PPM from scan CSV ─────────────────────────────────────────────────────────

def ppm_from_csv(csv_path, tau=TAU):
    """Return (sorted_positions, PPM) where PPM shape=(L,4), positions 1-based."""
    df = pd.read_csv(csv_path)
    pos_df = df.drop_duplicates("position").sort_values("position")
    positions = pos_df["position"].tolist()
    seq = "".join(pos_df["original"].tolist())
    L = len(seq)

    pos_to_idx = {p: i for i, p in enumerate(positions)}
    wt_ddg = df["wt_ddg"].iloc[0] if "wt_ddg" in df.columns else df["mut_ddg"].min()
    energies = np.full((L, 4), wt_ddg)
    for i, b in enumerate(seq):
        energies[i][BASES.index(b)] = wt_ddg
    for _, row in df.iterrows():
        idx = pos_to_idx.get(int(row["position"]))
        if idx is None:
            continue
        if row["mutant"] != row["original"]:
            energies[idx][BASES.index(row["mutant"])] = row["mut_ddg"]

    dE = energies - energies.min(axis=1, keepdims=True)
    PPM = np.exp(-dE / tau)
    PPM /= PPM.sum(axis=1, keepdims=True)
    PPM = np.clip(PPM, 1e-6, 1.0)
    PPM /= PPM.sum(axis=1, keepdims=True)
    return positions, PPM


# ── Alignment ─────────────────────────────────────────────────────────────────

def best_align_ppm(pred, true):
    """Slide pred over true (or vice versa) to maximise Pearson r.

    Returns pred slice of length min(L_pred, L_true) best aligned to true[:K].
    """
    L_pred, K = len(pred), len(true)
    if L_pred <= K:
        return pred                    # pred is shorter; return as-is
    flat_true = true.flatten()
    best_r, best_start = -2.0, 0
    for start in range(L_pred - K + 1):
        flat_pred = pred[start:start + K].flatten()
        if flat_pred.std() < 1e-8 or flat_true.std() < 1e-8:
            continue
        r = pearsonr(flat_pred, flat_true)[0]
        if r > best_r:
            best_r, best_start = r, start
    return pred[best_start:best_start + K]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred_ppm, true_ppm):
    """Both arrays (L,4), PPM (rows sum to ~1). Returns metric dict."""
    L = min(len(pred_ppm), len(true_ppm))
    pred, true = pred_ppm[:L], true_ppm[:L]

    # Per-position Pearson r on PPM rows, then mean
    rs = []
    for i in range(L):
        if true[i].std() > 1e-8 and pred[i].std() > 1e-8:
            rs.append(pearsonr(true[i], pred[i])[0])
    r = float(np.nanmean(rs)) if rs else float("nan")

    # MAE: DeepPBS convention — sum |pred-true| per position, mean over positions
    mae = float(np.mean(np.sum(np.abs(pred - true), axis=1)))

    # IC-r: Pearson r between per-position IC vectors
    def ic(ppm):
        p = np.clip(ppm, 1e-10, 1.0)
        return (p * np.log2(p / 0.25)).sum(axis=1)

    ic_pred, ic_true = ic(pred), ic(true)
    if L > 1 and ic_pred.std() > 1e-8 and ic_true.std() > 1e-8:
        ic_r = float(pearsonr(ic_pred, ic_true)[0])
    else:
        ic_r = float("nan")

    top1 = float(np.mean(np.argmax(pred, axis=1) == np.argmax(true, axis=1)))

    return {"pearson_r": r, "mae": mae, "ic_r": ic_r, "top1": top1, "n_pos": L}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=CRYSTAL_RUN_DIR)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--npz-dir", default=DEEPPBS_NPZ_DIR)
    ap.add_argument("--deeppbs-per-sample", default=DEEPPBS_PER_SAMPLE)
    ap.add_argument("--out", default="results/crystal_pwm_rosetta")
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.split) as f:
        test_ids = json.load(f)["test"]

    # DeepPBS per-sample (keyed by pdb+chain prefix)
    deeppbs = {}
    if os.path.exists(args.deeppbs_per_sample):
        with open(args.deeppbs_per_sample) as f:
            for s in json.load(f):
                name = s.get("name", "")
                parts = name.split("_")
                if len(parts) >= 2:
                    deeppbs[parts[0] + "_" + parts[1]] = s

    rows, skipped = [], []

    for tid in test_ids:
        entry = tid.replace(".txt", "")
        parts = entry.split("_")
        pdb_id, chain_id = parts[0], parts[1]
        work_dir = os.path.join(args.run_dir, entry)
        csv_path = os.path.join(work_dir, "pwm_results_hybrid.csv")

        # ── Load ground truth from DeepPBS NPZ ──
        npz_path = pick_deeppbs_npz(tid, args.npz_dir)
        if npz_path is None or not os.path.exists(csv_path):
            skipped.append(tid)
            continue

        try:
            npz = np.load(npz_path, allow_pickle=True)
            Y_pwm = npz["Y_pwm"]          # (2, L, 4)
            pwm_mask = npz["pwm_mask"]    # (2, L) bool
            true_ppm = Y_pwm[0][pwm_mask[0]]   # (K, 4)  binding-site PPM
        except Exception as e:
            print(f"[WARN] {entry}: NPZ load failed: {e}")
            skipped.append(tid)
            continue

        if len(true_ppm) == 0:
            skipped.append(tid)
            continue

        # ── Load predicted PPM from scan CSV ──
        try:
            _, pred_full = ppm_from_csv(csv_path, tau=args.tau)
        except Exception as e:
            print(f"[WARN] {entry}: ppm_from_csv failed: {e}")
            skipped.append(tid)
            continue

        # ── Align predicted to ground truth ──
        pred_aligned = best_align_ppm(pred_full, true_ppm)

        # ── Compute metrics ──
        m = compute_metrics(pred_aligned, true_ppm)
        row = {"filename": tid, "entry": entry,
               "n_binding": len(true_ppm), **m}

        dp_key = pdb_id + "_" + chain_id
        if dp_key in deeppbs:
            dp = deeppbs[dp_key]
            row["deeppbs_r"]    = dp.get("pearson_r", float("nan"))
            row["deeppbs_mae"]  = dp.get("mae", float("nan"))
            row["deeppbs_ic_r"] = dp.get("ic_pearson", float("nan"))

        rows.append(row)

    # ── Output ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "per_sample.csv"), index=False)

    n = len(df)
    print(f"\n{'='*62}")
    print(f"pwm_rosetta (crystal)  n={n}/{len(test_ids)}"
          f"  ground truth: DeepPBS Y_pwm @ pwm_mask")
    print(f"{'='*62}")
    print(f"  Mean Pearson r   : {df['pearson_r'].mean():.3f}")
    print(f"  Median Pearson r : {df['pearson_r'].median():.3f}")
    print(f"  MAE (×4 scale)   : {df['mae'].mean():.3f}")
    print(f"  IC-r             : {df['ic_r'].mean():.3f}")
    print(f"  Top-1            : {df['top1'].mean():.3f}")
    print(f"  Mean binding pos : {df['n_binding'].mean():.1f}")

    if "deeppbs_r" in df.columns:
        dp_sub = df.dropna(subset=["deeppbs_r"])
        print(f"\n  --- DeepPBS (same {len(dp_sub)} samples) ---")
        print(f"  Mean Pearson r   : {dp_sub['deeppbs_r'].mean():.3f}")
        print(f"  MAE (×4 scale)   : {dp_sub['deeppbs_mae'].mean():.3f}")
        print(f"  IC-r             : {dp_sub['deeppbs_ic_r'].mean():.3f}")

    if skipped:
        print(f"\nSkipped {len(skipped)}: {skipped[:5]}")

    agg = {
        "n": n, "total": len(test_ids),
        "mean_r":   float(df["pearson_r"].mean()),
        "median_r": float(df["pearson_r"].median()),
        "mae":      float(df["mae"].mean()),
        "ic_r":     float(df["ic_r"].mean()),
        "top1":     float(df["top1"].mean()),
        "mean_binding_pos": float(df["n_binding"].mean()),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
