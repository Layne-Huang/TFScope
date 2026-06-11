#!/usr/bin/env python
"""Re-trim already-canonicalized PWMs at a stricter IC threshold.

The existing canon parquets were trimmed at IC>=0.25 bits, leaving soft flanking
columns (aug novel right-flank IC=0.785 bits vs DeepPBS 0.969 bits). This script
re-trims both ends at IC>=0.5 bits to remove those flanks without touching the
strand orientation (already canonical).

Usage:
    python scripts/retrim_pwms.py --ic-thresh 0.5
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path


def pwm_ic(pwm):
    p = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)


def retrim(pwm, thresh):
    ic = pwm_ic(pwm)
    inf = np.where(ic >= thresh)[0]
    if len(inf) == 0:
        return pwm
    return pwm[:, inf[0]:inf[-1] + 1]


def process(in_path, out_path, thresh, max_len=20):
    df = pd.read_parquet(in_path)
    new_pwm, new_len = [], []
    n_changed = n_zero = 0
    for _, row in df.iterrows():
        raw = row["pwm"]
        L = int(row["motif_length"])
        if not isinstance(raw, bytes):
            new_pwm.append(raw); new_len.append(L); continue
        pwm = np.frombuffer(raw, dtype=np.float32).reshape(4, -1)[:, :L]
        trimmed = retrim(pwm, thresh)
        if trimmed.shape[1] == 0:
            trimmed = pwm; n_zero += 1  # keep original if all columns stripped
        if trimmed.shape[1] != L:
            n_changed += 1
        trimmed = trimmed[:, :max_len].astype(np.float32)
        trimmed = trimmed / trimmed.sum(0, keepdims=True).clip(1e-8)
        new_pwm.append(trimmed.tobytes())
        new_len.append(trimmed.shape[1])
    df = df.copy()
    df["pwm"] = new_pwm
    df["motif_length"] = new_len
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  {in_path}")
    print(f"  → {out_path}")
    print(f"     {len(df)} rows | changed={n_changed} ({100*n_changed/len(df):.1f}%)"
          f" | zero_stripped={n_zero}")
    s = df["motif_length"]
    print(f"     length: mean={s.mean():.1f}  med={s.median():.0f}"
          f"  p95={s.quantile(.95):.0f}  at_cap={(s==20).sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic-thresh", type=float, default=0.5)
    ap.add_argument("--max-len",   type=int,   default=20)
    ap.add_argument("--pairs", nargs="+", default=[
        "data/processed/tf_pwm_deeppbs_only_canon.parquet:"
        "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet",
        "data/processed/tf_pwm_aug_dbd_canon.parquet:"
        "data/processed/tf_pwm_aug_dbd_canon_trim.parquet",
    ])
    args = ap.parse_args()
    print(f"Re-trimming at IC>={args.ic_thresh} bits, max_len={args.max_len}\n")
    for pair in args.pairs:
        src, dst = pair.split(":")
        process(src, dst, args.ic_thresh, args.max_len)
    print("\nDone.")


if __name__ == "__main__":
    main()
