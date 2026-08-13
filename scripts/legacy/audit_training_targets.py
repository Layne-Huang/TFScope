#!/usr/bin/env python
"""Audit row duplication, family metadata, and contradictory PWM supervision."""
import argparse
import json

import numpy as np
import pandas as pd

from tfscope.models.alignment import align_pwm


def decode_pwm(value):
    if isinstance(value, bytes):
        return np.frombuffer(value, dtype=np.float32).reshape(4, -1)
    return np.asarray(value, dtype=np.float32)


def pair_score(a, b):
    _, _, _, score = align_pwm(
        a, b, max_shift=10, consider_revcomp=True, min_overlap=4
    )
    return float(score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--out", default="results/data_audit/target_consistency.json")
    args = parser.parse_args()

    frame = pd.read_parquet(args.data)
    if args.split:
        ids = set(json.load(open(args.split))[args.split_name])
        frame = frame[frame["filename"].isin(ids)].copy()
    frame["gene_key"] = frame["gene_symbol"].fillna("").astype(str).str.upper()
    frame["target_group"] = frame["gene_key"] + "|" + frame["sequence"].astype(str)

    group_rows = []
    for group, sub in frame.groupby("target_group"):
        pwms = [decode_pwm(value) for value in sub["pwm"]]
        scores = []
        # Bound quadratic work for highly duplicated genes while remaining deterministic.
        for i in range(min(len(pwms), 20)):
            for j in range(i + 1, min(len(pwms), 20)):
                scores.append(pair_score(pwms[i], pwms[j]))
        group_rows.append(
            {
                "group_id": group,
                "gene": sub["gene_key"].iat[0],
                "n_rows": len(sub),
                "n_sources": int(
                    sub.get("motif_source", pd.Series(["unknown"])).nunique()
                ),
                "length_min": int(sub["motif_length"].min()),
                "length_max": int(sub["motif_length"].max()),
                "pair_r_mean": float(np.mean(scores)) if scores else None,
                "pair_r_min": float(np.min(scores)) if scores else None,
            }
        )

    groups = pd.DataFrame(group_rows)
    payload = {
        "n_rows": int(len(frame)),
        "n_genes": int(frame["gene_key"].nunique()),
        "n_target_groups": int(frame["target_group"].nunique()),
        "row_to_gene_ratio": float(len(frame) / max(frame["gene_key"].nunique(), 1)),
        "family_counts": frame.get(
            "family_name", pd.Series(["unknown"] * len(frame))
        ).value_counts().to_dict(),
        "family_source_counts": frame.get(
            "family_source", pd.Series(["unknown"] * len(frame))
        ).value_counts().to_dict(),
        "conflicting_groups_r_lt_0_5": int(
            (groups["pair_r_mean"].fillna(1.0) < 0.5).sum()
        ),
        "groups": group_rows,
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({key: value for key, value in payload.items() if key != "groups"}, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
