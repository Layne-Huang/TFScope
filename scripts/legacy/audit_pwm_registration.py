#!/usr/bin/env python
"""Audit same-gene PWM consistency and build consensus-relative E3 labels."""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, "src")

from tfscope.data.registration_audit import (
    AuditThresholds,
    build_pair_rows,
    deduplicate_motifs,
    summarize_gene,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/tf_pwm_aug_dbd_canon_trim.parquet",
    )
    parser.add_argument(
        "--split",
        default="data/processed/splits/cluster40_clean/split.json",
    )
    parser.add_argument("--out-dir", default="results/v19_e3_registration")
    parser.add_argument("--min-overlap", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=10)
    parser.add_argument("--consistent-median-r", type=float, default=0.75)
    parser.add_argument("--consistent-q25-r", type=float, default=0.60)
    parser.add_argument("--cluster-r", type=float, default=0.65)
    parser.add_argument("--registration-gain", type=float, default=0.15)
    parser.add_argument("--discordant-pair-fraction", type=float, default=0.25)
    parser.add_argument("--anchor-min-r", type=float, default=0.80)
    parser.add_argument("--anchor-min-coverage", type=float, default=0.60)
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = AuditThresholds(
        min_overlap=args.min_overlap,
        max_shift=args.max_shift,
        consistent_median_r=args.consistent_median_r,
        consistent_q25_r=args.consistent_q25_r,
        cluster_r=args.cluster_r,
        registration_gain=args.registration_gain,
        discordant_pair_fraction=args.discordant_pair_fraction,
        anchor_min_r=args.anchor_min_r,
        anchor_min_coverage=args.anchor_min_coverage,
    )

    df = pd.read_parquet(args.data)
    with open(args.split) as handle:
        split = json.load(handle)
    filename_to_split = {
        filename: split_name
        for split_name in ("train", "val", "test")
        for filename in split[split_name]
    }
    df = df[df["filename"].isin(filename_to_split)].copy()
    df["split"] = df["filename"].map(filename_to_split)
    unique, deduplication = deduplicate_motifs(df)

    pair_rows = []
    gene_rows = []
    anchor_rows = []
    for _, records in unique.groupby("gene_symbol", sort=True):
        pairs = build_pair_rows(records, thresholds)
        summary, anchors = summarize_gene(records, pairs, thresholds)
        pair_rows.extend(pairs)
        gene_rows.append(summary)
        anchor_rows.extend(anchors)

    pairs_df = pd.DataFrame(pair_rows)
    genes_df = pd.DataFrame(gene_rows)
    anchors_df = pd.DataFrame(anchor_rows)
    os.makedirs(args.out_dir, exist_ok=True)
    pairs_path = os.path.join(args.out_dir, "pairwise_alignments.tsv")
    genes_path = os.path.join(args.out_dir, "gene_consistency.tsv")
    anchors_path = os.path.join(args.out_dir, "relative_registration_anchors.tsv")
    train_anchors_path = os.path.join(
        args.out_dir, "relative_registration_anchors_train.tsv"
    )
    report_path = os.path.join(args.out_dir, "audit_summary.json")
    pairs_df.to_csv(pairs_path, sep="\t", index=False)
    genes_df.to_csv(genes_path, sep="\t", index=False)
    anchors_df.to_csv(anchors_path, sep="\t", index=False)
    train_anchors_df = anchors_df.loc[anchors_df["split"] == "train"].copy()
    train_anchors_df.to_csv(train_anchors_path, sep="\t", index=False)

    classification_counts = Counter(genes_df["classification"])
    split_counts = {
        split_name: dict(
            Counter(
                genes_df.loc[
                    genes_df["split"] == split_name, "classification"
                ]
            )
        )
        for split_name in ("train", "val", "test")
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "data": args.data,
            "split": args.split,
            "deduplication_key": [
                "gene_symbol",
                "source_id",
                "rounded_pwm_sha256",
            ],
            "alignment": (
                "Offset and reverse-complement search maximizing mean per-column "
                "Pearson multiplied by overlap/max(lengths)."
            ),
            "anchor_semantics": (
                "Consensus-relative pseudo-anchors only. Same-gene agreement does "
                "not resolve absolute strand symmetry."
            ),
            "thresholds": thresholds.to_dict(),
        },
        "deduplication": deduplication,
        "n_genes": int(len(genes_df)),
        "n_multi_record_genes": int((genes_df["n_unique_motifs"] > 1).sum()),
        "n_pairwise_alignments": int(len(pairs_df)),
        "n_relative_anchor_labels": int(len(anchors_df)),
        "n_train_relative_anchor_labels": int(len(train_anchors_df)),
        "n_anchor_genes": int(anchors_df["gene_symbol"].nunique())
        if len(anchors_df)
        else 0,
        "n_train_anchor_genes": int(train_anchors_df["gene_symbol"].nunique())
        if len(train_anchors_df)
        else 0,
        "classification_counts": dict(classification_counts),
        "classification_counts_by_split": split_counts,
        "outputs": {
            "pairwise_alignments": pairs_path,
            "gene_consistency": genes_path,
            "relative_registration_anchors": anchors_path,
            "train_relative_registration_anchors": train_anchors_path,
        },
    }
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
