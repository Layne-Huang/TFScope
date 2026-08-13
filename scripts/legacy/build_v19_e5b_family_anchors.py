#!/usr/bin/env python
"""Expand E5 anchors with validated train-only family orientation labels."""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from build_v19_e5_structural_anchors import load_forward_structural_pwm
from tfscope.data.registration_audit import align_pair, decode_pwm


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
    parser.add_argument(
        "--structural-anchors",
        default="results/v19_e5_registration/structural_anchors_train.tsv",
    )
    parser.add_argument(
        "--out",
        default="results/v19_e5b_registration/family_anchors_train.tsv",
    )
    parser.add_argument("--min-r", type=float, default=0.90)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--min-loo-orientation-accuracy", type=float, default=0.85)
    parser.add_argument("--min-loo-genes", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=10)
    parser.add_argument("--min-overlap", type=int, default=4)
    return parser.parse_args()


def best_alignment(target, candidates, args):
    best = None
    for reference_gene, path, reference_pwm in candidates:
        alignment = align_pair(
            reference_pwm,
            target,
            max_shift=args.max_shift,
            min_overlap=min(args.min_overlap, target.shape[1]),
        )
        candidate = (
            alignment["aligned_r"] * alignment["coverage"],
            alignment["aligned_r"],
            alignment["coverage"],
            reference_gene,
            path,
            alignment,
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return best


def main():
    args = parse_args()
    data = pd.read_parquet(args.data)
    with open(args.split) as handle:
        split = json.load(handle)
    data = data[data["filename"].isin(set(split["train"]))].copy()
    structural = pd.read_csv(args.structural_anchors, sep="\t")
    if not (structural["split"] == "train").all():
        raise ValueError("Structural anchor input must contain train rows only")

    direct = structural[
        structural["anchor_type"] == "deeppbs_forward_structural_frame"
    ]
    reference_rows = direct[
        ["gene_symbol", "family_name", "structural_npz"]
    ].drop_duplicates()
    references_by_family = {}
    for row in reference_rows.itertuples(index=False):
        references_by_family.setdefault(str(row.family_name), []).append(
            (
                str(row.gene_symbol),
                str(row.structural_npz),
                load_forward_structural_pwm(row.structural_npz),
            )
        )

    truth = structural.set_index("filename")
    loo_rows = []
    anchored_data = data[data["filename"].isin(set(structural["filename"]))]
    for row in anchored_data.itertuples(index=False):
        candidates = [
            reference
            for reference in references_by_family.get(str(row.family_name), [])
            if reference[0] != str(row.gene_symbol)
        ]
        best = best_alignment(decode_pwm(row.pwm), candidates, args)
        if best is None:
            continue
        _, aligned_r, coverage, _, _, alignment = best
        if aligned_r < args.min_r or coverage < args.min_coverage:
            continue
        label = truth.loc[row.filename]
        loo_rows.append(
            {
                "family_name": row.family_name,
                "gene_symbol": row.gene_symbol,
                "correct": (
                    alignment["orientation"]
                    == label["orientation_to_reference"]
                ),
            }
        )
    loo = pd.DataFrame(loo_rows)
    qualified = set()
    family_reliability = {}
    if not loo.empty:
        for family, group in loo.groupby("family_name"):
            by_gene = group.groupby("gene_symbol")["correct"].mean()
            accuracy = float(by_gene.mean())
            family_reliability[str(family)] = {
                "n_rows": int(len(group)),
                "n_genes": int(len(by_gene)),
                "gene_macro_orientation_accuracy": accuracy,
            }
            if (
                len(by_gene) >= args.min_loo_genes
                and accuracy >= args.min_loo_orientation_accuracy
            ):
                qualified.add(str(family))

    exact = structural.copy()
    exact["anchor_mode"] = "state"
    pseudo = []
    anchored_filenames = set(structural["filename"])
    for row in data.itertuples(index=False):
        family = str(row.family_name)
        if row.filename in anchored_filenames or family not in qualified:
            continue
        candidates = [
            reference
            for reference in references_by_family.get(family, [])
            if reference[0] != str(row.gene_symbol)
        ]
        best = best_alignment(decode_pwm(row.pwm), candidates, args)
        if best is None:
            continue
        _, aligned_r, coverage, reference_gene, path, alignment = best
        if aligned_r < args.min_r or coverage < args.min_coverage:
            continue
        pseudo.append(
            {
                "filename": row.filename,
                "gene_symbol": row.gene_symbol,
                "family_name": family,
                "split": "train",
                "source_id": row.source_id,
                "structural_npz": path,
                "orientation_to_reference": alignment["orientation"],
                "offset_to_reference": 0,
                "aligned_r": aligned_r,
                "overlap": alignment["overlap"],
                "coverage": coverage,
                "target_length": decode_pwm(row.pwm).shape[1],
                "structural_pwm_length": "",
                "anchor_type": (
                    "leave_one_gene_out_validated_family_orientation"
                ),
                "absolute_orientation_resolved": True,
                "anchor_mode": "orientation",
                "reference_gene": reference_gene,
            }
        )

    combined = pd.concat([exact, pd.DataFrame(pseudo)], ignore_index=True)
    combined = combined.sort_values(["gene_symbol", "filename"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combined.to_csv(args.out, sep="\t", index=False)
    summary = {
        "protocol": (
            "Retain exact E5 structural states. Add orientation-only labels "
            "for unanchored training records only in families whose "
            "leave-one-gene-out structural transfer meets the configured "
            "gene-macro accuracy and coverage thresholds."
        ),
        "thresholds": {
            "min_r": args.min_r,
            "min_coverage": args.min_coverage,
            "min_loo_orientation_accuracy": args.min_loo_orientation_accuracy,
            "min_loo_genes": args.min_loo_genes,
        },
        "family_reliability": family_reliability,
        "qualified_families": sorted(qualified),
        "exact_state_rows": int(len(exact)),
        "exact_state_genes": int(exact["gene_symbol"].nunique()),
        "orientation_only_rows": int(len(pseudo)),
        "orientation_only_genes": int(
            pd.DataFrame(pseudo)["gene_symbol"].nunique() if pseudo else 0
        ),
        "total_rows": int(len(combined)),
        "total_genes": int(combined["gene_symbol"].nunique()),
        "output": args.out,
    }
    summary_path = os.path.join(os.path.dirname(args.out), "anchor_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
