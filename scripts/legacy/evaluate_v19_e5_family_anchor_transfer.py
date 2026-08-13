#!/usr/bin/env python
"""Evaluate train-only family-frame transfer of E5 structural anchors."""

import argparse
import json
import sys

import pandas as pd

sys.path.insert(0, "src")

from tfscope.data.registration_audit import align_pair, decode_pwm

from build_v19_e5_structural_anchors import load_forward_structural_pwm


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
        "--anchors",
        default="results/v19_e5_registration/structural_anchors_train.tsv",
    )
    parser.add_argument("--min-r", type=float, default=0.90)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--max-shift", type=int, default=10)
    parser.add_argument("--min-overlap", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_parquet(args.data)
    with open(args.split) as handle:
        split = json.load(handle)
    data = data[data["filename"].isin(set(split["train"]))].copy()
    anchors = pd.read_csv(args.anchors, sep="\t")
    if not (anchors["split"] == "train").all():
        raise ValueError("Family-frame transfer diagnostic requires train-only anchors")

    direct = anchors[
        anchors["anchor_type"] == "deeppbs_forward_structural_frame"
    ].copy()
    reference_rows = direct[
        ["gene_symbol", "family_name", "structural_npz"]
    ].drop_duplicates()
    references = {}
    for row in reference_rows.itertuples(index=False):
        key = (str(row.family_name), str(row.gene_symbol))
        references.setdefault(key, []).append(
            (str(row.structural_npz), load_forward_structural_pwm(row.structural_npz))
        )

    truth = anchors.set_index("filename")
    rows = []
    for row in data.itertuples(index=False):
        if row.filename not in truth.index:
            continue
        target = decode_pwm(row.pwm)
        candidates = []
        for (family, reference_gene), family_references in references.items():
            if family != str(row.family_name) or reference_gene == str(row.gene_symbol):
                continue
            candidates.extend(
                (reference_gene, path, pwm)
                for path, pwm in family_references
            )
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
        if best is None:
            continue
        _, aligned_r, coverage, reference_gene, path, alignment = best
        if aligned_r < args.min_r or coverage < args.min_coverage:
            continue
        label = truth.loc[row.filename]
        rows.append(
            {
                "filename": row.filename,
                "gene_symbol": row.gene_symbol,
                "family_name": row.family_name,
                "reference_gene": reference_gene,
                "structural_npz": path,
                "predicted_orientation": alignment["orientation"],
                "predicted_offset": alignment["shift"],
                "true_orientation": label["orientation_to_reference"],
                "true_offset": int(label["offset_to_reference"]),
                "aligned_r": aligned_r,
                "coverage": coverage,
                "orientation_correct": (
                    alignment["orientation"]
                    == label["orientation_to_reference"]
                ),
                "state_correct": (
                    alignment["orientation"]
                    == label["orientation_to_reference"]
                    and alignment["shift"] == int(label["offset_to_reference"])
                ),
            }
        )

    evaluated = pd.DataFrame(rows)
    if evaluated.empty:
        summary = {"n_rows": 0, "n_genes": 0}
    else:
        by_gene = evaluated.groupby("gene_symbol").agg(
            orientation_correct=("orientation_correct", "mean"),
            state_correct=("state_correct", "mean"),
        )
        summary = {
            "n_rows": int(len(evaluated)),
            "n_genes": int(evaluated["gene_symbol"].nunique()),
            "row_orientation_accuracy": float(
                evaluated["orientation_correct"].mean()
            ),
            "row_state_accuracy": float(evaluated["state_correct"].mean()),
            "gene_macro_orientation_accuracy": float(
                by_gene["orientation_correct"].mean()
            ),
            "gene_macro_state_accuracy": float(by_gene["state_correct"].mean()),
            "families": {
                family: {
                    "n_rows": int(len(group)),
                    "n_genes": int(group["gene_symbol"].nunique()),
                    "orientation_accuracy": float(
                        group["orientation_correct"].mean()
                    ),
                    "state_accuracy": float(group["state_correct"].mean()),
                }
                for family, group in evaluated.groupby("family_name")
            },
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
