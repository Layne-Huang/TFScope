#!/usr/bin/env python
"""Build clean train-only absolute registration anchors from DeepPBS structures."""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

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
        "--assembly",
        default="/data1/leihuang/DeepPBS/deeppbsmar24/data/assembly2024",
    )
    parser.add_argument(
        "--out",
        default="results/v19_e5_registration/structural_anchors_train.tsv",
    )
    parser.add_argument("--min-r", type=float, default=0.90)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--max-shift", type=int, default=10)
    parser.add_argument("--min-overlap", type=int, default=4)
    return parser.parse_args()


def parquet_key(row):
    match = re.match(r"^([0-9A-Za-z]{4})_([A-Za-z0-9]+)_", row["filename"])
    if not match:
        return None
    return match.group(1).lower(), match.group(2), str(row["source_id"])


def npz_key(filename):
    stem = os.path.basename(filename).removesuffix(".npz")
    parts = stem.split("_", 2)
    if len(parts) < 3:
        return None
    pdb_id, chain, suffix = parts
    match = re.search(r"(MA\d+\.\d+)\.jaspar$", suffix)
    if match:
        source_id = match.group(1)
    else:
        match = re.search(
            r"([A-Z0-9][A-Z0-9\-]*_(?:HUMAN|MOUSE)\.H11MO\.\d+\.[A-Z])$",
            suffix,
        )
        if not match:
            return None
        source_id = match.group(1)
    return pdb_id.lower(), chain, source_id


def load_forward_structural_pwm(path):
    data = np.load(path, allow_pickle=True)
    pwm = np.asarray(data["Y_pwm"][0][data["pwm_mask"][0]], dtype=np.float32).T
    pwm = np.clip(pwm, 1e-8, None)
    return pwm / pwm.sum(axis=0, keepdims=True)


def main():
    args = parse_args()
    if not os.path.isdir(args.assembly):
        raise FileNotFoundError(f"DeepPBS assembly directory not found: {args.assembly}")

    df = pd.read_parquet(args.data)
    with open(args.split) as handle:
        split = json.load(handle)
    train_filenames = set(split["train"])
    df = df[df["filename"].isin(train_filenames)].copy()

    candidates = {}
    for name in sorted(os.listdir(args.assembly)):
        if not name.endswith(".npz"):
            continue
        key = npz_key(name)
        if key is not None:
            candidates.setdefault(key, []).append(os.path.join(args.assembly, name))

    direct_references = {}
    mapped_rows = 0
    for _, row in df.iterrows():
        key = parquet_key(row)
        if key is None or key not in candidates:
            continue
        mapped_rows += 1
        target = decode_pwm(row["pwm"])
        best = None
        for path in candidates[key]:
            structural_pwm = load_forward_structural_pwm(path)
            alignment = align_pair(
                structural_pwm,
                target,
                max_shift=args.max_shift,
                min_overlap=min(args.min_overlap, target.shape[1]),
            )
            candidate = (
                alignment["aligned_r"] * alignment["coverage"],
                alignment["aligned_r"],
                alignment["coverage"],
                path,
                structural_pwm,
                alignment,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        _, aligned_r, coverage, path, structural_pwm, alignment = best
        if aligned_r < args.min_r or coverage < args.min_coverage:
            continue
        direct_references.setdefault(str(row["gene_symbol"]), []).append(
            (path, structural_pwm)
        )

    anchors = []
    for _, row in df.iterrows():
        references = direct_references.get(str(row["gene_symbol"]), [])
        if not references:
            continue
        target = decode_pwm(row["pwm"])
        best = None
        for path, structural_pwm in references:
            alignment = align_pair(
                structural_pwm,
                target,
                max_shift=args.max_shift,
                min_overlap=min(args.min_overlap, target.shape[1]),
            )
            candidate = (
                alignment["aligned_r"] * alignment["coverage"],
                alignment["aligned_r"],
                alignment["coverage"],
                path,
                structural_pwm,
                alignment,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        _, aligned_r, coverage, path, structural_pwm, alignment = best
        if aligned_r < args.min_r or coverage < args.min_coverage:
            continue
        exact_structure_record = parquet_key(row) == npz_key(path)
        anchors.append(
            {
                "filename": row["filename"],
                "gene_symbol": row["gene_symbol"],
                "family_name": row["family_name"],
                "split": "train",
                "source_id": row["source_id"],
                "structural_npz": path,
                "orientation_to_reference": alignment["orientation"],
                "offset_to_reference": alignment["shift"],
                "aligned_r": aligned_r,
                "overlap": alignment["overlap"],
                "coverage": coverage,
                "target_length": target.shape[1],
                "structural_pwm_length": structural_pwm.shape[1],
                "anchor_type": (
                    "deeppbs_forward_structural_frame"
                    if exact_structure_record
                    else "same_gene_to_deeppbs_structural_frame"
                ),
                "absolute_orientation_resolved": True,
            }
        )

    anchors_df = pd.DataFrame(anchors).sort_values(
        ["gene_symbol", "filename"]
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    anchors_df.to_csv(args.out, sep="\t", index=False)
    summary = {
        "data": args.data,
        "split": args.split,
        "assembly": args.assembly,
        "protocol": (
            "Map exact PDB-chain-source records to DeepPBS assembly entries; "
            "use Y_pwm strand 0 as the absolute structure-oriented reference; "
            "align the canonical parquet PWM into that frame; retain high-r, "
            "high-coverage clean-training matches only."
        ),
        "thresholds": {
            "min_r": args.min_r,
            "min_coverage": args.min_coverage,
            "max_shift": args.max_shift,
            "min_overlap": args.min_overlap,
        },
        "mapped_train_rows": mapped_rows,
        "direct_structural_genes": len(direct_references),
        "anchor_rows": int(len(anchors_df)),
        "anchor_genes": int(anchors_df["gene_symbol"].nunique())
        if len(anchors_df)
        else 0,
        "orientation_counts": anchors_df["orientation_to_reference"]
        .value_counts()
        .to_dict()
        if len(anchors_df)
        else {},
        "output": args.out,
    }
    summary_path = os.path.join(os.path.dirname(args.out), "anchor_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
