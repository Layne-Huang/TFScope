#!/usr/bin/env python
"""Summarize paired V19 E1 no-RAG/RAG metrics across seeds."""

import argparse
import json
import os

import numpy as np


PRIMARY_METRICS = ("gate_r", "canon_r", "mae", "top1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="checkpoints/v19_e1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "values": values.tolist(),
    }


def main():
    args = parse_args()
    runs = {}
    for mode in ("norag", "rag"):
        runs[mode] = {}
        for seed in args.seeds:
            path = os.path.join(args.base, f"{mode}_seed{seed}", "test_metrics.json")
            with open(path) as handle:
                runs[mode][seed] = json.load(handle)

    summary = {"seeds": args.seeds, "modes": {}, "paired_rag_minus_norag": {}}
    for mode in ("norag", "rag"):
        summary["modes"][mode] = {
            metric: mean_std(
                [runs[mode][seed]["gene_macro"][metric] for seed in args.seeds]
            )
            for metric in PRIMARY_METRICS
        }

    for metric in PRIMARY_METRICS:
        deltas = [
            runs["rag"][seed]["gene_macro"][metric]
            - runs["norag"][seed]["gene_macro"][metric]
            for seed in args.seeds
        ]
        summary["paired_rag_minus_norag"][metric] = mean_std(deltas)

    families = sorted(
        set.intersection(
            *[
                set(runs[mode][seed]["per_family_gene_macro"])
                for mode in ("norag", "rag")
                for seed in args.seeds
            ]
        )
    )
    summary["per_family_gate_r"] = {}
    for family in families:
        summary["per_family_gate_r"][family] = {
            mode: mean_std(
                [
                    runs[mode][seed]["per_family_gene_macro"][family]["gate_r"]
                    for seed in args.seeds
                ]
            )
            for mode in ("norag", "rag")
        }

    output_path = args.out or os.path.join(args.base, "summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
