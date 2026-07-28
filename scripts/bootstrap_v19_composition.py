#!/usr/bin/env python
"""Paired gene bootstrap for a validation-locked prediction composition."""

import argparse
import json

import numpy as np


METRICS = (
    "panel_r",
    "canon_r",
    "mae",
    "fixed_mae",
    "rmse",
    "ce",
    "kl",
    "top1",
    "auc",
    "f1",
    "mcc",
)
LOWER_IS_BETTER = {"mae", "fixed_mae", "rmse", "ce", "kl"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--composition", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.baseline) as handle:
        baseline = json.load(handle)["metrics"]["per_gene"]
    with open(args.composition) as handle:
        composition = json.load(handle)["metrics"]["per_gene"]
    genes = sorted(set(baseline).intersection(composition))
    rng = np.random.default_rng(args.seed)

    results = {}
    for metric in METRICS:
        metric_genes = [
            gene
            for gene in genes
            if baseline[gene][metric] is not None
            and composition[gene][metric] is not None
            and np.isfinite(baseline[gene][metric])
            and np.isfinite(composition[gene][metric])
        ]
        differences = np.asarray(
            [
                composition[gene][metric] - baseline[gene][metric]
                for gene in metric_genes
            ],
            dtype=float,
        )
        indices = rng.integers(
            0,
            len(metric_genes),
            size=(args.replicates, len(metric_genes)),
        )
        boot = differences[indices].mean(axis=1)
        results[metric] = {
            "n_paired_genes": len(metric_genes),
            "mean_delta": float(differences.mean()),
            "ci95": [float(value) for value in np.quantile(boot, [0.025, 0.975])],
            "probability_improvement": float(
                (boot < 0).mean()
                if metric in LOWER_IS_BETTER
                else (boot > 0).mean()
            ),
        }

    output = {
        "n_paired_genes": len(genes),
        "replicates": args.replicates,
        "seed": args.seed,
        "delta": "composition minus baseline; lower is better for error metrics",
        "metrics": results,
    }
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
