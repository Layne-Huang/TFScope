#!/usr/bin/env python
"""Aggregate the 3-seed v22 diagnostic JSONs into mean/std summaries.

Reports both the covR variants (from each run's `metrics` block) AND
gate-length prediction accuracy (computed here from the per-row `predicted_gate`
fields, since length is not in `metrics`): mean predicted/GT length, length MAE,
signed length bias, and coverage -- overall and on the long-motif stratum
(GT core >= LONG_CORE), which is where v20 truncated.
"""
import argparse
import glob
import json
import os

import numpy as np

LONG_CORE = 13


def length_stats(rows, variant="predicted_gate"):
    """Per-run gate-length metrics from the diagnostic rows."""
    lp = np.array([r[variant]["len_pred"] for r in rows], dtype=float)
    lg = np.array([r[variant]["len_gt"] for r in rows], dtype=float)
    cov = np.array([r[variant]["coverage"] for r in rows], dtype=float)

    def block(mask):
        if mask.sum() == 0:
            return {k: float("nan") for k in
                    ("n", "len_pred", "len_gt", "len_mae", "len_bias", "coverage")}
        return {
            "n": int(mask.sum()),
            "len_pred": float(lp[mask].mean()),
            "len_gt": float(lg[mask].mean()),
            "len_mae": float(np.abs(lp - lg)[mask].mean()),
            "len_bias": float((lp - lg)[mask].mean()),
            "coverage": float(cov[mask].mean()),
        }

    return {"overall": block(np.ones_like(lg, dtype=bool)),
            "long_motif": block(lg >= LONG_CORE)}


def agg_scalar(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": [float(v) for v in values],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/v22_ablation")
    parser.add_argument("--out", default="results/v22_ablation/summary.json")
    args = parser.parse_args()

    runs = {}
    for path in glob.glob(os.path.join(args.results, "*_seed*.json")):
        base = os.path.basename(path)
        if "smoke" in base:          # skip smoke-test artifacts
            continue
        name = base.split("_seed", 1)[0]
        with open(path) as handle:
            runs.setdefault(name, []).append(json.load(handle))

    summary = {}
    for stage, records in sorted(runs.items()):
        summary[stage] = {"n_seeds": len(records), "metrics": {}, "length": {}}
        keys = records[0]["metrics"]
        for variant in keys:
            summary[stage]["metrics"][variant] = {}
            for aggregation in ("row_mean", "gene_balanced_mean"):
                values = [
                    record["metrics"][variant][aggregation] for record in records
                ]
                summary[stage]["metrics"][variant][aggregation] = agg_scalar(values)

        # ── gate-length prediction accuracy (from per-row predicted_gate) ──
        per_run = [length_stats(record["rows"]) for record in records]
        for stratum in ("overall", "long_motif"):
            summary[stage]["length"][stratum] = {
                field: agg_scalar([run[stratum][field] for run in per_run])
                for field in ("len_pred", "len_gt", "len_mae", "len_bias", "coverage")
            }

    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2)

    # ── compact printed tables ──
    def cell(d):
        return f"{d['mean']:.3f}" + (f"±{d['std']:.3f}" if d["std"] else "")

    print("\n=== covR (gene-balanced) by stage ===")
    print(f"{'stage':<12}{'seeds':>6}{'pred_gate':>16}{'gt_span':>16}"
          f"{'fixed_10bp':>16}{'oracle_len':>16}")
    for stage, s in sorted(summary.items()):
        m = s["metrics"]
        print(f"{stage:<12}{s['n_seeds']:>6}"
              + "".join(f"{cell(m[v]['gene_balanced_mean']):>16}" for v in
                        ("predicted_gate", "gt_span", "fixed_10bp", "oracle_length_window")))

    print("\n=== gate-length prediction by stage (predicted_gate) ===")
    print(f"{'stage':<12}{'len_pred':>13}{'len_gt':>9}{'MAE':>12}{'bias':>13}{'cover':>13}"
          f"   | long-motif MAE / cover")
    for stage, s in sorted(summary.items()):
        o = s["length"]["overall"]; lm = s["length"]["long_motif"]
        print(f"{stage:<12}{cell(o['len_pred']):>13}{o['len_gt']['mean']:>9.1f}"
              f"{cell(o['len_mae']):>12}{cell(o['len_bias']):>13}{cell(o['coverage']):>13}"
              f"   | {cell(lm['len_mae'])} / {cell(lm['coverage'])}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
