#!/usr/bin/env python
"""Validation-select a target-free E2-frame / E5b-content PWM composition."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")

from eval_full_metrics import aligned_cols, canon_fixed_r, panel, trimmed_core
from evaluate_v19_e1 import aggregate, fixed_deeppbs_mae
from tfscope.models.alignment import align_pwm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-predictions", required=True)
    parser.add_argument("--content-predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--policy-json",
        default=None,
        help="Apply a previously validation-locked family_policy without reselection.",
    )
    parser.add_argument(
        "--alphas",
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated content-model blend weights.",
    )
    parser.add_argument("--ic-thresh", type=float, default=0.25)
    parser.add_argument("--min-positions", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=10)
    return parser.parse_args()


def load_predictions(path):
    return dict(np.load(path, allow_pickle=False))


def active_positions(gate):
    active = gate > 0.5
    if not active.any():
        active = gate > gate.max() * 0.5
    return active


def compose(frame_pwm, frame_gate, content_pwm, content_gate, alpha, args):
    frame_active = active_positions(frame_gate)
    content_active = active_positions(content_gate)
    if frame_active.sum() < args.min_positions or content_active.sum() < args.min_positions:
        return frame_pwm.copy()

    aligned, _, _, score = align_pwm(
        content_pwm[:, content_active],
        frame_pwm[:, frame_active],
        max_shift=args.max_shift,
        consider_revcomp=True,
        min_overlap=args.min_positions,
    )
    if score <= -1.5:
        return frame_pwm.copy()

    result = frame_pwm.copy()
    result[:, frame_active] = (
        (1.0 - alpha) * frame_pwm[:, frame_active] + alpha * aligned
    )
    result /= result.sum(axis=0, keepdims=True).clip(1e-8)
    return result


def score_predictions(frame, content, alpha, args):
    rows = []
    for index in range(len(frame["filename"])):
        target = frame["target"][index]
        mask = frame["mask"][index]
        core = trimmed_core(target, mask, args.ic_thresh)
        if core is None or core.shape[1] < args.min_positions:
            continue

        row_alpha = (
            alpha[str(frame["family"][index])]
            if isinstance(alpha, dict)
            else alpha
        )
        prediction = compose(
            frame["prediction"][index],
            frame["gate"][index],
            content["prediction"][index],
            content["gate"][index],
            row_alpha,
            args,
        )
        target_valid = target[:, mask.astype(bool)]
        prediction_panel = prediction[:, mask.astype(bool)]
        aligned, columns, _ = aligned_cols(
            prediction_panel,
            core,
            args.max_shift,
            min_overlap=args.min_positions,
        )
        metrics = panel(core, aligned, columns)
        if metrics is None:
            continue
        rows.append(
            {
                "gene": str(frame["gene"][index]),
                "family": str(frame["family"][index]),
                "gate_r": float("nan"),
                "panel_r": float(metrics["r"]),
                "canon_r": float(canon_fixed_r(core, prediction_panel)),
                "mae": float(4.0 * metrics["mae"]),
                "panel_mae": float(metrics["mae"]),
                "fixed_mae": fixed_deeppbs_mae(prediction_panel, target_valid),
                **{
                    key: float(metrics[key])
                    for key in ("rmse", "ce", "kl", "top1", "auc", "f1", "mcc")
                },
            }
        )
    gene_macro, per_gene = aggregate(rows, "gene")
    row_macro = aggregate(rows)
    per_family = {}
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        family_macro, family_genes = aggregate(family_rows, "gene")
        per_family[family] = {
            "n_genes": len(family_genes),
            **family_macro,
        }
    return {
        "alpha": alpha,
        "n_rows": len(rows),
        "gene_macro": gene_macro,
        "row_macro": row_macro,
        "per_family_gene_macro": per_family,
        "per_gene": per_gene,
    }


def main():
    args = parse_args()
    frame = load_predictions(args.frame_predictions)
    content = load_predictions(args.content_predictions)
    if not np.array_equal(frame["filename"], content["filename"]):
        raise ValueError("Prediction files have different row order or examples")

    if args.policy_json:
        with open(args.policy_json) as handle:
            locked = json.load(handle)
        family_policy = locked["family_policy"]
        output = {
            "policy_source": args.policy_json,
            "family_policy": family_policy,
            "metrics": score_predictions(frame, content, family_policy, args),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(output, handle, indent=2, allow_nan=True)
        print(json.dumps(output, indent=2, allow_nan=True))
        return

    results = [
        score_predictions(frame, content, float(alpha), args)
        for alpha in args.alphas.split(",")
    ]
    selected = max(results, key=lambda result: result["gene_macro"]["panel_r"])
    baseline = results[0]
    family_policy = {}
    for family, baseline_metrics in baseline["per_family_gene_macro"].items():
        candidates = [
            result
            for result in results
            if (
                result["per_family_gene_macro"][family]["canon_r"]
                >= baseline_metrics["canon_r"] - 0.02
                and result["per_family_gene_macro"][family]["fixed_mae"]
                <= baseline_metrics["fixed_mae"] + 0.02
            )
        ]
        if baseline_metrics["n_genes"] < 10:
            candidates = [baseline]
        family_policy[family] = max(
            candidates,
            key=lambda result: result["per_family_gene_macro"][family]["panel_r"],
        )["alpha"]
    output = {
        "selection_metric": "validation gene-macro panel_r",
        "selected_alpha": selected["alpha"],
        "family_policy": family_policy,
        "family_policy_constraints": {
            "minimum_validation_genes": 10,
            "maximum_canon_r_regression": 0.02,
            "maximum_fixed_mae_increase": 0.02,
        },
        "family_policy_validation": score_predictions(
            frame, content, family_policy, args
        ),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2, allow_nan=True)
    print(json.dumps(output, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
