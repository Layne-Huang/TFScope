#!/usr/bin/env python
"""Build final seed-42 tables and logo figures for the frozen E9 composition."""

import argparse
import json
import os
import sys

import logomaker
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from eval_full_metrics import aligned_cols, canon_fixed_r, panel, trimmed_core
from evaluate_v19_e1 import fixed_deeppbs_mae
from evaluate_v19_prediction_composition import compose, load_predictions
from tfscope.models.alignment import align_pwm


BASES = list("ACGT")
LOWER_IS_BETTER = {"mae", "fixed_mae", "rmse", "ce", "kl"}
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-predictions", required=True)
    parser.add_argument("--content-predictions", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--permutation-replicates", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ic-thresh", type=float, default=0.25)
    parser.add_argument("--min-positions", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=10)
    return parser.parse_args()


def score_row(prediction, target, mask, args):
    core = trimmed_core(target, mask, args.ic_thresh)
    if core is None or core.shape[1] < args.min_positions:
        return None
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
        return None
    return {
        "panel_r": float(metrics["r"]),
        "canon_r": float(canon_fixed_r(core, prediction_panel)),
        "mae": float(4.0 * metrics["mae"]),
        "fixed_mae": fixed_deeppbs_mae(prediction_panel, target_valid),
        **{
            metric: float(metrics[metric])
            for metric in ("rmse", "ce", "kl", "top1", "auc", "f1", "mcc")
        },
    }


def gene_macro(rows):
    frame = pd.DataFrame(rows)
    return frame.groupby("gene", sort=False)[list(METRICS)].mean()


def paired_statistics(
    baseline,
    candidate,
    bootstrap_replicates,
    permutation_replicates,
    rng,
):
    output = {}
    for metric in METRICS:
        paired = pd.concat(
            [baseline[metric], candidate[metric]], axis=1, keys=["base", "candidate"]
        ).dropna()
        differences = (paired["candidate"] - paired["base"]).to_numpy()
        indices = rng.integers(
            0,
            len(differences),
            size=(bootstrap_replicates, len(differences)),
        )
        boot = differences[indices].mean(axis=1)
        observed = abs(float(differences.mean()))
        exceedances = 0
        remaining = permutation_replicates
        while remaining:
            block = min(10000, remaining)
            signs = rng.choice(
                np.array([-1.0, 1.0]),
                size=(block, len(differences)),
            )
            permuted = (signs * differences).mean(axis=1)
            exceedances += int((np.abs(permuted) >= observed).sum())
            remaining -= block
        output[metric] = {
            "n_genes": int(len(differences)),
            "baseline": float(paired["base"].mean()),
            "composition": float(paired["candidate"].mean()),
            "delta": float(differences.mean()),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "probability_improvement": float(
                (boot < 0).mean()
                if metric in LOWER_IS_BETTER
                else (boot > 0).mean()
            ),
            "paired_permutation_p_two_sided": float(
                (exceedances + 1) / (permutation_replicates + 1)
            ),
        }
    return output


def information_content(pwm):
    pwm = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (pwm * np.log2(pwm)).sum(axis=0)


def logo_frame(pwm):
    pwm = np.clip(pwm, 1e-8, 1.0)
    pwm = pwm / pwm.sum(axis=0, keepdims=True).clip(1e-8)
    return pd.DataFrame(
        (pwm * information_content(pwm)[None, :]).T,
        columns=BASES,
    )


def draw_logo(axis, pwm, title, color):
    logomaker.Logo(
        logo_frame(pwm),
        ax=axis,
        color_scheme={"A": "green", "C": "blue", "G": "#FBA922", "T": "red"},
        show_spines=False,
        fade_probabilities=False,
        font_name="DejaVu Sans",
    )
    axis.set_ylim(0, 2.05)
    axis.set_yticks([0, 1, 2])
    axis.tick_params(axis="both", labelsize=7, length=0)
    axis.set_title(title, fontsize=9, color=color, pad=3)
    axis.set_xlabel("")
    axis.set_ylabel("bits", fontsize=7)


def aligned_for_logo(prediction, target, mask, args):
    core = trimmed_core(target, mask, args.ic_thresh)
    prediction_panel = prediction[:, mask.astype(bool)]
    aligned, _, _, score = align_pwm(
        prediction_panel,
        core,
        max_shift=args.max_shift,
        consider_revcomp=True,
        min_overlap=args.min_positions,
    )
    return core, aligned, float(score)


def make_logo_figure(selected, records, out_dir, args, aligned):
    figure, axes = plt.subplots(
        len(selected), 3, figsize=(14, 2.25 * len(selected)), squeeze=False
    )
    for row_index, record_index in enumerate(selected):
        record = records[record_index]
        target = record["target"]
        mask = record["mask"]
        if aligned:
            target_pwm, baseline_pwm, baseline_r = aligned_for_logo(
                record["baseline_pwm"], target, mask, args
            )
            _, candidate_pwm, candidate_r = aligned_for_logo(
                record["candidate_pwm"], target, mask, args
            )
        else:
            target_pwm = target[:, mask.astype(bool)]
            baseline_pwm = record["baseline_pwm"][:, mask.astype(bool)]
            candidate_pwm = record["candidate_pwm"][:, mask.astype(bool)]
            baseline_r = record["baseline"]["canon_r"]
            candidate_r = record["candidate"]["canon_r"]

        prefix = (
            f"{record['gene']} [{record['family']}] "
            f"alpha={record['alpha']:.2f}"
        )
        draw_logo(axes[row_index, 0], target_pwm, f"{prefix}\nGround truth", "#222222")
        draw_logo(
            axes[row_index, 1],
            baseline_pwm,
            f"E2: {'aligned r' if aligned else 'canon r'}={baseline_r:.3f}",
            "#2166ac",
        )
        draw_logo(
            axes[row_index, 2],
            candidate_pwm,
            f"Composition: {'aligned r' if aligned else 'canon r'}={candidate_r:.3f}\n"
            f"delta panel-r={record['delta_panel_r']:+.3f}",
            "#b2182b",
        )

    frame_label = "evaluation-aligned" if aligned else "fixed target frame"
    figure.suptitle(
        "TFScope V19 seed 42 frozen E2/E5b composition\n"
        f"Median-effect test gene per changed family, {frame_label}",
        fontsize=14,
        fontweight="bold",
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    stem = "representative_logos_aligned" if aligned else "representative_logos_fixed"
    for extension, kwargs in (("pdf", {}), ("png", {"dpi": 200})):
        figure.savefig(
            os.path.join(out_dir, f"{stem}.{extension}"),
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(figure)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    baseline = load_predictions(args.frame_predictions)
    content = load_predictions(args.content_predictions)
    if not np.array_equal(baseline["filename"], content["filename"]):
        raise ValueError("Prediction files have different examples or row order")
    with open(args.policy_json) as handle:
        policy = json.load(handle)["family_policy"]

    records = []
    baseline_rows = []
    candidate_rows = []
    for index in range(len(baseline["filename"])):
        family = str(baseline["family"][index])
        alpha = float(policy[family])
        candidate_pwm = compose(
            baseline["prediction"][index],
            baseline["gate"][index],
            content["prediction"][index],
            content["gate"][index],
            alpha,
            args,
        )
        base_metrics = score_row(
            baseline["prediction"][index],
            baseline["target"][index],
            baseline["mask"][index],
            args,
        )
        candidate_metrics = score_row(
            candidate_pwm,
            baseline["target"][index],
            baseline["mask"][index],
            args,
        )
        if base_metrics is None or candidate_metrics is None:
            continue
        identity = {
            "filename": str(baseline["filename"][index]),
            "gene": str(baseline["gene"][index]),
            "family": family,
            "alpha": alpha,
        }
        baseline_rows.append({**identity, **base_metrics})
        candidate_rows.append({**identity, **candidate_metrics})
        records.append(
            {
                **identity,
                "target": baseline["target"][index],
                "mask": baseline["mask"][index],
                "baseline_pwm": baseline["prediction"][index],
                "candidate_pwm": candidate_pwm,
                "baseline": base_metrics,
                "candidate": candidate_metrics,
                "delta_panel_r": candidate_metrics["panel_r"]
                - base_metrics["panel_r"],
            }
        )

    baseline_gene = gene_macro(baseline_rows)
    candidate_gene = gene_macro(candidate_rows)
    rng = np.random.default_rng(args.seed)
    overall = paired_statistics(
        baseline_gene,
        candidate_gene,
        args.bootstrap_replicates,
        args.permutation_replicates,
        rng,
    )

    family_results = {}
    families = sorted({row["family"] for row in baseline_rows})
    for family in families:
        family_baseline = gene_macro(
            [row for row in baseline_rows if row["family"] == family]
        )
        family_candidate = gene_macro(
            [row for row in candidate_rows if row["family"] == family]
        )
        family_results[family] = paired_statistics(
            family_baseline,
            family_candidate,
            args.bootstrap_replicates,
            args.permutation_replicates,
            rng,
        )

    overall_rows = [
        {"metric": metric, **values} for metric, values in overall.items()
    ]
    pd.DataFrame(overall_rows).to_csv(
        os.path.join(args.out_dir, "overall_comparison.csv"), index=False
    )
    family_rows = []
    for family, metrics in family_results.items():
        for metric, values in metrics.items():
            family_rows.append({"family": family, "metric": metric, **values})
    pd.DataFrame(family_rows).to_csv(
        os.path.join(args.out_dir, "family_comparison.csv"), index=False
    )

    per_record = []
    for baseline_row, candidate_row in zip(baseline_rows, candidate_rows):
        row = {
            key: baseline_row[key]
            for key in ("filename", "gene", "family", "alpha")
        }
        for metric in METRICS:
            row[f"e2_{metric}"] = baseline_row[metric]
            row[f"composition_{metric}"] = candidate_row[metric]
            row[f"delta_{metric}"] = candidate_row[metric] - baseline_row[metric]
        per_record.append(row)
    pd.DataFrame(per_record).to_csv(
        os.path.join(args.out_dir, "per_record_comparison.csv"), index=False
    )

    changed_families = [family for family in families if float(policy[family]) > 0]
    selected = []
    selection_rows = []
    for family in changed_families:
        family_indices = [
            index for index, record in enumerate(records) if record["family"] == family
        ]
        gene_effects = {}
        for index in family_indices:
            gene_effects.setdefault(records[index]["gene"], []).append(
                records[index]["delta_panel_r"]
            )
        gene_means = {
            gene: float(np.mean(values)) for gene, values in gene_effects.items()
        }
        median_effect = float(np.median(list(gene_means.values())))
        selected_gene = min(
            gene_means, key=lambda gene: abs(gene_means[gene] - median_effect)
        )
        selected_index = min(
            (index for index in family_indices if records[index]["gene"] == selected_gene),
            key=lambda index: abs(
                records[index]["delta_panel_r"] - gene_means[selected_gene]
            ),
        )
        selected.append(selected_index)
        record = records[selected_index]
        selection_rows.append(
            {
                "family": family,
                "gene": selected_gene,
                "filename": record["filename"],
                "alpha": record["alpha"],
                "family_median_gene_delta_panel_r": median_effect,
                "selected_record_delta_panel_r": record["delta_panel_r"],
            }
        )
    pd.DataFrame(selection_rows).to_csv(
        os.path.join(args.out_dir, "logo_selection.csv"), index=False
    )
    make_logo_figure(selected, records, args.out_dir, args, aligned=True)
    make_logo_figure(selected, records, args.out_dir, args, aligned=False)

    summary = {
        "seed": args.seed,
        "policy_source": args.policy_json,
        "policy": policy,
        "selection": (
            "One test gene per changed family nearest that family's median paired "
            "gene-level panel-r effect; one row nearest the selected gene mean."
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "n_test_rows": len(records),
        "n_test_genes": int(len(baseline_gene)),
        "overall": overall,
        "families": family_results,
    }
    with open(os.path.join(args.out_dir, "publication_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps({"overall": overall, "logo_selection": selection_rows}, indent=2))


if __name__ == "__main__":
    main()
