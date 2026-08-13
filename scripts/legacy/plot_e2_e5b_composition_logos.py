#!/usr/bin/env python
"""4-panel logo comparison: Ground truth | E2 | E5b | Composition."""

import json
import os
import sys

import logomaker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from evaluate_v19_prediction_composition import compose, load_predictions, active_positions
from tfscope.models.alignment import align_pwm

BASES = list("ACGT")
COLORS = {"A": "#1a9641", "C": "#2b83ba", "G": "#fdae61", "T": "#d7191c"}

IC_THRESH = 0.25
MIN_POS = 4
MAX_SHIFT = 10


def trimmed_core(target, mask):
    cols = target[:, mask.astype(bool)]
    ic = 2.0 + (np.clip(cols, 1e-8, 1.0) * np.log2(np.clip(cols, 1e-8, 1.0))).sum(axis=0)
    keep = ic >= IC_THRESH
    if keep.sum() < MIN_POS:
        return None
    return cols[:, keep]


def information_content(pwm):
    p = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(axis=0)


def logo_df(pwm):
    p = np.clip(pwm, 1e-8, 1.0)
    p = p / p.sum(axis=0, keepdims=True)
    ic = information_content(p)
    return pd.DataFrame((p * ic[None, :]).T, columns=BASES)


def draw_logo(ax, pwm, title, color):
    df = logo_df(pwm)
    logomaker.Logo(
        df, ax=ax,
        color_scheme=COLORS,
        show_spines=False,
        fade_probabilities=False,
        font_name="DejaVu Sans",
    )
    ax.set_ylim(0, 2.1)
    ax.set_yticks([0, 1, 2])
    ax.tick_params(axis="both", labelsize=7, length=0)
    ax.set_title(title, fontsize=8, color=color, pad=2)
    ax.set_ylabel("bits", fontsize=7)


def panel_r(pred, core):
    """Oracle-aligned Pearson r."""
    aligned, _, _, _ = align_pwm(
        pred, core, max_shift=MAX_SHIFT, consider_revcomp=True, min_overlap=MIN_POS
    )
    n = aligned.size
    a = aligned.ravel() - aligned.mean()
    b = core.ravel() - core.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def canon_r(pred_full, core):
    """Fixed-frame Pearson r (no alignment freedom)."""
    ncols = core.shape[1]
    pred_crop = pred_full[:, :ncols]
    if pred_crop.shape[1] < ncols:
        return 0.0
    a = pred_crop.ravel() - pred_crop.mean()
    b = core.ravel() - core.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class _Args:
    ic_thresh = IC_THRESH
    min_positions = MIN_POS
    max_shift = MAX_SHIFT


def main():
    args = _Args()

    e2 = load_predictions("results/v19_e9_model_composition/e2_test_predictions.npz")
    e5b = load_predictions("results/v19_e9_model_composition/e5b_test_predictions.npz")

    with open("results/v19_e9_model_composition/validation_composition_grid.json") as f:
        policy = json.load(f)["family_policy"]

    sel_csv = pd.read_csv("results/v19_e9_publication/logo_selection.csv")
    selected_files = list(sel_csv["filename"])

    rows = []
    for i, fname in enumerate(e2["filename"]):
        if str(fname) in selected_files:
            rows.append(i)

    # also collect MAE table while we're here
    print("\n=== MAE comparison (gene-macro, test set) ===")
    print(f"{'Metric':<20} {'E2':>10} {'E5b':>10} {'Composition':>12}")
    print("-" * 56)

    mae_data = {
        "E2":          {"panel_mae": 0.2096, "fixed_mae": 1.1440, "oracle_mae_4x": 0.8383},
        "E5b":         {"panel_mae": 0.2172, "fixed_mae": 1.1444, "oracle_mae_4x": 0.8687},
        "Composition": {"panel_mae": None,    "fixed_mae": 1.1168, "oracle_mae_4x": 0.8361},
    }
    for metric in ["oracle_mae_4x", "fixed_mae"]:
        print(f"{metric:<20} {mae_data['E2'][metric]:>10.4f} {mae_data['E5b'][metric]:>10.4f}  {mae_data['Composition'][metric] if mae_data['Composition'][metric] else 'N/A':>10}")

    print()

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 2.5 * n_rows), squeeze=False)

    col_labels = ["Ground truth", "E2 (frame model)", "E5b (content model)", "Composition"]
    col_colors = ["#222222", "#2166ac", "#4dac26", "#b2182b"]

    for row_idx, data_idx in enumerate(rows):
        family = str(e2["family"][data_idx])
        gene   = str(e2["gene"][data_idx])
        alpha  = float(policy[family])

        target = e2["target"][data_idx]
        mask   = e2["mask"][data_idx]
        core   = trimmed_core(target, mask)
        if core is None:
            continue

        e2_pred  = e2["prediction"][data_idx]
        e5b_pred = e5b["prediction"][data_idx]
        comp_pred = compose(
            e2_pred, e2["gate"][data_idx],
            e5b_pred, e5b["gate"][data_idx],
            alpha, args,
        )

        e2_active  = e2_pred[:, mask.astype(bool)]
        e5b_active = e5b_pred[:, mask.astype(bool)]
        comp_active = comp_pred[:, mask.astype(bool)]

        pr_e2   = panel_r(e2_active, core)
        pr_e5b  = panel_r(e5b_active, core)
        pr_comp = panel_r(comp_active, core)

        cr_e2   = canon_r(e2_active, core)
        cr_e5b  = canon_r(e5b_active, core)
        cr_comp = canon_r(comp_active, core)

        row_label = f"{gene} [{family}]  α={alpha:.2f}"

        # ground truth
        draw_logo(axes[row_idx, 0], core,
                  f"{row_label}\nGround truth  (L={core.shape[1]})",
                  col_colors[0])

        draw_logo(axes[row_idx, 1], e2_active,
                  f"E2\npanel-r={pr_e2:.3f}  canon-r={cr_e2:.3f}",
                  col_colors[1])

        draw_logo(axes[row_idx, 2], e5b_active,
                  f"E5b\npanel-r={pr_e5b:.3f}  canon-r={cr_e5b:.3f}",
                  col_colors[2])

        draw_logo(axes[row_idx, 3], comp_active,
                  f"Composition\npanel-r={pr_comp:.3f}  canon-r={cr_comp:.3f}",
                  col_colors[3])

        for c in range(4):
            axes[row_idx, c].axhline(0, color="black", linewidth=0.5)

    for c, (label, color) in enumerate(zip(col_labels, col_colors)):
        axes[0, c].set_title(
            f"[{label}]\n" + axes[0, c].get_title(),
            fontsize=9, color=color, pad=3,
        )

    fig.suptitle(
        "TFScope V19 seed-42: Ground truth | E2 | E5b | Composition\n"
        "One test gene per changed family (nearest median gene-level panel-r effect)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = "results/v19_e9_publication/comparison_gt_e2_e5b_composition.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
