#!/usr/bin/env python
"""Nature-style three-metric view of the v24 baseline ladder, single model only.

Figure contract
  Core conclusion : on the 20 co-crystal genes a single sequence-only TFScope model
                    matches or beats the structure-based DeepPBS on DeepPBS's own
                    labels, without ever seeing a structure.
  Evidence chain  : a = pearson_r (content agreement), b = auroc (base enrichment),
                    c = mae (per-base calibration). One claim each; the same 7 rungs
                    in the same order so the ladder reads across panels.
  Archetype       : quantitative grid.
  Export contract : 183 mm double-column, 7 pt body / 5 pt floor, editable text
                    (PDF Type 42, SVG fonttype none), PDF + SVG + TIFF + PNG.

Ground truth is DeepPBS's OWN masked label, `Y_pwm[pwm_mask]`, read straight from each
structure file -- the competitor is scored on its home target and TFScope is held to
the same one. Chosen over TFScope's v23 HOCOMOCO targets because those have a median of
12 valid columns against DeepPBS's 9, which inflates DeepPBS's coverage by construction.
The two label sets are the same motifs re-windowed (median overlap r 0.967, >0.9 for
15/20 genes), so this is a change of window, not of biology. Numbers come from
`iclr/eval_on_deeppbs_labels.py`.

The 5-seed ensemble is left out on request: it buys correlation at the cost of motif
sharpness (it loses `mae` and `ic_mae` to this single seed on both label sets), so the
single model is the cleaner reference point.

Bare per request: no panel titles, no "higher/lower is better" labels, no caption
block. Panel letters stay -- in Nature they identify the panel, they are not titles.

  python scripts/build_fig_baseline_ladder_nature.py
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = "results/baseline_ladder/eval_on_deeppbs_labels.json"
# figures_v24/, not figures_v24_ensemble/: per figures_v24/README.md the split is by
# checkpoint, and this figure uses ONLY the v24 seed42 single model. The 12-panel
# ladder, which contrasts seed42 against the 5-seed ensemble, stays on the other side.
OUTD = "figures_v24/figure_baseline_ladder"
STEM = f"{OUTD}/figure_baseline_ladder_nature"
# mae_dpbs / rmse_dpbs, not mae / rmse: DeepPBS sums the absolute error over the
# four bases before averaging over columns (deeppbs/nn/metrics/metrics.py:89),
# so its MAE lives in [0, 2] while ours lives in [0, 0.5] -- exactly 4x apart
# (RMSE exactly 2x). The scale factor changes no ranking, but printing our
# number as plain "MAE" would read 4x better than DeepPBS's published value.
# Since the whole figure is scored on DeepPBS labels, use DeepPBS units too.
METRICS = ["pearson_r", "auroc", "mae_dpbs"]

RUNGS = ["random_uniform", "random_train_pwm", "B0_global", "B0_family",
         "B1_nearest_pwm", "deeppbs", "v24_seed42"]
LABEL = {"random_uniform": "Uniform PWM",
         "random_train_pwm": "Random training PWM",
         "B0_global": "Global training mean",
         "B0_family": "Family mean",
         "B1_nearest_pwm": "Nearest-homologue PWM",
         "deeppbs": "DeepPBS (structure-based)",
         "v24_seed42": "TFScope v24 (sequence-only)"}
ALL_METRICS = ["pearson_r", "cosine", "topbase_acc", "auroc", "macroF1", "covR",
               "mae_dpbs", "rmse_dpbs", "jsd_bits", "kl_bits", "ic_mae", "coverage"]
# both conventions go into Source Data so nothing is lost by the figure's choice
SOURCE_METRICS = ALL_METRICS + ["mae", "rmse"]
HIGHER_BETTER = {"pearson_r": True, "cosine": True, "topbase_acc": True, "auroc": True,
                 "macroF1": True, "covR": True, "coverage": True, "mae_dpbs": False,
                 "rmse_dpbs": False, "jsd_bits": False, "kl_bits": False, "ic_mae": False}
AXIS = {"pearson_r": "Pearson $r$", "auroc": "AUROC", "mae_dpbs": "MAE",
        "cosine": "Cosine similarity", "topbase_acc": "Consensus-base accuracy",
        # plain, not $F_1$: a mathtext subscript renders at 0.7x the base size, i.e.
        # 4.9 pt, which is under Nature's 5 pt glyph floor
        "macroF1": "Macro F1", "covR": "covR", "coverage": "Coverage",
        "rmse_dpbs": "RMSE", "jsd_bits": "JSD (bits)", "kl_bits": "KL (bits)",
        "ic_mae": "IC error (bits)"}

# Palette derived from the architecture schematic (Model.pdf) so figure and schematic
# read as one system, then desaturated for Nature's restraint: the schematic's colours
# are poster-bright, and a quantitative panel should not compete with it.
#   #acc7f0 light blue -> #92A9CE   training-free baselines, recede to background
#   #127961 teal       -> #4A8C7E   DeepPBS, a muted comparator tone
#   #796bbe violet     -> #5C4FAA   TFScope, the one chromatic accent on the page
# Nature uses saturation sparingly and for one thing only: violet is the sole colour
# above the chroma floor, so TFScope carries the salience without being the brightest.
# The five baselines share ONE colour on purpose. A light-to-dark ramp across them was
# tried and fails hard -- five steps inside one hue give adjacent normal-vision dE ~5.6
# (floor 15) and the darkest step collides with the teal (CVD dE 4.4) -- so the ladder
# order is carried by the y-axis order, not by shading.
# Validated on the adjacent pairlist (bars are an adjacent form): lightness band,
# CVD separation (worst adjacent dE 14.3) and normal-vision floor (16.3) all pass.
# The baseline blue and the muted teal fail the CATEGORICAL chroma floor by design --
# this is an emphasis palette (one accent + de-emphasis tones), not a categorical one --
# and the sub-3:1 contrast is covered by the always-visible rung labels.
BASE = "#92A9CE"         # training-free baselines
TEAL = "#4A8C7E"         # DeepPBS
VIOLET = "#5C4FAA"       # TFScope
COLORS = [BASE] * 5 + [TEAL, VIOLET]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans",
                        "DejaVu Sans", "sans-serif"],
    # Without this, mathtext ($r$) silently falls back to DejaVu and the exported PDF
    # embeds two unrelated typefaces.
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.cal": "sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 0.0,
    "legend.frameon": False,
})

MM = 1 / 25.4


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-metrics", action="store_true",
                    help="12-panel version covering the whole metric suite")
    ap.add_argument("--annotate", action="store_true",
                    help="print each bar's value and mark the better direction with an "
                         "arrow; kept off the clean version, where the axis carries the "
                         "value and the Source Data table carries the exact number")
    a = ap.parse_args()
    metrics = ALL_METRICS if a.all_metrics else METRICS
    ncol = 3
    nrow = (len(metrics) + ncol - 1) // ncol
    stem = STEM + ("_full" if a.all_metrics else "") + ("_annotated" if a.annotate else "")

    doc = json.load(open(SRC))
    L = doc["ladder"]

    # 180, not 183: bbox_inches="tight" grows the canvas past the declared figsize, and
    # the trimmed PDF must still land inside the 183 mm double-column limit
    fig, axarr = plt.subplots(nrow, ncol, figsize=(180 * MM, 47 * MM * nrow + 10 * MM),
                              squeeze=False)
    fig.subplots_adjust(left=0.245, right=0.995,
                        top=1 - 0.045 / nrow, bottom=0.205 / nrow,
                        wspace=0.16, hspace=0.62)
    axes = axarr.ravel()
    y = np.arange(len(RUNGS))[::-1]          # best rung at the top

    for k, m in enumerate(metrics):
        ax = axes[k]
        vals = np.array([L[t][m] for t in RUNGS], float)
        ci = np.array([L[t][m + "_ci95"] for t in RUNGS], float)
        err = np.vstack([vals - ci[:, 0], ci[:, 1] - vals])
        ax.barh(y, vals, height=0.68, color=COLORS, linewidth=0, zorder=2)
        ax.errorbar(vals, y, xerr=err, fmt="none", ecolor="#2b2b2b",
                    elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=3)
        # U+2191/2193 are present in Nimbus Sans (checked), so the arrow needs no
        # second font -- a mathtext arrow would risk a DejaVu fallback
        lab = AXIS[m] + ("  \u2191" if HIGHER_BETTER[m] else "  \u2193") if a.annotate \
            else AXIS[m]
        ax.set_xlabel(lab, labelpad=2)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[t] for t in RUNGS] if k % ncol == 0 else [])
        ax.set_ylim(-0.7, len(RUNGS) - 0.3)
        top = max(ci[:, 1].max(), vals.max())
        ax.set_xlim(0, top * (1.34 if a.annotate else 1.06))
        if a.annotate:
            for i, v in zip(y, vals):
                hi = ci[list(y).index(i), 1]
                ax.text(hi + top * 0.035, i, f"{v:.3f}", va="center", ha="left",
                        fontsize=6, color="#2b2b2b")
        ax.tick_params(axis="y", pad=2)
        ax.text(-0.035 if k % ncol == 0 else -0.02, 1.10,
                "abcdefghijkl"[k], transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

    for ax in axes[len(metrics):]:          # unused cells in a partial last row
        ax.set_visible(False)

    os.makedirs(OUTD, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    print(f"saved {stem}.pdf/.svg/.tiff/.png  ({len(metrics)} panels)")

    # Source Data: Nature expects exact values in a flat table, not printed on the bars.
    # Every number any panel plots, plus the CIs the whiskers show.
    src = []
    for t in RUNGS:
        row = {"rung": t, "label": LABEL[t], "n_structures": L[t]["n"]}
        for m in SOURCE_METRICS:
            row[m] = L[t][m]
            row[m + "_ci95_lo"], row[m + "_ci95_hi"] = L[t][m + "_ci95"]
        src.append(row)
    pd.DataFrame(src).to_csv(f"{OUTD}/source_data_ladder.csv", index=False)
    print(f"saved {OUTD}/source_data_ladder.csv  "
          f"({len(RUNGS)} rungs x {len(SOURCE_METRICS)} metrics + 95% CIs; both MAE/RMSE conventions)")
    print(f"  ground truth: {doc['target']}")
    print(f"  n = {doc['n_structures']} structures, label length "
          f"{doc['label_length']['min']}-{doc['label_length']['max']} "
          f"(median {doc['label_length']['median']})")
    for m in metrics:
        print(f"  {m:<13} " + "  ".join(f"{t}={L[t][m]:.3f}"
                                        for t in ("deeppbs", "v24_seed42")))


if __name__ == "__main__":
    main()
