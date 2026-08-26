#!/usr/bin/env python
"""Figure: full-metric baseline ladder topped by the v24 5-seed ensemble.

Reads `results/baseline_ladder/ladder_v24ens_full.json` and writes a grid of bar
charts -- one panel per metric, the same 7 rungs in the same order in every panel,
95% gene-bootstrap CIs -- into `figures_v24_ensemble/figure_baseline_ladder/`,
together with the source tables.

Keeping the rung order identical across panels is the whole point: it makes the
places where the ranking flips (the ensemble wins r but loses mae and ic_mae)
visible by eye, without any extra apparatus.

  python scripts/build_fig_baseline_ladder_ens.py
"""
from __future__ import annotations

import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = "results/baseline_ladder/ladder_v24ens_full.json"
OUTD = "figures_v24_ensemble/figure_baseline_ladder"
SURFACE = "clean_combined"

# Validated reference palette (dataviz skill), ordinal blue: the lightest step still
# clears 2:1 on the light surface. Rung identity is carried by the shared y-axis
# labels, so colour is never the only encoding.
ORDINAL = ["#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
SURF = "#fcfcfb"

RUNGS = ["random_uniform", "random_train_pwm", "B0_global", "B0_family",
         "B1_nearest_pwm", "v24_seed42", "v24_ens5"]
NICE = {"random_uniform": "random uniform PWM",
        "random_train_pwm": "random real training PWM",
        "B0_global": "mean of all training PWMs",
        "B0_family": "mean of the TF's own family",
        "B1_nearest_pwm": "PWM of closest training TF",
        "v24_seed42": "v24 single model (seed 42)",
        "v24_ens5": "v24 5-seed ensemble"}
CAPABILITY = {"random_uniform": "floor",
              "random_train_pwm": "+ looks like a PWM",
              "B0_global": "+ average motif",
              "B0_family": "+ family identity",
              "B1_nearest_pwm": "+ homology lookup",
              "v24_seed42": "+ learned seq to PWM",
              "v24_ens5": "+ seed averaging"}
PANELS = ["pearson_r", "cosine", "topbase_acc", "auroc", "macroF1", "covR",
          "mae", "rmse", "jsd_bits", "kl_bits", "ic_mae", "coverage"]


def paired_delta(pg, metric, up, n_boot=10000, seed=0,
                 ref="v24_seed42", alt="v24_ens5"):
    """Per-gene (alt - ref), signed so positive = `alt` better; paired bootstrap.

    Paired on identical genes, which is the only way to compare two methods whose
    per-gene difficulty varies as much as it does here: the marginal means alone
    cannot say whether a 0.013 gap is real.
    """
    a = pg[pg.rung == ref].set_index("gene")[metric]
    b = pg[pg.rung == alt].set_index("gene")[metric]
    genes = sorted(set(a.index) & set(b.index))
    d = (b.loc[genes].values - a.loc[genes].values) * (1.0 if up else -1.0)
    d = d[~np.isnan(d)]
    rng = np.random.RandomState(seed)
    boot = np.array([np.mean(rng.choice(d, d.size, replace=True)) for _ in range(n_boot)])
    return float(np.mean(d)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), d.size


def main():
    os.makedirs(OUTD, exist_ok=True)
    doc = json.load(open(SRC))
    L = doc["ladder"][SURFACE]
    direction = dict(doc["metric_direction"])
    direction.setdefault("covR", True)
    direction.setdefault("coverage", True)
    n_genes = doc["surfaces"][SURFACE]["n_genes"]
    n_rows = doc["surfaces"][SURFACE]["n_rows"]

    ncol, nrow = 3, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(14.4, 11.6), facecolor=SURF)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.868, bottom=0.045,
                        hspace=0.55, wspace=0.30)
    y = np.arange(len(RUNGS))

    for k, m in enumerate(PANELS):
        ax = axes[k // ncol][k % ncol]
        ax.set_facecolor(SURF)
        vals = np.array([L[r][m] for r in RUNGS], float)
        ci = np.array([L[r][m + "_ci95"] for r in RUNGS], float)
        err = np.vstack([vals - ci[:, 0], ci[:, 1] - vals])
        ax.barh(y, vals, height=0.66, color=ORDINAL, edgecolor=SURF, linewidth=1.1,
                zorder=2)
        ax.errorbar(vals, y, xerr=err, fmt="none", ecolor=INK2, elinewidth=1.2,
                    capsize=2.5, capthick=1.2, zorder=3)
        top = max(ci[:, 1].max(), vals.max())
        for i, v in enumerate(vals):
            ax.text(ci[i, 1] + top * 0.035, i, f"{v:.3f}", va="center", ha="left",
                    fontsize=8.0, color=INK)
        ax.set_xlim(0, top * 1.30)
        ax.set_yticks(y)
        if k % ncol == 0:
            ax.set_yticklabels([NICE[r] for r in RUNGS], fontsize=8.6, color=INK)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(-0.65, len(RUNGS) - 0.35)
        arrow = "higher is better" if direction[m] else "lower is better"
        ax.set_title(f"{m}   ({arrow})", fontsize=9.8, fontweight="bold", color=INK,
                     loc="left", pad=6)
        ax.grid(axis="x", color="#e4e3df", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#d7d6d1")
        ax.tick_params(axis="both", length=0, colors=INK2, labelsize=8.0)

    fig.suptitle("TFScope v24 baseline ladder, full metric panel  —  clean held-out surface "
                 f"({n_rows} PWMs / {n_genes} genes, never trained on, never selected on)",
                 fontsize=12.2, fontweight="bold", color=INK, x=0.006, ha="left", y=0.977)
    fig.text(0.006, 0.951,
             "Same 7 rungs, same order, in every panel. Each rung adds one capability, so the gap between "
             "two adjacent bars is what that capability buys.\nEvery prediction is registered to the "
             "IC-trimmed ground-truth motif core with the same oracle shift + reverse-complement search "
             "before scoring.\nBars are gene-balanced means; whiskers are 95% gene-level bootstrap CIs.",
             fontsize=8.6, color=INK2, ha="left", va="top")

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUTD}/figure_baseline_ladder.{ext}", dpi=300,
                    bbox_inches="tight", facecolor=SURF)
    print(f"saved {OUTD}/figure_baseline_ladder.pdf/.png")

    # ── source tables next to the figure ──────────────────────────────────────
    shutil.copy(SRC, f"{OUTD}/ladder_v24ens_full.json")
    metrics = PANELS + ["gate_len_mae", "gate_len_bias"]
    tidy = []
    ALL_RUNGS = RUNGS[:5] + ["deeppbs"] + RUNGS[5:]
    CAPABILITY["deeppbs"] = "+ co-crystal structure input"
    for surf, lad in doc["ladder"].items():
        for r in ALL_RUNGS:
            e = lad.get(r)
            if not e:
                continue
            row = {"surface": surf, "rung": r, "capability": CAPABILITY[r],
                   "n_genes": e["n_genes"], "n_rows": e["n_rows"]}
            for m in metrics:
                row[m] = e[m]
                row[m + "_ci_lo"], row[m + "_ci_hi"] = e[m + "_ci95"]
            tidy.append(row)
    pd.DataFrame(tidy).to_csv(f"{OUTD}/ladder_v24ens_full.csv", index=False)

    # the ensemble-vs-single-seed question needs a PAIRED CI on the same genes,
    # which the per-rung CIs above cannot give
    pgpath = SRC.replace(".json", "_pergene.csv")
    if os.path.exists(pgpath):
        shutil.copy(pgpath, f"{OUTD}/ladder_v24ens_full_pergene.csv")
        pg = pd.read_csv(pgpath); pg = pg[pg.surface == SURFACE]
        rows = [(m, *paired_delta(pg, m, direction[m])) for m in PANELS]
        pd.DataFrame([{"metric": m,
                       "direction": "higher_better" if direction[m] else "lower_better",
                       "ens5_minus_seed42_oriented_positive_is_ensemble_better": round(v, 4),
                       "ci95_lo": round(l, 4), "ci95_hi": round(h, 4), "n_genes": n,
                       "significant": bool(not (l < 0 < h))}
                      for m, v, l, h, n in rows]).to_csv(
            f"{OUTD}/ens5_vs_seed42_paired.csv", index=False)
        print("  + ens5_vs_seed42_paired.csv")

        # DeepPBS comparison: only legitimate on the co-crystal surface, where every
        # method was rescored on identical genes.
        pg20 = pd.read_csv(pgpath)
        pg20 = pg20[pg20.surface == "deeppbs20"]
        if "deeppbs" in set(pg20.rung):
            out = []
            for alt in ("v24_seed42", "v24_ens5"):
                for m in PANELS:
                    v, lo, hi, n = paired_delta(pg20, m, direction[m],
                                                ref="deeppbs", alt=alt)
                    out.append({"surface": "deeppbs20", "comparison": f"{alt} - deeppbs",
                                "metric": m,
                                "direction": "higher_better" if direction[m] else "lower_better",
                                "delta_oriented_positive_is_tfscope_better": round(v, 4),
                                "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                                "n_genes": n, "significant": bool(not (lo < 0 < hi))})
            pd.DataFrame(out).to_csv(f"{OUTD}/v24_vs_deeppbs_paired.csv", index=False)
            print("  + v24_vs_deeppbs_paired.csv")
    print(f"saved tables in {OUTD}/")


if __name__ == "__main__":
    main()
