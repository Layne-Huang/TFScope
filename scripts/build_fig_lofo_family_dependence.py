#!/usr/bin/env python
"""Three standalone publication panels for the leave-one-family-out (LOFO) result.

A  seen vs family-held-out performance   (paired dumbbell)
B  effect size dr = seen - unseen        (forest plot with 95% CI)
C  what explains family dependence?      (C1 dr vs seen, C2 dr vs family size)

Each panel is written as its OWN pdf/svg/600-dpi-png so they can be laid out by hand
later; nothing is combined into a composite. Panel C keeps its two sub-plots (C1, C2)
because they are one argument -- "it is learnability, not size" -- and splitting them
would lose the contrast.

Numbers are the frozen LOFO table (12 families, v24 single seed, paired per-gene
bootstrap). Family order is fixed as given: sorted by dr, descending. Not re-sorted.

  python scripts/build_fig_lofo_family_dependence.py [--outdir DIR]
"""
from __future__ import annotations

import argparse
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── data (frozen; do not reorder) ────────────────────────────────────────────
FAMILIES = ["GATA", "T-box", "IRF", "bHLH", "HMG/SOX", "PAX",
            "bZIP", "Homeodomain", "RHD/NFkB", "Nuclear receptor",
            "E2F/DP", "C2H2"]
SEEN = [0.920, 0.857, 0.789, 0.797, 0.789, 0.778, 0.639, 0.852, 0.740, 0.684, 0.624, 0.497]
UNSEEN = [0.374, 0.475, 0.420, 0.482, 0.517, 0.526, 0.400, 0.623, 0.516, 0.466, 0.502, 0.431]
DELTA = [0.546, 0.383, 0.369, 0.315, 0.271, 0.252, 0.239, 0.229, 0.224, 0.218, 0.122, 0.066]
CI_LOW = [0.508, 0.008, 0.305, 0.201, 0.184, -0.016, 0.133, 0.181, 0.097, 0.119, -0.400, -0.012]
CI_HIGH = [0.589, 0.598, 0.447, 0.420, 0.366, 0.468, 0.344, 0.277, 0.362, 0.314, 0.448, 0.150]
PAIRED_N = [3, 4, 3, 22, 7, 3, 12, 40, 3, 12, 3, 40]
FAMILY_SIZE = [8, 15, 9, 88, 29, 9, 52, 196, 10, 51, 9, 610]

RHO_C1 = "Spearman $\\rho$ = 0.853, p = 0.0004"
RHO_C2 = "Spearman $\\rho$ = −0.43, p = 0.16"

BLUE = "#1f5fbf"      # seen
ORANGE = "#e66a1a"    # unseen
GREY_LINE = "#8c8c8c"
GREY_GRID = "#dcdcdc"
INK = "#1a1a1a"
INK2 = "#4d4d4d"


def set_theme():
    """One restrained house style shared by all three panels."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans",
                            "DejaVu Sans", "sans-serif"],
        # keep mathtext (rho, minus signs, the bold panel letter) in the SAME face,
        # otherwise the exported vector file quietly embeds a second typeface
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.cal": "sans",
        "svg.fonttype": "none",     # editable text in SVG
        "pdf.fonttype": 42,         # editable TrueType text in PDF
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": "#9a9a9a",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
    })


def title(ax, letter, text, pad=10, size=11.5):
    """Bold panel letter + regular title, in one text object so spacing stays right."""
    ax.set_title(f"$\\bf{{{letter}}}$   {text}", loc="left", fontsize=size,
                 color=INK, pad=pad)


def save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in ("pdf", "svg"):
        p = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white"); paths.append(p)
    p = os.path.join(outdir, f"{stem}.png")
    fig.savefig(p, dpi=600, bbox_inches="tight", facecolor="white"); paths.append(p)
    plt.close(fig)
    print("  " + "\n  ".join(paths))


# ── Panel A ──────────────────────────────────────────────────────────────────
def panel_a(outdir):
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    fig.subplots_adjust(left=0.235, right=0.975, top=0.925, bottom=0.135)

    y = np.arange(len(FAMILIES))
    for i, (u, s) in enumerate(zip(UNSEEN, SEEN)):
        ax.plot([u, s], [i, i], color=GREY_LINE, lw=1.1, zorder=1,
                solid_capstyle="round")
    ax.scatter(UNSEEN, y, s=58, color=ORANGE, zorder=3, linewidths=0)
    ax.scatter(SEEN, y, s=58, color=BLUE, zorder=3, linewidths=0)

    for i, (u, s) in enumerate(zip(UNSEEN, SEEN)):
        ax.text(u - 0.014, i, f"{u:.3f}", va="center", ha="right",
                fontsize=8.5, color=ORANGE)
        ax.text(s + 0.014, i, f"{s:.3f}", va="center", ha="left",
                fontsize=8.5, color=BLUE)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{i+1}. {f}" for i, f in enumerate(FAMILIES)], color=INK)
    ax.invert_yaxis()                       # rank 1 at the top
    ax.set_ylim(len(FAMILIES) - 0.4, -0.6)
    ax.set_xlim(0.28, 1.02)
    ax.set_xlabel("Pearson $r$")
    ax.xaxis.grid(True, color=GREY_GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    title(ax, "A", "Seen vs. family-held-out performance")

    # the two rows that carry the message. Both callouts sit in the empty band
    # BETWEEN rows -- an earlier placement inside the plotted range covered the IRF
    # and E2F/DP dumbbells.
    ax.annotate("strong family dependence", xy=(0.66, 0.16), xytext=(0.73, 1.52),
                fontsize=9, color=ORANGE, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0,
                                shrinkA=2, shrinkB=2,
                                connectionstyle="arc3,rad=0.28"))
    ax.annotate("little measurable\nfamily-removal penalty",
                xy=(0.472, 10.84), xytext=(0.79, 10.35),
                fontsize=9, color=ORANGE, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0,
                                shrinkA=2, shrinkB=2,
                                connectionstyle="arc3,rad=-0.25"))

    h = [plt.Line2D([], [], marker="o", ls="", ms=7.5, color=BLUE,
                    label="Seen (family in training)"),
         plt.Line2D([], [], marker="o", ls="", ms=7.5, color=ORANGE,
                    label="Unseen (family held out)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.085),
              ncol=2, handletextpad=0.4, columnspacing=2.2)
    save(fig, outdir, "panel_A_seen_vs_heldout")


# ── Panel B ──────────────────────────────────────────────────────────────────
def panel_b(outdir):
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    fig.subplots_adjust(left=0.215, right=0.775, top=0.915, bottom=0.145)

    y = np.arange(len(FAMILIES))
    ax.axvline(0, color=INK2, ls="--", lw=0.9, zorder=1)
    for i, (lo, hi, d) in enumerate(zip(CI_LOW, CI_HIGH, DELTA)):
        ax.plot([lo, hi], [i, i], color=GREY_LINE, lw=1.3, zorder=2,
                solid_capstyle="butt")
        crosses_zero = lo < 0 < hi
        # filled = CI excludes zero; open = CI crosses zero
        ax.plot(d, i, marker="o", ms=7.0, zorder=3,
                mfc="white" if crosses_zero else INK,
                mec=INK, mew=1.3, ls="")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{i+1}. {f}" for i, f in enumerate(FAMILIES)], color=INK)
    ax.invert_yaxis()
    ax.set_ylim(len(FAMILIES) - 0.4, -0.6)
    ax.set_xlim(-0.6, 0.8)
    ax.set_xlabel("$\\Delta r$ (seen − unseen)")
    ax.xaxis.grid(True, color=GREY_GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    title(ax, "B", "Effect size ($\\Delta r$ = seen − unseen) with 95% CI")

    # right-hand numeric columns, in axes-fraction x / data y so they stay aligned
    tr = ax.get_yaxis_transform()
    for i, (d, n) in enumerate(zip(DELTA, PAIRED_N)):
        ax.text(1.045, i, f"{d:+.3f}", transform=tr, va="center", ha="right",
                fontsize=9, color=INK, clip_on=False)
        ax.text(1.175, i, f"n = {n}", transform=tr, va="center", ha="right",
                fontsize=9, color=INK2, clip_on=False)
    ax.text(1.045, -0.95, "$\\Delta r$", transform=tr, va="center", ha="right",
            fontsize=9.5, color=INK, clip_on=False)
    ax.text(1.175, -0.95, "Paired n", transform=tr, va="center", ha="right",
            fontsize=9.5, color=INK, clip_on=False)

    h = [plt.Line2D([], [], marker="o", ls="", ms=7, mfc=INK, mec=INK,
                    label="95% CI excludes 0"),
         plt.Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec=INK,
                    label="95% CI crosses 0")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.082),
              ncol=2, handletextpad=0.4, columnspacing=2.2)
    fig.text(0.215, 0.020,
             "Three non-significant cases differ: low power for PAX and E2F/DP; "
             "near-zero effect for C2H2.",
             fontsize=8.8, color=INK2, ha="left")
    save(fig, outdir, "panel_B_effect_size_forest")


# ── Panel C ──────────────────────────────────────────────────────────────────
def panel_c(outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.815, bottom=0.145, wspace=0.24)
    fig.text(0.008, 0.955, "$\\bf{C}$   What explains family dependence?",
             fontsize=12.5, color=INK, ha="left", va="top")

    lab = dict(fontsize=9, color=INK)

    # C1 -- dr vs seen
    ax1.scatter(SEEN, DELTA, s=58, color=BLUE, zorder=3, linewidths=0)
    # labels sit at the SAME y as their point and are offset only horizontally: a
    # diagonal offset put "bHLH" midway between its own dot and IRF's, so it was
    # ambiguous which point it named
    off1 = {"GATA": (-0.022, 0.0, "right"), "bHLH": (0.020, 0.0, "left"),
            "Homeodomain": (0.020, 0.0, "left"), "C2H2": (0.020, 0.0, "left")}
    for f, (dx, dy, ha) in off1.items():
        i = FAMILIES.index(f)
        ax1.annotate(f, (SEEN[i], DELTA[i]), xytext=(SEEN[i] + dx, DELTA[i] + dy),
                     ha=ha, va="center", **lab)
    ax1.set_xlim(0.40, 1.00); ax1.set_ylim(0.0, 0.70)
    ax1.set_xlabel("Seen performance ($r$)")
    ax1.set_ylabel("$\\Delta r$")
    ax1.grid(True, color=GREY_GRID, lw=0.7)
    ax1.set_axisbelow(True)
    ax1.set_title("$\\bf{C1}$   $\\Delta r$ tracks how well a family is learned when seen",
                  loc="left", fontsize=10.5, color=INK, pad=8)
    ax1.text(0.975, 0.045, RHO_C1, transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=9, color=INK2)

    # C2 -- dr vs family size
    ax2.scatter(FAMILY_SIZE, DELTA, s=58, color=BLUE, zorder=3, linewidths=0)
    # C2H2 goes ABOVE its point, not left of it: at the right edge a same-y label
    # collides with the Spearman annotation
    off2 = {"GATA": (0.16, 0.0, "left"), "C2H2": (0.0, 0.050, "center")}
    for f, (dxlog, dy, ha) in off2.items():
        i = FAMILIES.index(f)
        ax2.annotate(f, (FAMILY_SIZE[i], DELTA[i]),
                     xytext=(FAMILY_SIZE[i] * 10 ** dxlog, DELTA[i] + dy),
                     ha=ha, va="center", **lab)
    ax2.set_xscale("log")
    # data spans 8..610; the 1-10 decade would be empty, so the axis starts at 5 and
    # the ticks are the decades that actually carry points
    ax2.set_xlim(5, 1200); ax2.set_ylim(0.0, 0.70)
    ax2.set_xticks([10, 100, 1000])
    ax2.set_xticklabels(["10", "100", "1,000"])
    ax2.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax2.set_xlabel("Family size (test genes, log scale)")
    ax2.set_ylabel("$\\Delta r$")
    ax2.grid(True, which="major", color=GREY_GRID, lw=0.7)
    ax2.grid(True, which="minor", axis="x", color="#eeeeee", lw=0.5)
    ax2.set_axisbelow(True)
    ax2.set_title("$\\bf{C2}$   $\\Delta r$ does not significantly track family size",
                  loc="left", fontsize=10.5, color=INK, pad=8)
    ax2.text(0.975, 0.045, RHO_C2, transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=9, color=INK2)

    save(fig, outdir, "panel_C_explaining_family_dependence")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures_v24/figure_lofo_family_dependence")
    a = ap.parse_args()
    set_theme()

    print("Panel A:");  panel_a(a.outdir)
    print("Panel B:");  panel_b(a.outdir)
    print("Panel C:");  panel_c(a.outdir)

    src = pd.DataFrame({"rank": np.arange(1, 13), "family": FAMILIES,
                        "seen": SEEN, "unseen": UNSEEN, "delta_r": DELTA,
                        "ci95_low": CI_LOW, "ci95_high": CI_HIGH,
                        "paired_n": PAIRED_N, "family_size_test_genes": FAMILY_SIZE})
    src["ci_excludes_zero"] = ~((src.ci95_low < 0) & (src.ci95_high > 0))
    p = os.path.join(a.outdir, "source_data_lofo_family_dependence.csv")
    src.to_csv(p, index=False)
    print(f"Source data:\n  {p}")


if __name__ == "__main__":
    main()
