#!/usr/bin/env python
"""Six independent subfigures (a-f) for the MoE biochemical-specialisation figure,
plus an assembled version.

  a  expert chemical signatures (amino-acid enrichment, family-controlled)
  b  expert structural-role signatures
  c  expert co-routing structure (top-2 routing)
  d  case study: C2H2 zinc finger
  e  case study: bZIP / bHLH
  f  cross-family reuse of biochemical programs

Every panel is written on its own as panel_<x>.pdf / .svg / .png, and the same drawing
functions are reused for figure_moe_biochemical_programs_full.*.

Usage:
  python scripts/moe_interp/07_subfigures.py [--indir results/moe_biochem]
                                             [--outdir figures_v24/figure_moe_biochemical_programs]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

DIV = LinearSegmentedColormap.from_list("div", [
    "#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"])
SEQP = LinearSegmentedColormap.from_list("seqp", [
    "#FFFFFF", "#EFEDF5", "#DADAEB", "#BCBDDC", "#9E9AC8", "#807DBA", "#6A51A3"])
GREY = "#4D4D4D"
BLUE, ORANGE = "#3C6E9E", "#E07B1A"

ORDER = [5, 3, 6, 0, 7, 2, 1, 4]          # DNA-contact axis, shared by a, b, c
EXPERT_COLOR = {0: "#E9A13B", 1: "#C0392B", 2: "#2E8B8B", 3: "#3C6E9E",
                4: "#8E6BAF", 5: "#8A9A5B", 6: "#E07B1A", 7: "#CC6699"}
# names are the roles the structural regressions support (see README)
EXPERT_NAME = {0: "major-groove readout", 1: "backbone interface",
               2: "base readout (β-element)", 3: "helical / buried scaffold",
               4: "His–Zn scaffold", 5: "buried / loop scaffold",
               6: "Cys–Zn scaffold", 7: "minor-groove / β-element"}

AA_CLASSES = [("Sulfur\n/metal", ["C", "M"], "#E07B1A"),
              ("Basic", ["H", "K", "R"], "#3C6E9E"),
              ("Acidic", ["D", "E"], "#C0392B"),
              ("Polar\nuncharged", ["Q", "N", "S", "T"], "#2E8B8B"),
              ("Hydrophobic", ["A", "V", "L", "I"], "#7F7F7F"),
              ("Aromatic", ["F", "Y", "W"], "#8E6BAF"),
              ("G/P", ["G", "P"], "#8C6D31")]
AA_ORDER = [c for _n, cs, _col in AA_CLASSES for c in cs]

ROLES_B = ["zn_coord", "base_contact", "backbone_contact", "buried_core",
           "helix", "loop", "exposed_nondna"]
ROLE_LABEL_B = {"zn_coord": "Zn coord.", "base_contact": "Base contact",
                "backbone_contact": "Backbone", "buried_core": "Buried core",
                "helix": "Helix", "loop": "Loop/linker",
                "exposed_nondna": "Surface"}
GROUPS_B = [("Structural\nscaffolds", [5, 3, 6], BLUE),
            ("Mixed", [0, 7], "#8A8A8A"),
            ("DNA\ninterface", [2, 1, 4], ORANGE)]

D_EXPERTS = [6, 4, 2, 1]                   # tracks shown in panel d
E_EXPERTS = [0, 3]                         # tracks shown in panel e
F_EXPERTS = [1, 3, 6, 7]                   # mini-panels in panel f
F_FAMS = ["C2H2", "bHLH", "bZIP", "HD", "ETS"]


def bare(ax, keep=()):
    for s in ("top", "right", "bottom", "left"):
        if s not in keep:
            ax.spines[s].set_visible(False)


def head(ax, letter, title, dy=10, dx=-30):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=9, fontweight="bold",
                ha="left", va="bottom", annotation_clip=False)
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(dx + 11, dy),
                textcoords="offset points", fontsize=8, fontweight="bold",
                ha="left", va="bottom", annotation_clip=False)


# ═══ panel a ════════════════════════════════════════════════════════════════════
def draw_a(fig, cell, D):
    gs = GridSpecFromSubplotSpec(2, 3, subplot_spec=cell,
                                 width_ratios=[0.46, 1.0, 0.26],
                                 height_ratios=[0.17, 1.0],
                                 wspace=0.06, hspace=0.09)
    ax_ar = fig.add_subplot(gs[1, 0])
    ax_cl = fig.add_subplot(gs[0, 1])
    ax = fig.add_subplot(gs[1, 1])
    ax_ct = fig.add_subplot(gs[1, 2])

    mat, usage = D["mat"], D["usage"]
    M = mat[np.ix_(ORDER, [D["aa_index"][c] for c in AA_ORDER])]
    vmax = float(np.percentile(np.abs(mat), 98))
    im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(len(AA_ORDER)))
    ax.set_xticklabels(AA_ORDER, fontsize=6)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([f"e{e} ({usage[e] * 100:.0f}%)" for e in ORDER], fontsize=6.6)
    ax.tick_params(length=1.5, pad=1.5)
    bare(ax)
    for x in np.cumsum([len(c) for _n, c, _k in AA_CLASSES])[:-1]:
        ax.axvline(x - .5, color="white", lw=0.9)

    # residue-class header
    x = 0
    for name, cs, col in AA_CLASSES:
        ax_cl.plot([x + .12, x + len(cs) - .12], [0.06, 0.06], color=col, lw=1.3,
                   solid_capstyle="butt", clip_on=False)
        ax_cl.text(x + len(cs) / 2, 0.22, name, ha="center", va="bottom",
                   fontsize=5.0, color=col, fontweight="bold", linespacing=0.95)
        x += len(cs)
    ax_cl.set_xlim(0, len(AA_ORDER)); ax_cl.set_ylim(0, 1)
    ax_cl.set_xticks([]); ax_cl.set_yticks([]); bare(ax_cl)

    # left scaffold -> interface arrow
    ax_ar.annotate("", xy=(0.30, 0.02), xytext=(0.30, 0.98), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#9A9A9A",
                                   shrinkA=0, shrinkB=0))
    ax_ar.text(0.24, 0.97, "Structural\nscaffold", fontsize=6, color=BLUE,
               fontweight="bold", ha="right", va="top", linespacing=1.1)
    ax_ar.text(0.24, 0.52, "Mixed", fontsize=6, color="#8A8A8A", fontweight="bold",
               ha="right", va="center")
    ax_ar.text(0.24, 0.03, "DNA-\ninterface", fontsize=6, color=ORANGE,
               fontweight="bold", ha="right", va="bottom", linespacing=1.1)
    ax_ar.set_xlim(0, 1); ax_ar.set_ylim(0, 1)
    ax_ar.set_xticks([]); ax_ar.set_yticks([]); bare(ax_ar)

    # right DNA-contact enrichment bars
    y = np.arange(len(ORDER))
    vals = [D["contact"][e] for e in ORDER]
    ax_ct.barh(y, vals, height=0.55, color=[BLUE if v < 0 else ORANGE for v in vals],
               ec="none")
    ax_ct.axvline(0, color="#BBBBBB", lw=0.5)
    lim = max(abs(min(vals)), abs(max(vals))) * 1.05
    ax_ct.set_xlim(-lim, lim)
    ax_ct.set_ylim(len(ORDER) - .5, -.5)
    for yy, v in zip(y, vals):
        ax_ct.text(lim * 1.12, yy, f"{v:+.2f}", fontsize=5.8, va="center", ha="left",
                   color="#333333", clip_on=False)
    ax_ct.set_xticks([]); ax_ct.set_yticks([]); bare(ax_ct)
    ax_ct.annotate("DNA\ncontact\nenrichment", xy=(0.5, 1), xycoords="axes fraction",
                   xytext=(0, 3), textcoords="offset points", fontsize=5.8,
                   ha="center", va="bottom", color="#333333", linespacing=1.15,
                   annotation_clip=False)

    ax_cb = ax.inset_axes([0.03, -0.30, 0.60, 0.052])
    cb = fig.colorbar(im, cax=ax_cb, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.2, length=1.4, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Enrichment (log$_2$, family-controlled)", fontsize=5.6, labelpad=1.5)
    ax_cb.annotate("Depleted", xy=(0, 0), xycoords="axes fraction", xytext=(-2, -1),
                   textcoords="offset points", fontsize=5.2, color=GREY,
                   ha="right", va="top", annotation_clip=False)
    ax_cb.annotate("Enriched", xy=(1, 0), xycoords="axes fraction", xytext=(2, -1),
                   textcoords="offset points", fontsize=5.2, color=GREY,
                   ha="left", va="top", annotation_clip=False)
    head(ax_cl, "a", "Expert chemical signatures (amino-acid enrichment)", dy=16, dx=-58)
    return ax


# ═══ panel b ════════════════════════════════════════════════════════════════════
def draw_b(fig, cell, D):
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=cell,
                                 width_ratios=[1.0, 0.30, 0.30], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_gr = fig.add_subplot(gs[0, 1])
    ax_lg = fig.add_subplot(gs[0, 2])

    b = D["b"]
    ok = b[~b.unstable.astype(bool)]
    piv = ok.pivot_table(index="expert", columns="feature", values="effect_size")
    fdr = ok.pivot_table(index="expert", columns="feature", values="fdr")
    v = np.abs(piv[[c for c in ROLES_B if c in piv.columns]].values)
    vmax = float(np.nanpercentile(v, 80))
    smax = float(np.nanpercentile(v, 85))
    norm = TwoSlopeNorm(0, -vmax, vmax)

    for r, e in enumerate(ORDER):
        for c, role in enumerate(ROLES_B):
            if e not in piv.index or role not in piv.columns:
                continue
            val = piv.loc[e, role]
            if not np.isfinite(val):
                ax.plot(c, r, marker="x", ms=2.2, mew=0.6, color="#CCCCCC")
                continue
            s = 3 + 165 * min(1.0, abs(val) / max(smax, 1e-9))
            sig = np.isfinite(fdr.loc[e, role]) and fdr.loc[e, role] < 0.05
            ax.scatter(c, r, s=s, color=DIV(norm(val)), zorder=3,
                       edgecolors="#33333380" if sig else "none",
                       linewidths=0.5 if sig else 0)
    for r in range(len(ORDER)):
        ax.plot([-0.45, len(ROLES_B) - 0.55], [r, r], color="#F0F0F0", lw=0.4, zorder=0)
    for gname, mem, col in GROUPS_B[:-1]:
        last = max(ORDER.index(e) for e in mem)
        ax.plot([-0.5, len(ROLES_B) - 0.5], [last + .5] * 2, color="#DDDDDD", lw=0.5)

    ax.set_xlim(-0.6, len(ROLES_B) - 0.4)
    ax.set_ylim(len(ORDER) - 0.5, -0.5)
    ax.set_xticks(range(len(ROLES_B)))
    ax.set_xticklabels([ROLE_LABEL_B[r] for r in ROLES_B], fontsize=5.8,
                       rotation=34, ha="left", rotation_mode="anchor")
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([f"e{e}" for e in ORDER], fontsize=6.6)
    ax.tick_params(length=1.5, pad=1.5)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(s in ("left", "top"))
        ax.spines[s].set_color("#DDDDDD")

    # group brackets
    for gname, mem, col in GROUPS_B:
        rows = sorted(ORDER.index(e) for e in mem)
        ax_gr.plot([0.10, 0.10], [rows[0] - .35, rows[-1] + .35], color=col, lw=1.4,
                   solid_capstyle="round")
        ax_gr.text(0.22, np.mean(rows), gname, fontsize=5.8, color=col,
                   fontweight="bold", va="center", ha="left", linespacing=1.1)
    ax_gr.set_ylim(len(ORDER) - 0.5, -0.5); ax_gr.set_xlim(0, 1)
    ax_gr.set_xticks([]); ax_gr.set_yticks([]); bare(ax_gr)

    # colour + size legend
    ax_lg.set_xlim(0, 1); ax_lg.set_ylim(0, 1)
    ax_lg.set_xticks([]); ax_lg.set_yticks([]); bare(ax_lg)
    cax = ax_lg.inset_axes([0.02, 0.52, 0.20, 0.44])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=DIV, norm=norm), cax=cax)
    cb.ax.tick_params(labelsize=5.2, length=1.4, pad=1)
    cb.outline.set_linewidth(0.4)
    ax_lg.text(0.02, 1.0, "Enrichment\n(log$_2$ odds)", fontsize=5.6, va="bottom",
               ha="left", linespacing=1.15)
    ax_lg.text(0.02, 0.40, "|effect|", fontsize=5.6, va="center", ha="left")
    for k, frac in enumerate((0.25, 0.5, 1.0)):
        ax_lg.scatter(0.13, 0.30 - 0.115 * k, s=3 + 165 * frac, facecolor="none",
                      edgecolor="#555555", lw=0.5)
        ax_lg.text(0.32, 0.30 - 0.115 * k, f"{frac * smax:.1f}", fontsize=5.2,
                   va="center", ha="left", color=GREY)
    head(ax, "b", "Expert structural-role signatures", dy=38, dx=-30)
    return ax


# ═══ panel c ════════════════════════════════════════════════════════════════════
def draw_c(fig, cell, D):
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=cell, width_ratios=[1.0, 0.11],
                                 height_ratios=[1.0, 0.28], wspace=0.06, hspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    ax_tp = fig.add_subplot(gs[1, :])
    c, E = D["c"], 8

    top = c[(c.fdr < 0.05) & (c.log2_observed_expected > 0)]
    top = top.reindex(top.log2_observed_expected.sort_values(ascending=False).index)
    marked = set(map(tuple, top.head(4)[["expert_i", "expert_j"]].values))

    F = np.full((E, E), np.nan)
    SIG = np.zeros((E, E), bool)
    for r in c.itertuples():
        F[r.expert_i, r.expert_j] = F[r.expert_j, r.expert_i] = r.observed_frequency
        s = (int(r.expert_i), int(r.expert_j)) in marked
        SIG[r.expert_i, r.expert_j] = SIG[r.expert_j, r.expert_i] = s
    Fo = F[np.ix_(ORDER, ORDER)] * 100
    So = SIG[np.ix_(ORDER, ORDER)]

    im = ax.imshow(Fo, cmap=SEQP, norm=Normalize(0, np.nanmax(Fo)),
                   interpolation="nearest")
    for i in range(E):
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fc="#D9D9D9", ec="white",
                               lw=0.6, zorder=3))
    for i in range(E):
        for j in range(E):
            if i != j and So[i, j]:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       ec="#B2182B", lw=0.7, zorder=4))
    ax.set_xticks(range(E)); ax.set_yticks(range(E))
    ax.set_xticklabels([f"e{e}" for e in ORDER], fontsize=6.2)
    ax.set_yticklabels([f"e{e}" for e in ORDER], fontsize=6.2)
    ax.tick_params(length=1.5, pad=1.5)
    bare(ax)

    cb = fig.colorbar(im, cax=ax_cb)
    cb.ax.tick_params(labelsize=5.2, length=1.4, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Co-routing freq. (%)", fontsize=5.4, labelpad=2)

    # typical pairs box
    ax_tp.set_xlim(0, 1); ax_tp.set_ylim(0, 1)
    ax_tp.set_xticks([]); ax_tp.set_yticks([]); bare(ax_tp)
    ax_tp.add_patch(FancyBboxPatch((0.01, 0.03), 0.98, 0.94, boxstyle="round,pad=0.008",
                                   fc="white", ec="#CCCCCC", lw=0.5,
                                   transform=ax_tp.transAxes, clip_on=False))
    ax_tp.text(0.04, 0.90, "Enriched pairs (red outline)   log$_2$ obs/exp",
               fontsize=5.4, fontweight="bold", va="top", ha="left")
    SHORT = {0: "major-groove readout", 1: "backbone interface", 2: "base readout",
             3: "helical scaffold", 4: "His–Zn scaffold", 5: "buried/loop scaffold",
             6: "Cys–Zn scaffold", 7: "minor-groove reader"}
    for k, r in enumerate(top.head(4).itertuples()):
        i, j = int(r.expert_i), int(r.expert_j)
        yy = 0.62 - 0.185 * k
        ax_tp.text(0.05, yy, f"e{i} – e{j}", fontsize=5.4, va="center", ha="left",
                   fontweight="bold", color="#333333")
        ax_tp.text(0.20, yy, f"{SHORT[i]} + {SHORT[j]}", fontsize=5.0,
                   va="center", ha="left", color=GREY)
        ax_tp.text(0.975, yy, f"{r.log2_observed_expected:+.2f}", fontsize=5.0,
                   va="center", ha="right", color="#B2182B")
    head(ax, "c", "Expert co-routing structure (top-2 routing)", dy=10, dx=-26)
    return ax


# ═══ shared case-study helpers ══════════════════════════════════════════════════
def schematic(ax, blocks, brackets, span):
    """Domain cartoon: a grey rod with coloured functional blocks."""
    lo, hi = span
    ax.add_patch(FancyBboxPatch((lo, 0.38), hi - lo, 0.24,
                                boxstyle="round,pad=0.0,rounding_size=0.10",
                                fc="#E8E8E8", ec="#C8C8C8", lw=0.5))
    for x0, x1, col, lab in blocks:
        ax.add_patch(Rectangle((x0, 0.34), x1 - x0, 0.32, fc=col, ec="none"))
        if lab:
            ax.text((x0 + x1) / 2, 0.74, lab, fontsize=5.6, ha="center", va="bottom",
                    color="#333333")
    for x0, x1, col, lab in brackets:
        ax.plot([x0, x1], [0.22, 0.22], color=col, lw=1.2, solid_capstyle="butt")
        ax.text((x0 + x1) / 2, 0.10, lab, fontsize=5.6, ha="center", va="top",
                color=col, fontweight="bold")
    ax.set_xlim(lo - (hi - lo) * 0.03, hi + (hi - lo) * 0.03)
    ax.set_ylim(-0.10, 1.05)
    ax.set_xticks([]); ax.set_yticks([]); bare(ax)


def legend_box(ax, experts, title=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([]); bare(ax)
    n = len(experts)
    ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0.01",
                                fc="white", ec="#CCCCCC", lw=0.5,
                                transform=ax.transAxes, clip_on=False))
    y0, dy = 0.86, 0.86 / max(n, 1)
    for k, e in enumerate(experts):
        y = y0 - dy * (k + 0.35)
        ax.add_patch(FancyBboxPatch((0.07, y - 0.055), 0.11, 0.11,
                                    boxstyle="round,pad=0.004,rounding_size=0.03",
                                    fc=EXPERT_COLOR[e], ec="none",
                                    transform=ax.transAxes, clip_on=False))
        ax.text(0.125, y, f"e{e}", fontsize=5.4, color="white", fontweight="bold",
                ha="center", va="center")
        ax.text(0.22, y, EXPERT_NAME[e], fontsize=5.4, ha="left", va="center",
                color="#333333")


def preference_tracks(ax, piv, experts, positions, xlabel, xticks, hi_pos=(), hi_col=None,
                      ticklabels=None):
    """One row per expert; bar height = fraction of modules routed to it."""
    n = len(experts)
    for k, e in enumerate(experts):
        base = n - 1 - k
        ax.add_patch(Rectangle((positions[0] - .5, base + .08), len(positions),
                               0.84, fc="#FAFAFA", ec="#DDDDDD", lw=0.4))
        for p in positions:
            if p in piv.index and e in piv.columns:
                f = float(piv.loc[p, e])
                if f > 0.01:
                    ax.add_patch(Rectangle((p - .38, base + .10), 0.76, 0.80 * f,
                                           fc=EXPERT_COLOR[e], ec="none"))
        ax.text(positions[0] - 0.9, base + 0.5, f"e{e}", fontsize=5.6, ha="right",
                va="center", color=EXPERT_COLOR[e], fontweight="bold")
    for p in hi_pos:
        ax.plot([p, p], [-0.10, n], color=hi_col or "#BBBBBB", lw=0.4, ls=(0, (1, 1)),
                zorder=0)
    ax.set_xlim(positions[0] - 1.0, positions[-1] + 0.6)
    ax.set_ylim(-0.12, n + 0.02)
    ax.set_yticks([])
    ax.set_xticks(xticks)
    ax.set_xticklabels(ticklabels if ticklabels else
                       [f"{t:+d}" if isinstance(t, int) and t < 0 else str(t)
                        for t in xticks], fontsize=5.4)
    ax.tick_params(length=1.4, pad=1)
    ax.set_xlabel(xlabel, fontsize=5.8, labelpad=1)
    bare(ax, keep=("bottom",))
    ax.spines["bottom"].set_color("#CCCCCC")


def put_image(ax, path):
    if os.path.exists(path):
        ax.imshow(plt.imread(path))
    else:
        ax.text(0.5, 0.5, "render missing", ha="center", va="center", fontsize=5,
                color="#BBBBBB", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([]); bare(ax)


# ═══ panel d ════════════════════════════════════════════════════════════════════
def draw_d(fig, cell, D, outdir):
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=cell, height_ratios=[0.30, 1.0],
                                 width_ratios=[1.0, 0.80], hspace=0.20, wspace=0.10)
    ax_sc = fig.add_subplot(gs[0, :])
    ax_im = fig.add_subplot(gs[1, 0])
    gsr = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1],
                                  height_ratios=[0.62, 1.0], hspace=0.30)
    ax_lg = fig.add_subplot(gsr[0])
    ax_tr = fig.add_subplot(gsr[1])

    reg = D["d"]
    piv = reg.pivot(index="position", columns="expert", values="frac")
    positions = sorted(piv.index)
    schematic(
        ax_sc,
        blocks=[(-9.5, -8.5, EXPERT_COLOR[6], None), (-6.5, -5.5, EXPERT_COLOR[6], None),
                (-1.5, 6.4, EXPERT_COLOR[2], "Recognition helix"),
                (6.6, 7.5, EXPERT_COLOR[4], None), (10.5, 11.5, EXPERT_COLOR[4], None)],
        brackets=[(-9.6, -5.4, EXPERT_COLOR[6], "e6"),
                  (-1.5, 6.4, EXPERT_COLOR[2], "e2 / e1"),
                  (6.5, 11.6, EXPERT_COLOR[4], "e4")],
        span=(positions[0] - 0.5, positions[-1] + 0.5))
    for x, lab in ((-9, "Cys"), (-6, "Cys"), (7, "His"), (11, "His")):
        ax_sc.text(x, 0.74, lab, fontsize=5.4, ha="center", va="bottom",
                   color="#333333")

    put_image(ax_im, os.path.join(outdir, "render_d_zincfinger.png"))
    legend_box(ax_lg, D_EXPERTS)
    # canonical C2H2 helix numbering skips 0 (..., -2, -1, 1, 2, ...), so the
    # continuous register index p maps to canonical c = p for p>=1, else p-1.
    # Recognition positions -1/+2/+3/+6 therefore sit at p = 0/2/3/6.
    preference_tracks(ax_tr, piv, D_EXPERTS, positions,
                      "Recognition-helix position (canonical C2H2 numbering)",
                      [-9, -6, 0, 3, 7, 11],
                      ticklabels=["-10", "-7", "-1", "+3", "+7", "+11"],
                      hi_pos=(-9, -6, 0, 2, 3, 6, 7, 11))
    ax_tr.annotate(f"n = {int(reg.n_fingers.max()):,} fingers", xy=(1, 1),
                   xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                   fontsize=5.2, color=GREY, ha="right", va="bottom",
                   annotation_clip=False)
    head(ax_sc, "d", "Case study: C2H2 zinc finger", dy=6, dx=-24)
    return ax_sc


# ═══ panel e ════════════════════════════════════════════════════════════════════
def draw_e(fig, cell, D, outdir):
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=cell, height_ratios=[0.30, 1.0],
                                 width_ratios=[1.0, 0.80], hspace=0.20, wspace=0.10)
    ax_sc = fig.add_subplot(gs[0, :])
    ax_im = fig.add_subplot(gs[1, 0])
    gsr = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1],
                                  height_ratios=[0.45, 1.0], hspace=0.30)
    ax_lg = fig.add_subplot(gsr[0])
    ax_tr = fig.add_subplot(gsr[1])

    reg = D["e"]
    if "subset" in reg:                      # bHLH only: see 06_case_and_family_stats.py
        reg = reg[reg.subset == "bHLH"]
    reg = reg[reg.n_domains >= 12]
    piv = reg.pivot(index="position", columns="expert", values="frac")
    positions = sorted(piv.index)
    lo, hi = positions[0], positions[-1]
    schematic(
        ax_sc,
        blocks=[(lo, 0, EXPERT_COLOR[0], "Basic DNA-binding region"),
                (0, hi, EXPERT_COLOR[3], "Helix 1 / HLH (dimerisation)")],
        brackets=[(lo, 0, EXPERT_COLOR[0], "e0"), (0, hi, EXPERT_COLOR[3], "e3")],
        span=(lo - 0.5, hi + 0.5))

    put_image(ax_im, os.path.join(outdir, "render_e_bhlh.png"))
    legend_box(ax_lg, E_EXPERTS)
    preference_tracks(ax_tr, piv, E_EXPERTS, positions,
                      "Domain position (0 = end of basic region)",
                      [t for t in (-12, -8, -4, 0, 4, 8, 12, 16, 20, 24)
                       if lo <= t <= hi], hi_pos=(0,), hi_col="#888888")
    ax_tr.annotate(f"n = {int(reg.n_domains.max())} domains", xy=(1, 1),
                   xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                   fontsize=5.2, color=GREY, ha="right", va="bottom",
                   annotation_clip=False)
    head(ax_sc, "e", "Case study: bHLH", dy=6, dx=-24)
    return ax_sc


# ═══ panel f ════════════════════════════════════════════════════════════════════
def draw_f(fig, cell, D):
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=cell, hspace=0.62, wspace=0.34)
    f = D["f"]
    axes = []
    for k, e in enumerate(F_EXPERTS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        axes.append(ax)
        g = f[f.expert == e].set_index("family_label").reindex(F_FAMS)
        x = np.arange(len(F_FAMS))
        vals = g.profile_cosine.values.astype(float)
        lo = g.cos_ci_lo.values.astype(float)
        hi = g.cos_ci_hi.values.astype(float)
        ok = np.isfinite(vals)
        ax.bar(x[ok], vals[ok], width=0.62, color=EXPERT_COLOR[e], ec="none")
        err = np.vstack([np.clip(vals[ok] - lo[ok], 0, None),
                         np.clip(hi[ok] - vals[ok], 0, None)])
        ax.errorbar(x[ok], vals[ok], yerr=err, fmt="none", ecolor="#555555",
                    elinewidth=0.5, capsize=1.2, capthick=0.5)
        for i in np.where(~ok)[0]:
            ax.text(i, 0.04, "n/a", fontsize=4.8, ha="center", va="bottom",
                    color="#AAAAAA", rotation=90)
        ax.axhline(0, color="#BBBBBB", lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(F_FAMS, fontsize=5.2, rotation=35, ha="right",
                           rotation_mode="anchor")
        ax.set_ylim(-0.25, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=5.2, length=1.4, pad=1)
        bare(ax, keep=("left", "bottom"))
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#CCCCCC")
        ax.set_title(f"e{e}: {EXPERT_NAME[e]}", fontsize=5.5, pad=2.5,
                     color=EXPERT_COLOR[e], fontweight="bold")
        if k % 2 == 0:
            ax.set_ylabel("Program\nconsistency", fontsize=5.4, labelpad=1.5,
                          linespacing=1.1)
    head(axes[0], "f", "Cross-family reuse of biochemical programs", dy=16, dx=-26)
    return axes[0]


# ═══════════════════════════════════════════════════════════════════════════════
def load(indir):
    a = pd.read_csv(os.path.join(indir, "expert_residue_stats.csv"))
    summ = pd.read_csv(os.path.join(indir, "expert_summary.csv")).set_index("expert")
    mat = a.pivot_table(index="expert", columns="feature", values="effect_size")
    return dict(
        mat=mat[sorted(mat.columns)].values,
        aa_index={c: i for i, c in enumerate(sorted(mat.columns))},
        usage=summ.expert_usage.reindex(range(8)).values,
        contact=summ.contact_enrichment.reindex(range(8)).values,
        b=pd.read_csv(os.path.join(indir, "expert_role_stats.csv")),
        c=pd.read_csv(os.path.join(indir, "corouting_stats.csv")),
        d=pd.read_csv(os.path.join(indir, "c2h2_register.csv")),
        e=pd.read_csv(os.path.join(indir, "bzip_bhlh_register.csv")),
        f=pd.read_csv(os.path.join(indir, "family_reuse.csv")),
    )


PANELS = {
    "a": dict(fn=lambda f, c, D, o: draw_a(f, c, D), size=(5.3, 2.75),
              m=dict(left=0.075, right=0.905, top=0.80, bottom=0.235)),
    "b": dict(fn=lambda f, c, D, o: draw_b(f, c, D), size=(4.2, 2.85),
              m=dict(left=0.105, right=0.985, top=0.68, bottom=0.045)),
    "c": dict(fn=lambda f, c, D, o: draw_c(f, c, D), size=(3.7, 3.15),
              m=dict(left=0.10, right=0.875, top=0.895, bottom=0.045)),
    "d": dict(fn=draw_d, size=(5.1, 3.35),
              m=dict(left=0.075, right=0.985, top=0.88, bottom=0.075)),
    "e": dict(fn=draw_e, size=(4.3, 3.35),
              m=dict(left=0.090, right=0.985, top=0.88, bottom=0.075)),
    "f": dict(fn=lambda f, c, D, o: draw_f(f, c, D), size=(3.3, 2.75),
              m=dict(left=0.145, right=0.962, top=0.79, bottom=0.155)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/moe_biochem")
    ap.add_argument("--outdir", default="figures_v24/figure_moe_biochemical_programs")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    D = load(a.indir)

    for key, spec in PANELS.items():
        if a.only and key not in a.only:
            continue
        fig = plt.figure(figsize=spec["size"])
        gs = fig.add_gridspec(1, 1, **spec["m"])
        spec["fn"](fig, gs[0, 0], D, a.outdir)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(os.path.join(a.outdir, f"panel_{key}.{ext}"),
                        dpi=600 if ext == "png" else None)
        plt.close(fig)
        print(f"wrote panel_{key}.[pdf|svg|png]  {spec['size'][0]}x{spec['size'][1]} in")

    if a.only:
        return
    fig = plt.figure(figsize=(13.2, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.30, 1.02, 0.90],
                          height_ratios=[1.0, 1.16],
                          left=0.048, right=0.968, top=0.925, bottom=0.055,
                          wspace=0.28, hspace=0.42)
    draw_a(fig, gs[0, 0], D)
    draw_b(fig, gs[0, 1], D)
    draw_c(fig, gs[0, 2], D)
    draw_d(fig, gs[1, 0], D, a.outdir)
    draw_e(fig, gs[1, 1], D, a.outdir)
    draw_f(fig, gs[1, 2], D)
    base = os.path.join(a.outdir, "figure_moe_biochemical_programs_full")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(base + "." + ext, dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote {base}.[pdf|svg|png]")


if __name__ == "__main__":
    main()
