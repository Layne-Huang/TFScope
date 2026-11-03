#!/usr/bin/env python
"""Main figure: the v24 MoE decomposes TF specificity into biochemical programs.

  A  expert x amino-acid chemistry signature, TF-family composition controlled,
     flanked by route usage and DNA-contact enrichment
  B  expert x structural-role enrichment, controlled for BOTH family and amino acid
  C  top-2 co-routing against an independence null
  D  canonical case studies: a C2H2 zinc finger and a bHLH domain

Experts are ordered along the observed DNA-contact axis in every panel, so the rows
read scaffold-enriched -> mixed -> DNA-interface-enriched throughout.

Usage:
  python scripts/moe_interp/04_figure.py [--indir results/moe_biochem] [--outdir figures_v24/figure_moe_biochemical_programs]
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ── house style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

DIV = LinearSegmentedColormap.from_list(
    "muted_div", ["#2C5985", "#7BA3C4", "#DCE6ED", "#F7F7F5",
                  "#F0DCD3", "#D08D77", "#AC4A31"])
GREY = "#4D4D4D"
PANEL_KW = dict(fontsize=9, fontweight="bold", va="top", ha="left")

# amino acids grouped by biochemical class
AA_CLASSES = [
    ("S", ["C", "M"], "#8C8C8C"),
    ("basic", ["H", "K", "R"], "#7A93AE"),
    ("acid", ["D", "E"], "#A8724F"),
    ("polar", ["Q", "N", "S", "T"], "#8FA98C"),
    ("aliphatic", ["A", "V", "L", "I"], "#B0A88F"),
    ("arom", ["F", "Y", "W"], "#9A7FA8"),
    ("G/P", ["G", "P"], "#B58C8C"),
]
AA_ORDER = [c for _n, cs, _col in AA_CLASSES for c in cs]

ROLE_ORDER = ["base_contact", "major_groove", "minor_groove", "backbone_contact",
              "zn_coord", "buried_core", "helix", "strand", "loop", "ppi",
              "exposed_nondna"]
ROLE_SHORT = {
    "base_contact": "DNA base contact",
    "major_groove": "major groove",
    "minor_groove": "minor groove",
    "backbone_contact": "DNA backbone",
    "zn_coord": "Zn coordination",
    "buried_core": "buried core",
    "helix": "α-helix",
    "strand": "β-strand",
    "loop": "loop / turn",
    "ppi": "protein–protein interface",
    "exposed_nondna": "exposed non-DNA",
}
# short descriptors used only when panel B actually supports them
ROLE_TAG = {
    "base_contact": "base readout",
    "major_groove": "major groove",
    "minor_groove": "minor groove",
    "backbone_contact": "DNA backbone",
    "zn_coord": "Zn scaffold",
    "buried_core": "buried core",
    "helix": "helical",
    "strand": "β-strand",
    "loop": "loop/turn",
    "ppi": "protein interface",
    "exposed_nondna": "exposed surface",
}


def strip_axes(ax, keep=()):
    for s in ("top", "right", "bottom", "left"):
        if s not in keep:
            ax.spines[s].set_visible(False)


def head(ax, title, subtitle, dy_title=26, dy_sub=4, size=7.5, sub_size=5.8):
    """Panel heading placed in offset points, so it survives a change of aspect."""
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, dy_title), textcoords="offset points",
                fontsize=size, fontweight="bold", ha="left", va="bottom",
                annotation_clip=False)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, dy_sub), textcoords="offset points",
                    fontsize=sub_size, color=GREY, style="italic", ha="left",
                    va="bottom", linespacing=1.35, annotation_clip=False)


# ── panel A ────────────────────────────────────────────────────────────────────
def panel_a(fig, cell, order, mat, usage, contact, aa_index, contact_struct=None):
    gs = GridSpecFromSubplotSpec(2, 4, subplot_spec=cell,
                                 width_ratios=[0.19, 1.0, 0.24, 0.05],
                                 height_ratios=[0.055, 1.0],
                                 wspace=0.10, hspace=0.06)
    ax_cls = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 0])
    ax_h = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[1, 2])
    ax_cb = fig.add_subplot(gs[1, 3])

    M = mat[np.ix_(order, [aa_index[c] for c in AA_ORDER])]
    vmax = float(np.percentile(np.abs(mat), 98))
    im = ax_h.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0.0, -vmax, vmax),
                     aspect="auto", interpolation="nearest")
    ax_h.set_xticks(range(len(AA_ORDER)))
    ax_h.set_xticklabels(AA_ORDER, fontsize=6.2)
    ax_h.set_yticks([])
    ax_h.tick_params(length=1.5, pad=1)
    strip_axes(ax_h)
    # mark the two strongest residues per expert
    for r in range(len(order)):
        for c in np.argsort(-M[r])[:2]:
            ax_h.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                     ec="#1A1A1A", lw=0.65))

    # residue-class strip
    x = 0
    for name, cs, col in AA_CLASSES:
        ax_cls.add_patch(Rectangle((x, 0), len(cs), 1, fc=col, ec="white", lw=0.6))
        ax_cls.text(x + len(cs) / 2, 0.5, name, ha="center", va="center",
                    fontsize=5.2, color="white", fontweight="bold")
        x += len(cs)
    ax_cls.set_xlim(0, len(AA_ORDER)); ax_cls.set_ylim(0, 1)
    ax_cls.set_xticks([]); ax_cls.set_yticks([]); strip_axes(ax_cls)

    # left: route usage
    y = np.arange(len(order))
    ax_u.barh(y, [usage[e] * 100 for e in order], height=0.52, color="#C3C9CF",
              ec="#8A9299", lw=0.35)
    ax_u.set_yticks(y)
    ax_u.set_yticklabels([f"e{e}" for e in order], fontsize=6.8)
    ax_u.invert_xaxis()
    ax_u.set_ylim(len(order) - .5, -.5)
    ax_u.set_xlabel("route\nusage (%)", fontsize=6, labelpad=1)
    ax_u.tick_params(length=1.5, pad=1, labelsize=5.8)
    ax_u.set_xticks([0, 10])
    strip_axes(ax_u, keep=("bottom",))

    # right: DNA-contact enrichment
    vals = [contact[e] for e in order]
    ax_c.barh(y, vals, height=0.52,
              color=[DIV(TwoSlopeNorm(0, -vmax, vmax)(v * 3)) for v in vals],
              ec="#8A9299", lw=0.35)
    ax_c.axvline(0, color=GREY, lw=0.6)
    if contact_struct is not None:
        ax_c.plot([contact_struct[e] for e in order], y, ls="", marker="o",
                  ms=1.9, mew=0, color="#1A1A1A", zorder=4, clip_on=False)
    ax_c.set_ylim(len(order) - .5, -.5)
    ax_c.set_yticks([])
    ax_c.set_xlabel("DNA-contact\nenrichment (log$_2$)", fontsize=6, labelpad=1)
    ax_c.set_xticks([-1, 0])
    ax_c.tick_params(length=1.5, pad=1, labelsize=5.8)
    strip_axes(ax_c, keep=("bottom",))

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("family-adjusted\nresidue enrichment (log$_2$)", fontsize=5.8, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)

    sub = "TF-family composition controlled; box = 2 strongest residues"
    if contact_struct is not None:
        sub += "\nbars = v23 supervision set, dots = recomputed co-crystal contacts"
    head(ax_cls, "Experts show distinct residue-chemistry signatures", sub,
         dy_title=22, dy_sub=3)
    return ax_u


# ── panel B ────────────────────────────────────────────────────────────────────
def panel_b(fig, cell, order, bdf, roles, tags=None):
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=cell, width_ratios=[1.0, 0.05],
                                 wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])

    ok = bdf if "unstable" not in bdf else bdf[~bdf.unstable.astype(bool)]
    piv = ok.pivot_table(index="expert", columns="feature", values="effect_size")
    fdr = ok.pivot_table(index="expert", columns="feature", values="fdr")
    piv = piv.reindex(index=range(max(bdf.expert) + 1), columns=list(piv.columns))
    fdr = fdr.reindex(index=piv.index, columns=piv.columns)
    # Zn coordination carries odds ratios an order of magnitude above every other role,
    # so colour and area both saturate well below the maximum; otherwise the whole grid
    # collapses to white with two red dots.
    vmax = float(np.nanpercentile(np.abs(piv.values), 75)) or 1.0
    norm = TwoSlopeNorm(0.0, -vmax, vmax)
    smax = float(np.nanpercentile(np.abs(piv.values), 80))

    for r, e in enumerate(order):
        for c, role in enumerate(roles):
            if e not in piv.index or role not in piv.columns:
                continue
            v = piv.loc[e, role]
            if not np.isfinite(v):
                ax.plot(c, r, marker="x", ms=2.4, mew=0.6, color="#BBBBBB")
                continue
            s = 6 + 150 * min(1.0, abs(v) / max(smax, 1e-9))
            sig = np.isfinite(fdr.loc[e, role]) and fdr.loc[e, role] < 0.05
            ax.scatter(c, r, s=s, color=DIV(norm(v)), zorder=3,
                       edgecolors="#1A1A1A" if sig else "none",
                       linewidths=0.65 if sig else 0)
            if abs(v) > vmax * 1.6:      # flag the saturated cells with their value
                ax.text(c, r + 0.44, f"{v:+.1f}", fontsize=4.4, ha="center",
                        va="center", color="#1A1A1A", zorder=6)

    # reserve in-axes room on the right for the data-supported role descriptors
    longest = max((len(t) for t in tags.values()), default=0) if tags else 0
    pad_right = (0.9 + 0.225 * longest) if tags else 0.4
    ax.set_xlim(-0.6, len(roles) - 1 + pad_right)
    ax.set_ylim(len(order) - 0.5, -1.9)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels([ROLE_SHORT.get(r, r) for r in roles], fontsize=5.5,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"e{e}" for e in order], fontsize=6.8)
    ax.tick_params(length=1.5, pad=1)
    ax.set_axisbelow(True)
    for r in range(len(order)):
        ax.plot([-0.45, len(roles) - 0.55], [r, r], color="#EFEFEF", lw=0.45, zorder=0)
    strip_axes(ax)

    if tags:
        for r, e in enumerate(order):
            ax.text(len(roles) - 0.35, r, tags[e], fontsize=5.0, va="center",
                    ha="left", color=GREY)

    # bubble-size key, tucked above the first row inside the descriptor column
    ax.text(1.0, -1.25, "|log$_2$ OR|", fontsize=4.8, ha="right", va="center",
            color=GREY)
    for k, frac in enumerate((1.0, 0.5)):
        x = 1.75 + 1.05 * k
        ax.scatter(x, -1.25, s=6 + 150 * frac, color="#D9D9D9", ec="#8A9299",
                   lw=0.4, clip_on=False, zorder=5)
        ax.text(x, -1.78, f"{frac * smax:.1f}", fontsize=4.8, ha="center",
                va="center", color=GREY)

    sm = plt.cm.ScalarMappable(cmap=DIV, norm=norm)
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("role log$_2$ odds ratio", fontsize=5.8, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)

    head(ax, "Experts separate structural scaffolding from\nDNA-interface functions",
         "amino-acid identity and TF family both controlled; outline = FDR < 0.05\n"
         f"colour and area saturate at |log$_2$ OR| = {vmax:.1f}; "
         "saturated cells are labelled",
         dy_title=27, dy_sub=3)
    return ax


# ── panel C ────────────────────────────────────────────────────────────────────
def panel_c(fig, cell, order, cdf, E):
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=cell, width_ratios=[1.0, 0.05],
                                 wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])

    M = np.full((E, E), np.nan)
    for r in cdf.itertuples():
        M[r.expert_i, r.expert_j] = r.log2_observed_expected
        M[r.expert_j, r.expert_i] = r.log2_observed_expected
    Mo = M[np.ix_(order, order)]
    vmax = float(np.nanpercentile(np.abs(Mo), 98))
    im = ax.imshow(Mo, cmap=DIV, norm=TwoSlopeNorm(0.0, -vmax, vmax),
                   interpolation="nearest")
    for i in range(E):
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fc="white", ec="none", zorder=3))

    ax.set_xticks(range(E)); ax.set_yticks(range(E))
    ax.set_xticklabels([f"e{e}" for e in order], fontsize=6.5)
    ax.set_yticklabels([f"e{e}" for e in order], fontsize=6.5)
    ax.tick_params(length=1.5, pad=1)
    strip_axes(ax)

    # annotate the strongest enriched pairs
    pos = cdf[(cdf.fdr < 0.05) & (cdf.log2_observed_expected > 0)]
    pos = pos.reindex(pos.log2_observed_expected.sort_values(ascending=False).index).head(4)
    ranks = {e: i for i, e in enumerate(order)}
    for r in pos.itertuples():
        i, j = ranks[r.expert_i], ranks[r.expert_j]
        for (a, b) in ((i, j), (j, i)):
            ax.text(b, a, f"{r.log2_observed_expected:+.2f}", ha="center", va="center",
                    fontsize=5.0, color="#1A1A1A", zorder=4)
            ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fill=False, ec="#1A1A1A",
                                   lw=0.7, zorder=4))

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("co-routing\nlog$_2$(obs/exp)", fontsize=5.8, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)

    head(ax, "Top-2 routing combines experts non-randomly",
         "expectation preserves each expert's marginal usage\n"
         "boxed = strongest enriched pairs", dy_title=19, dy_sub=3)
    return ax


# ── panel D ────────────────────────────────────────────────────────────────────
def draw_track(ax, case, contact, norm, arch, title, subtitle):
    n = len(case["pos"])
    seq = case["seq"]
    vals = np.array([contact[e] for e in case["e1"]])

    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((i, 0), 1, 1, fc=DIV(norm(v)), ec="none"))
    ax.add_patch(Rectangle((0, 0), n, 1, fill=False, ec="#9A9A9A", lw=0.45))

    zn = np.array(case["zn_coord"])
    bc = np.array(case["base_contact"])
    pp = np.array(case["ppi"])
    ms = max(1.2, min(2.6, 130.0 / n))
    for i in np.where(zn)[0]:
        cys = seq[i] == "C"
        ax.plot(i + .5, 1.52, marker="D" if cys else "s", ms=ms,
                color="#3B3B3B" if cys else "#7F8B96", mew=0)
    for i in np.where(bc)[0]:
        ax.plot(i + .5, -0.55, marker="o", ms=ms, color="#AC4A31", mew=0)
    for i in np.where(pp & ~bc)[0]:
        ax.plot(i + .5, -0.55, marker="_", ms=ms * 1.4, mew=0.7, color="#2C5985")

    for i in case.get("_recog", []):
        ax.add_patch(Rectangle((i, 0), 1, 1, fill=False, ec="#333333", lw=0.38))

    for (s, e, lab, col) in arch:
        ax.add_patch(Rectangle((s, -1.95), e - s, 0.52, fc=col, ec="none"))
        if (e - s) / n > 0.20:
            ax.text((s + e) / 2, -1.69, lab, ha="center", va="center", fontsize=4.9,
                    color="white", fontweight="bold")
        elif (s + e) / 2 > 0.7 * n:
            ax.text(n, -2.18, lab, ha="right", va="top", fontsize=4.9, color=col)
        else:
            ax.text((s + e) / 2, -2.18, lab, ha="center", va="top",
                    fontsize=4.9, color=col)

    ax.set_xlim(-0.5, n + 0.5); ax.set_ylim(-2.9, 2.15)
    ax.set_xticks([]); ax.set_yticks([])
    strip_axes(ax)
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 14),
                textcoords="offset points", fontsize=6.6, fontweight="bold",
                ha="left", va="bottom", annotation_clip=False)
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4),
                textcoords="offset points", fontsize=5.6, color=GREY,
                ha="left", va="bottom", annotation_clip=False)


def draw_inset(ax, png):
    if png and os.path.exists(png):
        ax.imshow(plt.imread(png))
    else:
        ax.text(0.5, 0.5, "inset\npending", ha="center", va="center",
                fontsize=5, color="#BBBBBB", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([]); strip_axes(ax)


def panel_d(fig, cell, cases, contact, norm, inset_dir, key_label=None):
    gs = GridSpecFromSubplotSpec(3, 2, subplot_spec=cell, width_ratios=[1.0, 0.40],
                                 height_ratios=[1.0, 1.0, 0.14], hspace=0.62,
                                 wspace=0.05)
    specs = [
        ("c2h2", "C2H2: metal-scaffold and base-readout programs"),
        ("bhlh", "bHLH: helical-scaffold and basic DNA-contact programs"),
    ]
    axes = []
    for k, (key, title) in enumerate(specs):
        ax = fig.add_subplot(gs[k, 0])
        axi = fig.add_subplot(gs[k, 1])
        axes.append(ax)
        if key not in cases:
            ax.text(0.5, 0.5, f"{key}: no mapped structure", ha="center", va="center",
                    fontsize=6, color=GREY, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([]); strip_axes(ax)
            axi.set_xticks([]); axi.set_yticks([]); strip_axes(axi)
            continue
        case = cases[key]
        draw_track(ax, case, contact, norm, case.get("_arch", []), title,
                   f"{case['gene']} · PDB {case['pdb']}")
        draw_inset(axi, os.path.join(inset_dir, f"inset_{key}.png"))

    if key_label:
        axk = fig.add_subplot(gs[2, 0])
        grad = np.linspace(norm.vmin, norm.vmax, 256).reshape(1, -1)
        axk.imshow(grad, cmap=DIV, norm=norm, aspect="auto", extent=[0, 1, 0, 1])
        axk.set_yticks([])
        axk.set_xticks([0, 0.5, 1])
        axk.set_xticklabels([f"{norm.vmin:+.1f}", "0", f"{norm.vmax:+.1f}"], fontsize=4.8)
        axk.tick_params(length=1.2, pad=1)
        for s in ("top", "right", "left"):
            axk.spines[s].set_visible(False)
        axk.spines["bottom"].set_linewidth(0.4)
        axk.annotate(key_label, xy=(0, 1), xycoords="axes fraction", xytext=(0, 2),
                     textcoords="offset points", fontsize=4.8, color=GREY,
                     ha="left", va="bottom", annotation_clip=False)
    return axes


# ─────────────────────────────────────────────────────────────────────────────
def build_arch(case):
    """Segment bars + canonical recognition positions, derived from the annotation."""
    n = len(case["pos"])
    zn = np.array(case["zn_coord"])
    hel = np.array(case["helix"])
    if case["family"].startswith("C2H2") and zn.sum() >= 4:
        seq = case["seq"]
        # group Zn ligands into fingers: a C following an H opens the next CCHH module
        fingers, cur = [], []
        for i in np.where(zn)[0]:
            if cur and seq[i] == "C" and seq[cur[-1]] == "H":
                fingers.append(cur); cur = []
            cur.append(int(i))
        if cur:
            fingers.append(cur)
        arch, recog = [], []
        for f, lig in enumerate(fingers):
            s, e = lig[0], lig[-1]
            arch.append((max(s - 2, 0), min(e + 2, n), f"ZF{f + 1}", "#7F8B96"))
            # recognition helix = first helical stretch after the second Cys ligand
            cys = [i for i in lig if seq[i] == "C"]
            anchor = cys[-1] if cys else s
            after = np.where(hel[anchor:e + 1])[0]
            if len(after):
                h = int(anchor + after[0])
                recog += [p for p in (h - 1, h + 2, h + 3, h + 6) if 0 <= p < n]
        return arch, sorted(set(recog))
    bc = np.array(case["base_contact"])
    if bc.any():
        last = int(np.where(bc)[0].max())
        return [(0, last + 1, "basic / DNA-binding", "#AC4A31"),
                (last + 1, n, "helix 1 / dimer.", "#2C5985")], []
    return [(0, n, "domain", "#7F8B96")], []


def expert_labels(order, bdf, contact, n_tags=1):
    """Only name an expert after a role that panel B actually supports (FDR < 0.05)."""
    out = {}
    for e in order:
        g = bdf[(bdf.expert == e) & (bdf.fdr < 0.05) & (bdf.effect_size > 0)]
        if "unstable" in g:
            g = g[~g.unstable.astype(bool)]
        g = g.reindex(g.effect_size.sort_values(ascending=False).index)
        tags = [ROLE_TAG.get(r.feature, r.feature) for r in g.head(n_tags).itertuples()]
        out[e] = " + ".join(tags) if tags else "n.s."
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/moe_biochem")
    ap.add_argument("--outdir", default="figures_v24/figure_moe_biochemical_programs")
    ap.add_argument("--order", default="5,3,6,0,7,2,1,4")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    adf = pd.read_csv(os.path.join(a.indir, "expert_residue_stats.csv"))
    bdf = pd.read_csv(os.path.join(a.indir, "expert_role_stats.csv"))
    cdf = pd.read_csv(os.path.join(a.indir, "corouting_stats.csv"))
    cases = json.load(open(os.path.join(a.indir, "case_studies.json")))
    order = [int(x) for x in a.order.split(",")]
    E = int(adf.expert.max()) + 1

    summ = pd.read_csv(os.path.join(a.indir, "expert_summary.csv")).set_index("expert")
    aa_index = {c: i for i, c in enumerate(sorted(adf.feature.unique()))}
    mat = adf.pivot_table(index="expert", columns="feature", values="effect_size")
    mat = mat[sorted(mat.columns)].values
    usage = summ.expert_usage.reindex(range(E)).values
    contact = summ.contact_enrichment.reindex(range(E)).values
    contact_struct = (summ.contact_enrichment_structural_any.reindex(range(E)).values
                      if "contact_enrichment_structural_any" in summ else None)
    roles = [r for r in ROLE_ORDER if r in set(bdf.feature)]
    assert np.isfinite(contact).all() and np.isfinite(usage).all(), \
        "expert_summary.csv is missing experts"

    for k in cases:
        cases[k]["_arch"], cases[k]["_recog"] = build_arch(cases[k])

    labs = expert_labels(order, bdf, contact, n_tags=2)

    fig = plt.figure(figsize=(7.20, 8.30))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[1.0, 0.92],
                          left=0.055, right=0.965, top=0.862, bottom=0.055,
                          hspace=0.52, wspace=0.26)

    panel_a(fig, gs[0, 0], order, mat, usage, contact, aa_index, contact_struct)
    panel_b(fig, gs[0, 1], order, bdf, roles, tags=labs)
    panel_c(fig, gs[1, 0], order, cdf, E)
    # colour panel-D residues by their expert's DNA-base-contact odds ratio from panel b,
    # i.e. directly by the scaffold-vs-readout axis the structural analysis established
    dcol = (bdf[bdf.feature == "base_contact"].set_index("expert").effect_size
            .reindex(range(E)).fillna(0.0).values)
    vmaxD = float(np.nanmax(np.abs(dcol)))
    normD = TwoSlopeNorm(0.0, -vmaxD * 1.02, vmaxD * 1.02)
    D_KEY = "expert DNA-base-contact log$_2$ OR (from b)"
    panel_d(fig, gs[1, 1], cases, dcol, normD, a.outdir, key_label=D_KEY)

    for lbl, x, y in (("a", 0.005, 0.930), ("b", 0.520, 0.930),
                      ("c", 0.005, 0.435), ("d", 0.520, 0.435)):
        fig.text(x, y, lbl, **PANEL_KW)

    fig.text(0.055, 0.975,
             "Mixture-of-experts separates structural scaffolding from "
             "DNA-recognition chemistry",
             fontsize=8.8, fontweight="bold", ha="left", va="top")
    fig.text(0.055, 0.951,
             "experts ordered along the observed DNA-contact axis in every panel: "
             "scaffold-enriched → mixed → DNA-interface-enriched",
             fontsize=6.2, color=GREY, style="italic", ha="left", va="top")

    handles = [
        Line2D([], [], marker="D", ls="", ms=2.4, color="#3B3B3B", label="Zn–Cys"),
        Line2D([], [], marker="s", ls="", ms=2.4, color="#7F8B96", label="Zn–His"),
        Line2D([], [], marker="o", ls="", ms=2.4, color="#AC4A31", label="DNA base contact"),
        Line2D([], [], marker="_", ls="", ms=4, mew=1.0, color="#2C5985",
               label="protein–protein interface"),
        Line2D([], [], marker="s", ls="", ms=3.0, mfc="none", mec="#333333", mew=0.6,
               label="canonical recognition position"),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=5, frameon=False,
               fontsize=5.6, bbox_to_anchor=(0.985, 0.014), handletextpad=0.35,
               columnspacing=0.9)

    base = os.path.join(a.outdir, "moe_biochemical_interpretability_main")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=600)
    print(f"wrote {base}.[pdf|svg|png]")

    # individual panels
    for name, fn, size, margins in (
            ("panel_A_chemistry",
             lambda f, c: panel_a(f, c, order, mat, usage, contact, aa_index,
                                  contact_struct),
             (4.4, 2.8), dict(left=0.10, right=0.90, top=0.80, bottom=0.19)),
            ("panel_B_structural_roles",
             lambda f, c: panel_b(f, c, order, bdf, roles, tags=labs),
             (5.4, 4.0), dict(left=0.10, right=0.86, top=0.82, bottom=0.27)),
            ("panel_C_corouting",
             lambda f, c: panel_c(f, c, order, cdf, E),
             (3.4, 3.0), dict(left=0.11, right=0.90, top=0.83, bottom=0.10)),
            ("panel_D_cases",
             lambda f, c: panel_d(f, c, cases, dcol, normD, a.outdir, key_label=D_KEY),
             (4.6, 3.2), dict(left=0.05, right=0.88, top=0.88, bottom=0.06)),
    ):
        f2 = plt.figure(figsize=size)
        g2 = f2.add_gridspec(1, 1, **margins)
        fn(f2, g2[0, 0])
        f2.savefig(os.path.join(a.outdir, name + ".pdf"))
        plt.close(f2)
        print(f"wrote {os.path.join(a.outdir, name)}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
