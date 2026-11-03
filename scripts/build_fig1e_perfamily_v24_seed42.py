#!/usr/bin/env python
"""Fig 1e, redrawn: per-family motif recovery of the v24 seed42 single model, under the
CORRECTED DBD taxonomy, on the two available test surfaces -- one figure each.

Why two figures and not one
---------------------------
The two surfaces are complementary, not nested, and neither alone supports a per-family
claim:

  deeppbs_cluster40 test   84 rows /  38 genes / 16 families -- but 33 of 38 genes are
                           in the v23 training set, so most bars are leaked
  v24 clean surface       560 rows /  95 genes /  8 families -- zero leakage, but half
  (canonical test +        the family coverage, and 4 of the 8 families are the
   excluded buffer)        near-zero-shot ones (Forkhead / ETS / POU / p53)

So the wide-coverage surface is contaminated and the clean surface is narrow. Leaked
genes are drawn hollow on the first figure so the honest subset stays visible; the
properly powered per-family answer is the LOFO experiment, not either of these.

Taxonomy: scripts/reclassify_tf_families.py (Lambert-2018 curated), C2H2_short/medium/
long merged into one C2H2 -- the old figure used the broken scheme where `Other` was
the largest bucket and C2H2 was split three ways.

Metric: iclr/unified_eval Panel A -- the prediction's gate-committed core is registered
to the IC-trimmed ground-truth core with an oracle shift + reverse-complement search,
then scored as mean per-column Pearson r over the overlap.

  python scripts/build_fig1e_perfamily_v24_seed42.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from iclr.baselines import _decode_pwm                       # noqa: E402
from iclr.ladder_full_metrics import SEED42                  # noqa: E402
from iclr.score_v24_ensemble import predict_ensemble         # noqa: E402
from iclr.unified_eval import panel_A, trimmed_core          # noqa: E402
from reclassify_tf_families import classify                  # noqa: E402

V23 = "data/processed/tf_pwm_training_v23.parquet"
CANON = "data/processed/splits/train_v22/split.json"
DPBS = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
DPBS_SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
OUTD = "figures_v24/figure1e_heldout_perfamily"

# same palette as the other v24 panels (Model.pdf family, desaturated)
BLUE = "#5C4FAA"          # clean genes
BLUE_FILL = "#8E84C9"     # box fill
GREY_GRID = "#dcdcdc"
INK, INK2 = "#1a1a1a", "#4d4d4d"
MM = 1 / 25.4


def set_theme():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans",
                            "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "custom", "mathtext.rm": "sans",
        "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold", "mathtext.cal": "sans",
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 7, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7,
        "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 0.0,
        "legend.frameon": False, "legend.fontsize": 6.8,
    })


def taxonomy_fn(v23):
    """gene symbol -> corrected DBD family (C2H2_* merged)."""
    g = v23.gene_symbol.fillna("").astype(str).str.upper()
    seed = {gg: fn for gg, fn in zip(g, v23.family_name)
            if fn in ("C2H2_short", "C2H2_medium", "C2H2_long")}
    def f(sym):
        x = classify(str(sym).upper(), seed)
        return "C2H2" if str(x).startswith("C2H2") else x
    return f


def score_surface(ckpt, parquet, filenames, device):
    """Panel-A content_r per row for one model over an arbitrary row set."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"test": sorted(filenames)}, tmp); tmp.close()
    try:
        preds, _ = predict_ensemble([ckpt], parquet, tmp.name, device)
    finally:
        os.unlink(tmp.name)
    df = pd.read_parquet(parquet); df["filename"] = df.filename.astype(str)
    sub = df[df.filename.isin(preds)].set_index("filename")
    out = []
    for fn, p in preds.items():
        core = trimmed_core(_decode_pwm(sub.loc[fn].pwm))
        if core is None:
            continue
        out.append({"filename": fn,
                    "gene": str(sub.loc[fn].gene_symbol).upper(),
                    "r": panel_A(p, core)["content_r"]})
    return pd.DataFrame(out)


def draw(df, stem, subtitle, mark_leaked):
    """One horizontal box+strip per family, families ordered by median r."""
    fams = (df.groupby("fam").r.median().sort_values().index.tolist())
    data = [df[df.fam == f].r.values for f in fams]
    h = max(52, 12 * len(fams) + 34)
    fig, ax = plt.subplots(figsize=(120 * MM, h * MM))
    fig.subplots_adjust(left=0.30, right=0.90, top=0.90, bottom=0.135)

    y = np.arange(1, len(fams) + 1)
    bp = ax.boxplot(data, vert=False, widths=0.58, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color=INK, lw=1.0),
                    boxprops=dict(facecolor=BLUE_FILL, edgecolor="#7d72bd", lw=0.6,
                                  alpha=0.55),
                    whiskerprops=dict(color="#9a9a9a", lw=0.6),
                    capprops=dict(color="#9a9a9a", lw=0.6))
    rng = np.random.RandomState(0)
    for i, f in enumerate(fams):
        s = df[df.fam == f]
        jitter = rng.normal(i + 1, 0.075, len(s))
        if mark_leaked:
            cl, lk = ~s.leaked.values, s.leaked.values
            ax.scatter(s.r.values[cl], jitter[cl], s=9, facecolor=BLUE,
                       edgecolor="none", zorder=3)
            ax.scatter(s.r.values[lk], jitter[lk], s=9, facecolor="none",
                       edgecolor=BLUE, linewidths=0.6, zorder=3)
        else:
            ax.scatter(s.r.values, jitter, s=9, facecolor=BLUE, edgecolor="none",
                       zorder=3)
        n_g = s.gene.nunique()
        ax.text(1.015, i + 1, f"{len(s)} / {n_g}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=6.2, color=INK2, clip_on=False)

    med = float(df.r.median())
    ax.axvline(med, color="#c0392b", ls="--", lw=0.8, zorder=1)
    # left of the line, not centred on it: the median sits near the right edge and a
    # centred label runs into the "rows / genes" column header
    ax.text(med - 0.012, len(fams) + 0.72, f"overall median {med:.2f}", color="#c0392b",
            fontsize=6.5, ha="right", va="bottom")

    ax.set_yticks(y); ax.set_yticklabels(fams, color=INK)
    ax.set_ylim(0.35, len(fams) + 0.95)
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("motif recovery (oracle-aligned Pearson $r$)")
    ax.xaxis.grid(True, color=GREY_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.text(1.015, len(fams) + 0.72, "rows / genes",
            transform=ax.get_yaxis_transform(), va="bottom", ha="left",
            fontsize=6.2, color=INK)
    ax.set_title(subtitle, loc="left", fontsize=7.8, color=INK, pad=7)

    if mark_leaked:
        hh = [plt.Line2D([], [], marker="o", ls="", ms=4, mfc=BLUE, mec="none",
                         label="gene not in v23 training"),
              plt.Line2D([], [], marker="o", ls="", ms=4, mfc="none", mec=BLUE,
                         label="gene in v23 training (leaked)")]
        ax.legend(handles=hh, loc="upper center", bbox_to_anchor=(0.44, -0.075),
                  ncol=2, handletextpad=0.3, columnspacing=1.4)

    os.makedirs(OUTD, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(f"{OUTD}/{stem}.{ext}", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUTD}/{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {OUTD}/{stem}.pdf/.svg/.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    set_theme()

    v23 = pd.read_parquet(V23); v23["filename"] = v23.filename.astype(str)
    fam_of = taxonomy_fn(v23)
    canon = json.load(open(CANON))
    train_genes = set(v23[v23.filename.isin(set(canon["train"]))]
                      .gene_symbol.astype(str).str.upper())

    # ── surface 1: the DeepPBS cluster40 benchmark ───────────────────────────
    dsp = json.load(open(DPBS_SPLIT))
    d1 = score_surface(SEED42, DPBS, set(dsp["test"]), a.device)
    d1["fam"] = d1.gene.map(fam_of)
    d1["leaked"] = d1.gene.isin(train_genes)
    n_lk = d1.groupby("gene").leaked.first().sum()
    print(f"deeppbs_cluster40 test: {len(d1)} rows / {d1.gene.nunique()} genes / "
          f"{d1.fam.nunique()} families | leaked genes {n_lk}/{d1.gene.nunique()}")
    draw(d1, "figure1e_perfamily_deeppbs_test",
         f"v24 (seed 42) per-family motif recovery — DeepPBS cluster40 test\n"
         f"{len(d1)} PWMs / {d1.gene.nunique()} TFs / {d1.fam.nunique()} families; "
         f"{n_lk} of {d1.gene.nunique()} genes are in v23 training",
         mark_leaked=True)

    # ── surface 2: v24's own clean surface (canonical test + excluded) ───────
    fns = (set(canon["test"]) | set(canon["excluded"])) & set(v23.filename)
    d2 = score_surface(SEED42, V23, fns, a.device)
    d2["fam"] = d2.gene.map(fam_of)
    assert not d2.gene.isin(train_genes).any(), "clean surface must not touch train genes"
    print(f"v24 clean surface: {len(d2)} rows / {d2.gene.nunique()} genes / "
          f"{d2.fam.nunique()} families | leaked genes 0")
    draw(d2, "figure1e_perfamily_v24_test",
         f"v24 (seed 42) per-family motif recovery — v24 held-out surface\n"
         f"{len(d2)} PWMs / {d2.gene.nunique()} TFs / {d2.fam.nunique()} families; "
         f"never trained on, never selected on",
         mark_leaked=False)

    both = pd.concat([d1.assign(surface="deeppbs_cluster40_test"),
                      d2.assign(surface="v24_clean_test_plus_excluded", leaked=False)])
    p = f"{OUTD}/source_data_perfamily.csv"
    both.round(4).to_csv(p, index=False)
    print(f"  {p}")
    for name, d in [("deeppbs_cluster40_test", d1), ("v24_clean", d2)]:
        t = d.groupby("fam").agg(rows=("r", "size"), genes=("gene", "nunique"),
                                 median_r=("r", "median")).round(3)
        print(f"\n=== {name} ===\n{t.sort_values('median_r', ascending=False).to_string()}")


if __name__ == "__main__":
    main()
