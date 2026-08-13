#!/usr/bin/env python
"""Minimal amino-acid sequence strip for a rendered complex: show the DBD
sequence and mark only the essential DNA-contact residues.

Usage:
  <py> scripts/plot_contact_sequence_strip.py \
     --residues results/pymol_investigation/1B72_ESM/1B72_esm_residues.csv \
     --title "1B72 chain B (Homeodomain)" \
     --out results/pymol_investigation/1B72_ESM/1B72_esm_sequence_strip
The residues CSV needs: amino_acid, label, and one of {author_resid, position}.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ORANGE, GREY, INK = "#E76F51", "#B8B8B8", "#333333"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residues", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--numbering", default="author_resid",
                    choices=["author_resid", "position"])
    ap.add_argument("--per-row", type=int, default=45)
    args = ap.parse_args()

    df = pd.read_csv(args.residues).sort_values(args.numbering).reset_index(drop=True)
    aa = df["amino_acid"].tolist()
    num = df[args.numbering].astype(int).tolist()
    lab = df["label"].astype(int).tolist()
    n = len(aa)
    per = args.per_row
    nrows = (n + per - 1) // per

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "monospace", "figure.dpi": 300})

    fig, ax = plt.subplots(figsize=(0.24 * per + 0.6, 1.05 * nrows + 0.5))
    row_h = 1.6
    for i in range(n):
        r, c = divmod(i, per)
        y = -r * row_h
        is_c = lab[i] == 1
        col = ORANGE if is_c else GREY
        # residue letter
        ax.text(c, y, aa[i], ha="center", va="center", fontsize=11,
                fontweight="bold" if is_c else "normal", color=col, family="monospace")
        # marker + number only on contacts
        if is_c:
            ax.scatter([c], [y + 0.62], marker="v", s=26, color=ORANGE,
                       edgecolors="none", zorder=3)
            ax.text(c, y - 0.7, str(num[i]), ha="center", va="center",
                    fontsize=6.5, color=ORANGE, rotation=90, family="monospace")
        # sparse ruler ticks every 10 residues (grey)
        if num[i] % 10 == 0:
            ax.text(c, y - 0.7, str(num[i]), ha="center", va="center",
                    fontsize=6.0, color="#AAAAAA", rotation=90, family="monospace")

    ax.set_xlim(-1, per)
    ax.set_ylim(-(nrows - 1) * row_h - 1.1, 1.1)
    ax.axis("off")
    if args.title:
        ax.set_title(args.title, fontsize=10, family="sans-serif", pad=8)
    contacts = [f"{aa[i]}{num[i]}" for i in range(n) if lab[i] == 1]
    ax.text(0, 1.05 - (nrows - 1) * row_h - 1.0 + 0.0, "", fontsize=1)  # spacer
    fig.text(0.5, 0.015, "contacts: " + "  ".join(contacts), ha="center",
             fontsize=7, color=ORANGE, family="monospace")
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"{args.out}.png", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight", facecolor="white")
    print("saved", args.out + ".png/.pdf")
    print("sequence:", "".join(aa))
    print("contacts:", " ".join(contacts))


if __name__ == "__main__":
    main()
