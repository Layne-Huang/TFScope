#!/usr/bin/env python
"""Full-length logo-plot grid for the deployable AF3 before/after comparison (4 ZFs).

Rows per ZF: Ground truth | TFScope RAG-best (folded) | Rosetta-AF3 | MM-GBSA-AF3 | FoldX-AF3.
Each predicted PWM is shown at its FULL length (only oriented — revcomp if that aligns
better). The ground-truth register is indicated by a shaded span + the true bases
annotated above the aligned columns, instead of cropping the logo to the GT window.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
import pandas as pd

BASES = ["A", "C", "G", "T"]
TRUE = {"zf21": "ATTAGAAGTGA", "zf93": "CTTTGTTGGCA",
        "zf92": "AGATGTAGCCT", "zf129": "CGGGGACGTCA"}
OUT = "results/zf_struct/af3_logos.png"


def onehot(seq):
    idx = {b: i for i, b in enumerate(BASES)}
    M = np.zeros((len(seq), 4), np.float32)
    for i, b in enumerate(seq):
        if b in idx:
            M[i, idx[b]] = 1.0
    return M


def revcomp(p):
    return p[::-1, ::-1].copy()


def as_Lx4(a):
    a = np.asarray(a, float)
    if a.shape[0] == 4 and a.shape[1] != 4:
        a = a.T
    s = a.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return a / s


def load_rag_best(zf):
    """TFScope's canonical RAG prediction (cluster40_v18a_rag) — the PWM that was
    actually folded by AF3. Read from results/zf_struct/tfscope_rag_best/<zf>.pwm.tsv."""
    import csv
    rows = []
    with open(f"results/zf_struct/tfscope_rag_best/{zf}.pwm.tsv") as fh:
        rd = csv.reader(fh, delimiter="\t")
        next(rd)
        for r in rd:
            if r:
                rows.append([float(x) for x in r[1:5]])
    return as_Lx4(np.array(rows))


def oracle_orient(pred, gt):
    """Return (oriented full-length pred, off, r). GT col j aligns to pred index j-off."""
    Lg = gt.shape[0]

    def best(p):
        Lp = p.shape[0]
        br, bo = -2.0, 0
        for off in range(-(Lp - 1), Lg):
            rs = []
            for j in range(Lg):
                i = j - off
                if 0 <= i < Lp:
                    a, b = p[i], gt[j]
                    if a.std() > 1e-8 and b.std() > 1e-8:
                        rs.append(np.corrcoef(a, b)[0, 1])
            if len(rs) >= 4:
                m = float(np.mean(rs))
                if m > br:
                    br, bo = m, off
        return br, bo

    rf, of = best(pred)
    rr, orr = best(revcomp(pred))
    if rr > rf:
        return revcomp(pred), orr, rr
    return pred, of, rf


def ic_df(ppm):
    ppm = np.clip(ppm, 1e-9, 1.0)
    ppm = ppm / ppm.sum(1, keepdims=True)
    ic_total = (2.0 + (ppm * np.log2(ppm)).sum(1))
    heights = ppm * ic_total[:, None]
    return pd.DataFrame(heights, columns=BASES)


def draw(ax, ppm, title, r=None, off=None, gt_seq=None):
    df = ic_df(ppm)
    logomaker.Logo(df, ax=ax, color_scheme="classic")
    ax.set_ylim(0, 2.05)
    ax.set_xticks([])
    ax.set_yticks([0, 1, 2])
    ax.set_ylabel("bits", fontsize=7)
    lab = title if r is None else f"{title}  (r={r:.2f})"
    ax.set_title(lab, fontsize=9, loc="left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    # indicate ground-truth register: GT col j -> pred index j-off
    if off is not None and gt_seq is not None:
        Lp = ppm.shape[0]
        idxs = [(j - off) for j in range(len(gt_seq)) if 0 <= (j - off) < Lp]
        if idxs:
            ax.axvspan(min(idxs) - 0.5, max(idxs) + 0.5, color="gold", alpha=0.16, zorder=0)
            for j, b in enumerate(gt_seq):
                i = j - off
                if 0 <= i < Lp:
                    ax.text(i, 2.16, b, ha="center", va="bottom", fontsize=7,
                            color="#444", family="monospace")


def main():
    zfs = ["zf21", "zf93", "zf92", "zf129"]
    methods = ["Ground truth", "TFScope RAG-best (folded)", "Rosetta-AF3",
               "MM-GBSA-AF3", "FoldX-AF3"]
    ncol, nrow = len(zfs), len(methods)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 1.85 * nrow))

    def load_draw(ax, path, gt, label):
        if path and os.path.exists(path):
            p = as_Lx4(np.load(path))
            op, off, r = oracle_orient(p, gt)
            draw(ax, op, label, r, off, TRUE_curr)
        else:
            ax.text(0.5, 0.5, "not run", ha="center", va="center",
                    fontsize=10, color="grey", transform=ax.transAxes)
            ax.set_title(label, fontsize=9, loc="left")
            ax.set_xticks([]); ax.set_yticks([])

    for c, n in enumerate(zfs):
        TRUE_curr = TRUE[n]
        gt = onehot(TRUE_curr)
        # ground truth (full one-hot, no shading needed)
        draw(axes[0][c], gt, f"{n}")
        axes[0][c].set_title(f"{n}   GT = {TRUE_curr}", fontsize=10,
                             loc="left", fontweight="bold")

        # TFScope canonical RAG prediction (cluster40_v18a_rag) — the folded PWM
        rag = load_rag_best(n)
        op, off, r = oracle_orient(rag, gt)
        draw(axes[1][c], op, methods[1], r, off, TRUE_curr)

        load_draw(axes[2][c], f"results/zf_struct/pwm_rosetta_af3/{n}/pwm.npy", gt, methods[2])
        load_draw(axes[3][c], f"results/zf_struct/mmpbsa_evolve_af3/{n}_round0_pwm.npy", gt, methods[3])
        load_draw(axes[4][c], f"results/zf_struct/foldx_evolve_af3/{n}_round0_pwm.npy", gt, methods[4])

    fig.suptitle("Deployable AF3 PWM prediction (full-length logos) — gold span + letters = "
                 "ground-truth register  |  designed zinc fingers", fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
