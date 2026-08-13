#!/usr/bin/env python
"""Plot sequence logo comparison for all 130 test samples.

For each entry, draws 3 logos side-by-side:
  [Ground Truth]  [DeepPBS]  [pwm_rosetta (ours)]

Ground truth:  Y_pwm[0][pwm_mask[0]] from DeepPBS assembly NPZ
DeepPBS pred:  gene_preds.npz (per-gene prediction, case-insensitive lookup)
Ours:          scan CSV → Boltzmann PPM, aligned to GT length

Output: figures/logo_comparison.pdf  (multi-page, 5 entries per page)

Usage:
    python scripts/plot_logo_comparison.py
    python scripts/plot_logo_comparison.py --out figures/logo_comparison.pdf
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logomaker
from scipy.stats import pearsonr

CRYSTAL_RUN_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/crystal_test"
SPLIT           = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
DEEPPBS_NPZ_DIR = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/deeppbsmar24/data/assembly2024"
GENE_PREDS      = "results/deeppbs_blind_benchmark/gene_preds.npz"
BASES           = ["A", "C", "G", "T"]
TAU             = 1.5
ENTRIES_PER_PAGE = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_deeppbs_npz(tid, npz_dir):
    entry = tid.replace(".txt", "")
    parts = entry.split("_")
    prefix = parts[0] + "_" + parts[1] + "_"
    suffix = "_".join(parts[2:])
    motif_id = suffix.split(".", 1)[1] if "." in suffix else suffix
    cands = [f for f in os.listdir(npz_dir) if f.startswith(prefix)]
    if not cands:
        return None
    if len(cands) == 1:
        return os.path.join(npz_dir, cands[0])
    for c in cands:
        if motif_id in c[len(prefix):].replace(".npz", ""):
            return os.path.join(npz_dir, c)
    return os.path.join(npz_dir, cands[0])


def ppm_from_csv(csv_path, tau=TAU):
    df = pd.read_csv(csv_path)
    pos_df = df.drop_duplicates("position").sort_values("position")
    positions = pos_df["position"].tolist()
    seq = "".join(pos_df["original"].tolist())
    L = len(seq)
    pos_to_idx = {p: i for i, p in enumerate(positions)}
    wt_ddg = df["wt_ddg"].iloc[0] if "wt_ddg" in df.columns else df["mut_ddg"].min()
    energies = np.full((L, 4), wt_ddg)
    for i, b in enumerate(seq):
        energies[i][BASES.index(b)] = wt_ddg
    for _, row in df.iterrows():
        idx = pos_to_idx.get(int(row["position"]))
        if idx is None:
            continue
        if row["mutant"] != row["original"]:
            energies[idx][BASES.index(row["mutant"])] = row["mut_ddg"]
    dE = energies - energies.min(axis=1, keepdims=True)
    PPM = np.exp(-dE / tau)
    PPM /= PPM.sum(axis=1, keepdims=True)
    PPM = np.clip(PPM, 1e-6, 1.0)
    PPM /= PPM.sum(axis=1, keepdims=True)
    return PPM


def best_align(pred, true_len):
    """Return slice of pred aligned to length true_len by best Pearson r."""
    if len(pred) <= true_len:
        return pred
    # Can't align without true; just return first true_len positions
    # (caller passes true for proper alignment)
    return pred[:true_len]


def best_align_to(pred, true):
    """Slide pred over true, return best-aligned slice of length len(true)."""
    K = len(true)
    if len(pred) <= K:
        return pred
    flat_t = true.flatten()
    best_r, best_s = -2.0, 0
    for s in range(len(pred) - K + 1):
        flat_p = pred[s:s+K].flatten()
        if flat_p.std() < 1e-8 or flat_t.std() < 1e-8:
            continue
        r = pearsonr(flat_p, flat_t)[0]
        if r > best_r:
            best_r, best_s = r, s
    return pred[best_s:best_s+K]


def ppm_to_ic_df(ppm, bases=None):
    """Convert (L,4) PPM to IC-scaled DataFrame for logomaker."""
    if bases is None:
        bases = ["A", "C", "G", "T"]
    p = np.clip(ppm, 1e-10, 1.0)
    ic_per_pos = (p * np.log2(p / 0.25)).sum(axis=1)  # (L,)
    ic_mat = p * ic_per_pos[:, None]                    # scale cols by IC
    df = pd.DataFrame(ic_mat, columns=bases)
    return df


def draw_logo(ax, ppm, title, r_val=None, color_scheme="classic"):
    """Draw a sequence logo on ax from (L,4) PPM."""
    if ppm is None or len(ppm) == 0:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title(title, fontsize=8, pad=2)
        ax.axis("off")
        return

    ic_df = ppm_to_ic_df(ppm)
    try:
        logomaker.Logo(ic_df, ax=ax, color_scheme=color_scheme,
                       show_spines=False, baseline_width=0.5)
    except Exception:
        ax.text(0.5, 0.5, "render err", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="red")

    ax.set_ylim(bottom=0)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.set_xticks(range(len(ppm)))
    ax.set_xticklabels(range(1, len(ppm) + 1), fontsize=5)

    title_str = title
    if r_val is not None and not np.isnan(r_val):
        title_str += f"  r={r_val:.2f}"
    ax.set_title(title_str, fontsize=7, pad=2)


def pearson_ppm(pred, true):
    L = min(len(pred), len(true))
    p, t = pred[:L].flatten(), true[:L].flatten()
    if p.std() < 1e-8 or t.std() < 1e-8:
        return float("nan")
    return float(pearsonr(p, t)[0])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",  default=CRYSTAL_RUN_DIR)
    ap.add_argument("--split",    default=SPLIT)
    ap.add_argument("--npz-dir",  default=DEEPPBS_NPZ_DIR)
    ap.add_argument("--gene-preds", default=GENE_PREDS)
    ap.add_argument("--out", default="figures/logo_comparison.pdf")
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.split) as f:
        test_ids = json.load(f)["test"]

    # Load DeepPBS gene-level predictions (case-insensitive lookup)
    gene_preds_npz = np.load(args.gene_preds, allow_pickle=True)
    gene_preds = {k.upper(): gene_preds_npz[k] for k in gene_preds_npz.files}

    # Sort entries by gene name for grouped display
    def gene_key(tid):
        parts = tid.split("_")
        return parts[2].split(".")[0].upper() if len(parts) > 2 else tid
    test_ids_sorted = sorted(test_ids, key=gene_key)

    print(f"Plotting {len(test_ids_sorted)} entries → {args.out}")

    with PdfPages(args.out) as pdf:
        # Process in pages of ENTRIES_PER_PAGE
        for page_start in range(0, len(test_ids_sorted), ENTRIES_PER_PAGE):
            page_entries = test_ids_sorted[page_start:page_start + ENTRIES_PER_PAGE]
            n_rows = len(page_entries)

            fig, axes = plt.subplots(
                n_rows, 3,
                figsize=(13, 2.2 * n_rows),
                gridspec_kw={"hspace": 0.7, "wspace": 0.35},
            )
            if n_rows == 1:
                axes = axes[None, :]   # ensure 2D

            for row_i, tid in enumerate(page_entries):
                entry = tid.replace(".txt", "")
                parts = entry.split("_")
                gene = parts[2].split(".")[0] if len(parts) > 2 else "?"
                pdb_chain = parts[0] + "_" + parts[1]

                ax_gt, ax_dp, ax_us = axes[row_i]

                # ── Ground truth ──────────────────────────────────────────
                gt_ppm = None
                npz_path = pick_deeppbs_npz(tid, args.npz_dir)
                if npz_path:
                    try:
                        npz = np.load(npz_path, allow_pickle=True)
                        Y_pwm = npz["Y_pwm"]
                        mask  = npz["pwm_mask"]
                        gt_ppm = Y_pwm[0][mask[0]]   # (K, 4)
                    except Exception:
                        gt_ppm = None

                # ── DeepPBS prediction ────────────────────────────────────
                dp_ppm = None
                gene_upper = gene.upper()
                # also try stripping trailing digit or isoform suffix
                for key in [gene_upper, gene_upper.replace("ALPHA2",""),
                             gene_upper + "1", gene_upper[:-1]]:
                    if key in gene_preds:
                        dp_ppm = gene_preds[key]
                        break
                # Align to GT length if needed
                if dp_ppm is not None and gt_ppm is not None:
                    dp_ppm = best_align_to(dp_ppm, gt_ppm)

                # ── Our prediction ────────────────────────────────────────
                csv_path = os.path.join(args.run_dir, entry, "pwm_results_hybrid.csv")
                our_ppm = None
                if os.path.exists(csv_path):
                    try:
                        our_ppm = ppm_from_csv(csv_path, tau=args.tau)
                        if gt_ppm is not None:
                            our_ppm = best_align_to(our_ppm, gt_ppm)
                    except Exception:
                        our_ppm = None

                # Pearson r vs GT
                r_dp = pearson_ppm(dp_ppm, gt_ppm) if dp_ppm is not None and gt_ppm is not None else None
                r_us = pearson_ppm(our_ppm, gt_ppm) if our_ppm is not None and gt_ppm is not None else None

                # ── Draw logos ────────────────────────────────────────────
                row_label = f"{pdb_chain}  {gene}  (K={len(gt_ppm) if gt_ppm is not None else '?'})"
                draw_logo(ax_gt, gt_ppm, f"GT  {row_label}")
                draw_logo(ax_dp, dp_ppm, "DeepPBS", r_val=r_dp)
                draw_logo(ax_us, our_ppm, "pwm_rosetta", r_val=r_us)

            page_num = page_start // ENTRIES_PER_PAGE + 1
            total_pages = (len(test_ids_sorted) + ENTRIES_PER_PAGE - 1) // ENTRIES_PER_PAGE
            fig.text(0.5, 0.01, f"Page {page_num}/{total_pages}", ha="center",
                     fontsize=8, color="gray")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  page {page_num}/{total_pages}", end="\r")

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
