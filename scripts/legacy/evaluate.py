#!/usr/bin/env python
"""Evaluate a trained TFScope checkpoint on a held-out test split.

Usage:
    python scripts/evaluate.py \
        --ckpt  /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_semantic_long/ckpt_best.pt \
        --split data/processed/splits/lofo/Homeodomain.json \
        --data  data/processed/tf_pwm.parquet \
        --out   results/lofo_homeodomain_semantic_100
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pwm_rosetta"))

os.environ.setdefault("TORCH_HOME",
                      "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")

from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from pwm_hybrid.pwm.viz import makeLogo


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--data",  default="data/processed/tf_pwm.parquet")
    p.add_argument("--out",   default="results/eval")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Gate sigmoid threshold for inferred motif length")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--retrieval-index-path", default=None,
                   help="Override the retrieval index from config (e.g. LSO index "
                        "to measure leakage-free performance)")
    p.add_argument("--deeppbs-dir", default=None,
                   help="Directory containing gene_preds.npz from eval_deeppbs.py. "
                        "When provided, adds a DeepPBS column to the top-5 figure.")
    return p.parse_args()


# ── Metrics ───────────────────────────────────────────────────────────────────
def information_content(pwm: np.ndarray) -> np.ndarray:
    """IC per position (bits).  pwm shape: (4, L)."""
    pwm = np.clip(pwm, 1e-8, 1.0)
    entropy = -np.sum(pwm * np.log2(pwm), axis=0)
    return 2.0 - entropy                           # max IC = 2 bits


def pearson_per_sample(pred: np.ndarray, target: np.ndarray,
                        mask: np.ndarray) -> float:
    """Per-position Pearson r (4-nucleotide vector per position), averaged.
    Matches DeepPBS evaluator exactly."""
    # pred, target: (4, L)  mask: (L,)  → iterate over valid positions
    idx = mask.astype(bool)
    p = pred[:, idx].T   # (valid_L, 4)
    t = target[:, idx].T
    if len(p) < 2:
        return float('nan')
    rs = [pearsonr(t[i], p[i])[0] for i in range(len(t))]
    return float(np.nanmean(rs))


def ic_weighted_pearson(pred: np.ndarray, target: np.ndarray,
                         mask: np.ndarray) -> float:
    """Pearson r weighted by target information content per position."""
    idx = mask.astype(bool)
    ic = information_content(target[:, idx])       # (valid_L,)
    w = ic / (ic.sum() + 1e-8)
    p = pred[:, idx]     # (4, valid_L)
    t = target[:, idx]
    # flatten with IC weights replicated across 4 nucleotides
    w4 = np.tile(w, 4)
    p4 = p.flatten();  t4 = t.flatten()
    # weighted correlation
    mp = np.sum(w4 * p4);  mt = np.sum(w4 * t4)
    cov = np.sum(w4 * (p4 - mp) * (t4 - mt))
    sp  = np.sqrt(np.sum(w4 * (p4 - mp)**2))
    st  = np.sqrt(np.sum(w4 * (t4 - mt)**2))
    if sp < 1e-8 or st < 1e-8:
        return float('nan')
    return cov / (sp * st)


def top1_accuracy(pred: np.ndarray, target: np.ndarray,
                   mask: np.ndarray) -> float:
    """Fraction of valid positions where argmax nucleotide matches target."""
    idx = mask.astype(bool)
    pred_top  = pred[:, idx].argmax(axis=0)
    targ_top  = target[:, idx].argmax(axis=0)
    return (pred_top == targ_top).mean()


def kl_divergence(pred: np.ndarray, target: np.ndarray,
                   mask: np.ndarray) -> float:
    """Mean KL(target || pred) over valid positions."""
    idx = mask.astype(bool)
    p = np.clip(pred[:, idx],   1e-8, 1.0)
    t = np.clip(target[:, idx], 1e-8, 1.0)
    kl = (t * (np.log(t) - np.log(p))).sum(axis=0)
    return kl.mean()


def pwm_mae(pred: np.ndarray, target: np.ndarray,
            mask: np.ndarray) -> float:
    """Mean absolute error between predicted and target PWM over valid positions.
    pred, target: (4, L)  mask: (L,)  — returns scalar in [0, 1]."""
    idx = mask.astype(bool)
    return np.abs(pred[:, idx] - target[:, idx]).mean()


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device, threshold):
    model.eval()
    all_gate   = []
    all_pwm    = []
    all_target = []
    all_mask   = []

    for batch in loader:
        batch = {k: v.to(device, dtype=torch.float32
                         if v.is_floating_point() else torch.long)
                 for k, v in batch.items()}
        gate_logits, pwm_logits, _ = model(
            batch['sequence_tokens'], batch['dbd_mask'], batch['family_id'],
            retrieved_pwms=batch.get('retrieved_pwms'),
            retrieved_masks=batch.get('retrieved_masks'),
            retrieved_sims=batch.get('retrieved_sims'),
        )

        gate_prob = gate_logits.sigmoid()                       # (B, 20)
        pwm_prob  = F.softmax(pwm_logits, dim=1)               # (B, 4, 20)

        all_gate.append(gate_prob.cpu().numpy())
        all_pwm.append(pwm_prob.cpu().numpy())
        all_target.append(batch['target_pwm'].cpu().numpy())
        all_mask.append(batch['pwm_mask'].cpu().numpy())

    return (np.concatenate(all_gate),
            np.concatenate(all_pwm),
            np.concatenate(all_target),
            np.concatenate(all_mask))


# ── Plots ─────────────────────────────────────────────────────────────────────
COLORS = {"pred": "#0072B2", "target": "#D55E00", "neutral": "#999999"}


def plot_pwm_logo(ax, pwm, mask, title, color):
    """IC sequence logo using pwm_rosetta makeLogo. pwm shape: (4, L)."""
    L = int(mask.sum())
    ppm = pwm[:, :L].T                                  # (L, 4) ACGT
    ppm = np.clip(ppm, 1e-8, 1.0)
    ppm = ppm / ppm.sum(axis=1, keepdims=True)
    makeLogo(ppm, ax)
    ax.set_ylim(0, 2)
    ax.set_ylabel("bits", fontsize=7)
    ax.set_xlabel("Position", fontsize=7)
    ax.set_title(title, fontsize=8, color=color, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.1f"))


def make_figures(gate, pwm_pred, pwm_targ, gt_mask, pred_mask,
                 pearson_r, ic_pearson, top1_acc, kl_div,
                 tf_names, out_dir, deeppbs_preds=None,
                 deeppbs_pearson_r=None, deeppbs_mae=None):
    os.makedirs(out_dir, exist_ok=True)
    figs = {}

    # ── 1. Metric distributions ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, vals, label, color in zip(
        axes,
        [pearson_r, ic_pearson, top1_acc, kl_div],
        ["Pearson r\n(PWM)", "IC-weighted\nPearson r", "Top-1\nAccuracy",
         "KL Divergence\n(target∥pred)"],
        [COLORS["pred"]] * 4,
    ):
        valid = [v for v in vals if not np.isnan(v)]
        ax.hist(valid, bins=25, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(np.nanmedian(vals), color="#D55E00", lw=1.5,
                   linestyle="--", label=f"median={np.nanmedian(vals):.3f}")
        ax.set_xlabel(label, fontsize=9);  ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7.5)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle("Per-sample metric distributions (test set)", fontsize=10,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "metric_distributions.pdf")
    fig.savefig(path, bbox_inches="tight"); figs["distributions"] = path
    plt.close(fig)

    # ── 2. Length prediction scatter ──────────────────────────────────────────
    true_len = gt_mask.sum(axis=1).astype(int)
    pred_len = (gate > 0.5).sum(axis=1).astype(int)  # wait — gate is pred_mask here

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_len, pred_len, alpha=0.4, s=18,
               color=COLORS["pred"], edgecolors="none")
    lim = (0, 22)
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
    r, _ = pearsonr(true_len, pred_len)
    mae  = np.abs(true_len - pred_len).mean()
    ax.set_xlabel("True motif length (bp)", fontsize=9)
    ax.set_ylabel("Predicted motif length (bp)", fontsize=9)
    ax.set_title(f"Motif length   r={r:.3f}   MAE={mae:.2f} bp", fontsize=9)
    ax.set_xlim(*lim);  ax.set_ylim(*lim)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "length_prediction.pdf")
    fig.savefig(path, bbox_inches="tight"); figs["length"] = path
    plt.close(fig)

    # ── 3. Gate heatmap (mean gate prob by position and true length) ──────────
    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_idx = np.argsort(true_len)
    im = ax.imshow(gate[sorted_idx], aspect="auto", cmap="Blues",
                   vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.02, label="Gate probability")
    ax.set_xlabel("Position", fontsize=9)
    ax.set_ylabel("Sample (sorted by true length)", fontsize=9)
    ax.set_title("Predicted position gate probabilities", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "gate_heatmap.pdf")
    fig.savefig(path, bbox_inches="tight"); figs["gate"] = path
    plt.close(fig)

    # ── helper: draw one row of logos ────────────────────────────────────────
    def _draw_row(ax_row, i, tf_r, dp_r=None):
        name = tf_names[i] if i < len(tf_names) else f"Sample {i}"
        gene = name.split(".")[0].upper()
        plot_pwm_logo(ax_row[0], pwm_targ[i], gt_mask[i],
                      f"{name}  —  Target  (len={int(gt_mask[i].sum())})",
                      COLORS["target"])
        plot_pwm_logo(ax_row[1], pwm_pred[i], pred_mask[i],
                      (f"TFScope  (len={int(pred_mask[i].sum())})  r={tf_r:.3f}"),
                      COLORS["pred"])
        if deeppbs_preds is not None:
            dp_pwm = deeppbs_preds.get(gene)
            dp_label = f"DeepPBS  r={dp_r:.3f}" if dp_r is not None else "DeepPBS"
            if dp_pwm is not None:
                dp_4xL = dp_pwm.T
                plot_pwm_logo(ax_row[2], dp_4xL, np.ones(dp_4xL.shape[1]),
                              dp_label + f"  (len={dp_pwm.shape[0]})",
                              COLORS["neutral"])
            else:
                ax_row[2].text(0.5, 0.5, f"No DeepPBS data\nfor {gene}",
                               ha="center", va="center",
                               transform=ax_row[2].transAxes,
                               fontsize=8, color=COLORS["neutral"])
                ax_row[2].set_title(f"DeepPBS — {gene} (no structure)",
                                    fontsize=8, color=COLORS["neutral"])
                ax_row[2].axis("off")

    # ── 4. Top-5 TFScope predictions (by Pearson r) ───────────────────────
    top_idx = np.argsort(pearson_r)[::-1][:5]
    n_cols = 3 if deeppbs_preds is not None else 2
    fig, axes = plt.subplots(5, n_cols,
                             figsize=(6 * n_cols, 12), constrained_layout=True)
    for row, i in enumerate(top_idx):
        dp_r = deeppbs_pearson_r[i] if deeppbs_pearson_r is not None else None
        _draw_row(axes[row], i, pearson_r[i], dp_r)

    title = "Top-5 TFScope predictions (by Pearson r)"
    if deeppbs_preds is not None:
        title += "  |  TFScope vs DeepPBS"
    fig.suptitle(title, fontsize=10, fontweight="bold")
    path = os.path.join(out_dir, "pwm_examples_top5.pdf")
    fig.savefig(path, bbox_inches="tight"); figs["top5"] = path
    plt.close(fig)

    # ── 5. TFScope >> DeepPBS: top-5 by (TFScope r − DeepPBS r) ─────────
    if deeppbs_pearson_r is not None and deeppbs_preds is not None:
        delta = pearson_r - deeppbs_pearson_r
        adv_idx = np.argsort(delta)[::-1][:5]
        fig, axes = plt.subplots(5, 3, figsize=(18, 12), constrained_layout=True)
        for row, i in enumerate(adv_idx):
            _draw_row(axes[row], i, pearson_r[i], deeppbs_pearson_r[i])
        fig.suptitle(
            "TFScope advantage cases — top-5 by (TFScope Pearson r − DeepPBS Pearson r)",
            fontsize=10, fontweight="bold")
        path = os.path.join(out_dir, "pwm_tfscope_advantage.pdf")
        fig.savefig(path, bbox_inches="tight"); figs["advantage"] = path
        plt.close(fig)

    # ── 6. TFScope MAE advantage: top-5 by (DeepPBS_norm_mae − TFScope_mae) ─
    # DeepPBS mae = mean_pos(sum_bases |p-t|) = 4 × per-element MAE, so /4 normalises
    if deeppbs_mae is not None and deeppbs_preds is not None:
        dp_mae_norm = deeppbs_mae / 4.0
        # per_sample mae stored in pwm_mae; passed in as ic_pearson's companion — use kl_div index
        # (pwm_mae not passed separately; we recompute inline from pwm_pred/targ)
        tf_mae = np.array([pwm_mae(pwm_pred[i], pwm_targ[i], gt_mask[i])
                           for i in range(len(pwm_pred))])
        mae_delta = dp_mae_norm - tf_mae   # positive = TFScope better
        adv_idx = np.argsort(mae_delta)[::-1][:5]
        fig, axes = plt.subplots(5, 3, figsize=(18, 12), constrained_layout=True)
        for row, i in enumerate(adv_idx):
            dp_r = deeppbs_pearson_r[i] if deeppbs_pearson_r is not None else None
            _draw_row(axes[row], i, pearson_r[i], dp_r)
            # annotate MAE delta on the target panel title
            axes[row][0].set_title(
                axes[row][0].get_title() +
                f"  |  ΔMAE={mae_delta[i]:+.3f}  (TF={tf_mae[i]:.3f}, DP={dp_mae_norm[i]:.3f})",
                fontsize=7.5, color=COLORS["target"])
        n_wins = int((mae_delta > 0).sum())
        fig.suptitle(
            f"TFScope MAE advantage — top-5 by (DeepPBS MAE − TFScope MAE), same scale\n"
            f"TFScope better on MAE in {n_wins}/{len(tf_mae)} samples",
            fontsize=10, fontweight="bold")
        path = os.path.join(out_dir, "pwm_tfscope_mae_advantage.pdf")
        fig.savefig(path, bbox_inches="tight"); figs["mae_advantage"] = path
        plt.close(fig)

    # ── 7. Prediction sharpness diagnostic ───────────────────────────────────
    # max nucleotide probability per valid position; ~0.25 = flat, ~1.0 = sharp
    pred_maxp, targ_maxp = [], []
    for i in range(len(pwm_pred)):
        idx = gt_mask[i].astype(bool)
        pred_maxp.extend(pwm_pred[i][:, idx].max(axis=0).tolist())
        targ_maxp.extend(pwm_targ[i][:, idx].max(axis=0).tolist())
    pred_maxp = np.array(pred_maxp)
    targ_maxp = np.array(targ_maxp)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(targ_maxp, bins=40, color=COLORS["target"], alpha=0.8, edgecolor="white")
    axes[0].axvline(np.median(targ_maxp), color="k", lw=1.5, linestyle="--",
                    label=f"median={np.median(targ_maxp):.3f}")
    axes[0].set_title("Target — max nucleotide prob/position", fontsize=9)
    axes[0].set_xlabel("Max probability", fontsize=9); axes[0].legend(fontsize=8)

    axes[1].hist(pred_maxp, bins=40, color=COLORS["pred"], alpha=0.8, edgecolor="white")
    axes[1].axvline(np.median(pred_maxp), color="k", lw=1.5, linestyle="--",
                    label=f"median={np.median(pred_maxp):.3f}")
    axes[1].set_title("TFScope — max nucleotide prob/position", fontsize=9)
    axes[1].set_xlabel("Max probability", fontsize=9); axes[1].legend(fontsize=8)

    axes[2].scatter(targ_maxp[:5000], pred_maxp[:5000],
                    alpha=0.15, s=6, color=COLORS["pred"], edgecolors="none")
    axes[2].plot([0.25, 1], [0.25, 1], "k--", lw=1, alpha=0.5)
    r_sharp, _ = pearsonr(targ_maxp, pred_maxp)
    axes[2].set_xlabel("Target max prob", fontsize=9)
    axes[2].set_ylabel("Predicted max prob", fontsize=9)
    axes[2].set_title(f"Sharpness correlation  r={r_sharp:.3f}", fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle("Prediction sharpness: does TFScope learn nucleotide specificity?",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "sharpness_diagnostic.pdf")
    fig.savefig(path, bbox_inches="tight"); figs["sharpness"] = path
    plt.close(fig)

    return figs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # Load config saved alongside checkpoint
    cfg_path = os.path.join(os.path.dirname(args.ckpt), "config.json")
    config = TFScopeConfig()
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        for k, v in saved.items():
            if hasattr(config, k):
                try: setattr(config, k, type(getattr(config, k))(v))
                except: pass

    # Optional retrieval-index override (e.g. LSO index for leakage-free eval)
    if args.retrieval_index_path:
        config.retrieval_index_path = args.retrieval_index_path
        print(f"Overriding retrieval index → {args.retrieval_index_path}")

    # Load model
    model = TFScopeModel(config, use_dummy_backbone=False).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing:
        print(f"Missing keys (backbone not in ckpt — will be loaded from ESM-2): "
              f"{len(missing)} keys")

    print(f"Loaded checkpoint: epoch={ckpt['epoch']}  "
          f"val_loss={ckpt.get('best_val_loss', 'N/A')}")

    # ── Detect unseen family IDs and patch with mean of trained embeddings ────
    # In LOFO evaluation the held-out family's embedding was never trained.
    # Passing a random embedding to FamilyEmbedding / FamilyAwareGating causes
    # the MoE to route entirely via noise → near-zero Pearson r.
    # Fix: replace any unseen family's weights with the mean over trained ones.
    df_all_tmp = pd.read_parquet(args.data)
    with open(args.split) as _f:
        _split_tmp = json.load(_f)
    trained_family_ids = sorted(
        df_all_tmp[df_all_tmp["filename"].isin(_split_tmp["train"])]["family_id"]
        .unique().tolist()
    )
    test_family_ids = sorted(
        df_all_tmp[df_all_tmp["filename"].isin(_split_tmp["test"])]["family_id"]
        .unique().tolist()
    )
    unseen = [fid for fid in test_family_ids if fid not in trained_family_ids]
    if unseen:
        print(f"Unseen family IDs in test set: {unseen}  "
              f"→ patching MoE embeddings with mean of trained families {trained_family_ids}")
        trained_t = torch.tensor(trained_family_ids, device=device)
        with torch.no_grad():
            fe = model.moe.family_embed
            if hasattr(fe, "embedding"):          # LearnedFamilyEmbedding
                mean_fam_embed = fe.embedding.weight[trained_t].mean(0)
                for fid in unseen:
                    fe.embedding.weight[fid] = mean_fam_embed
            else:                                 # SemanticFamilyEmbedding
                mean_fam_embed = fe.vectors[trained_t].mean(0)
                for fid in unseen:
                    fe.vectors[fid] = mean_fam_embed
    del df_all_tmp, _split_tmp

    # Test dataset
    test_ds = TFDataset(config, args.data, args.split, split="test",
                        max_seq_len=args.max_seq_len)
    df_all  = pd.read_parquet(args.data)
    with open(args.split) as f:
        split_info = json.load(f)
    test_filenames = split_info["test"]
    tf_names = df_all[df_all["filename"].isin(test_filenames)]["tf_name"].tolist()

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_variable_length)
    print(f"Test samples: {len(test_ds)}")

    # Inference
    gate, pwm_pred, pwm_targ, gt_mask = run_inference(
        model, loader, device, args.threshold)
    pred_mask = (gate > args.threshold).astype(float)

    # ── Strand-symmetric orientation ──────────────────────────────────────────
    # DNA motifs are strand-symmetric: a site and its reverse complement are the
    # same physical site, and a target PWM may be deposited on either strand.
    # DeepPBS's own evaluator scores both strands; to match it (and not penalise
    # biologically-correct reverse-complement predictions, e.g. CEBPB 2e42),
    # flip each prediction to whichever strand best matches the target, then
    # compute ALL metrics on that orientation.
    def _revcomp(pwm):
        return pwm[[3, 2, 1, 0], ::-1]

    N = len(gate)
    n_flipped = 0
    for i in range(N):
        m = gt_mask[i].astype(bool)
        L = int(m.sum())
        if L < 2:
            continue
        # forward prediction restricted to valid (left-aligned) positions
        rf = pearson_per_sample(pwm_pred[i], pwm_targ[i], gt_mask[i])
        # reverse-complement the valid window in place, then re-left-align
        rc_full = pwm_pred[i].copy()
        rc_full[:, :L] = _revcomp(pwm_pred[i][:, :L])
        rr = pearson_per_sample(rc_full, pwm_targ[i], gt_mask[i])
        if (not np.isnan(rr)) and (np.isnan(rf) or rr > rf):
            pwm_pred[i] = rc_full
            n_flipped += 1
    print(f"Strand-symmetric scoring: reverse-complemented {n_flipped}/{N} predictions "
          f"to best-matching strand")

    # ── Motif-level metric (offset + reverse-complement aligned) ───────────────
    # Standard motif comparison (TOMTOM/STAMP): align the predicted PWM to the
    # target by best offset AND orientation, then score. Measures whether the
    # correct MOTIF was recovered, independent of where the model placed it in
    # the output window. Complements the strict registration-exact pearson_r.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from tfscope.models.alignment import align_pwm
    motif_r = np.full(N, np.nan)
    for i in range(N):
        L = int(gt_mask[i].sum())
        if L < 2:
            continue
        al, _, _, _ = align_pwm(pwm_pred[i][:, :L], pwm_targ[i][:, :L], max_shift=6)
        Lc = min(al.shape[1], L)
        rs = [pearsonr(pwm_targ[i][:, j], al[:, j])[0] for j in range(Lc)]
        motif_r[i] = np.nanmean(rs)

    # Per-sample metrics
    pearson_r  = np.array([pearson_per_sample(pwm_pred[i], pwm_targ[i], gt_mask[i])
                            for i in range(N)])
    ic_pearson = np.array([ic_weighted_pearson(pwm_pred[i], pwm_targ[i], gt_mask[i])
                            for i in range(N)])
    top1_acc   = np.array([top1_accuracy(pwm_pred[i], pwm_targ[i], gt_mask[i])
                            for i in range(N)])
    kl_div     = np.array([kl_divergence(pwm_pred[i], pwm_targ[i], gt_mask[i])
                            for i in range(N)])
    mae        = np.array([pwm_mae(pwm_pred[i], pwm_targ[i], gt_mask[i])
                            for i in range(N)])

    true_len = gt_mask.sum(axis=1).astype(int)
    pred_len = pred_mask.sum(axis=1).astype(int)
    len_mae  = np.abs(true_len - pred_len).mean()
    len_r, _ = pearsonr(true_len, pred_len)

    # Summary
    summary = {
        "n_samples":           int(N),
        "pearson_r_mean":      float(np.nanmean(pearson_r)),
        "pearson_r_median":    float(np.nanmedian(pearson_r)),
        "motif_level_r_mean":  float(np.nanmean(motif_r)),
        "motif_level_r_median":float(np.nanmedian(motif_r)),
        "ic_pearson_mean":     float(np.nanmean(ic_pearson)),
        "ic_pearson_median":   float(np.nanmedian(ic_pearson)),
        "top1_accuracy_mean":  float(np.nanmean(top1_acc)),
        "pwm_mae_mean":        float(np.nanmean(mae)),
        "pwm_mae_median":      float(np.nanmedian(mae)),
        "kl_divergence_mean":  float(np.nanmean(kl_div)),
        "length_mae_bp":       float(len_mae),
        "length_pearson_r":    float(len_r),
    }

    print("\n" + "="*55)
    print(f"  TEST RESULTS  —  {split_info.get('metadata', {}).get('held_out_family_name', '')}")
    print("="*55)
    for k, v in summary.items():
        print(f"  {k:<28} {v:.4f}")
    print("="*55)

    # Save summary
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-sample results
    per_sample = pd.DataFrame({
        "tf_name":    tf_names[:N],
        "true_length": true_len,
        "pred_length": pred_len,
        "pearson_r":  pearson_r,
        "motif_level_r": motif_r,
        "ic_pearson": ic_pearson,
        "top1_acc":   top1_acc,
        "pwm_mae":    mae,
        "kl_div":     kl_div,
    })
    per_sample.to_csv(os.path.join(args.out, "per_sample.csv"), index=False)
    print(f"\nPer-sample results → {args.out}/per_sample.csv")

    # Load DeepPBS predictions if provided
    deeppbs_preds = None
    deeppbs_pearson_r = None
    deeppbs_mae_arr = None
    if args.deeppbs_dir:
        npz_path = os.path.join(args.deeppbs_dir, "gene_preds.npz")
        if os.path.exists(npz_path):
            raw = np.load(npz_path, allow_pickle=True)
            deeppbs_preds = {k: raw[k] for k in raw.files}
            print(f"Loaded DeepPBS predictions for {len(deeppbs_preds)} genes "
                  f"from {npz_path}")
        else:
            print(f"Warning: gene_preds.npz not found in {args.deeppbs_dir}")

        # Load per-sample DeepPBS metrics and match to TFScope samples by gene
        ps_path = os.path.join(args.deeppbs_dir, "per_sample.json")
        if os.path.exists(ps_path):
            import json as _json
            dp_samples = _json.load(open(ps_path))
            # best pearson_r and lowest mae per gene (multiple structures may exist)
            dp_gene_r, dp_gene_mae = {}, {}
            for s in dp_samples:
                g = s.get("gene", "").upper()
                if not g:
                    continue
                if g not in dp_gene_r or s["pearson_r"] > dp_gene_r[g]:
                    dp_gene_r[g] = s["pearson_r"]
                if g not in dp_gene_mae or s["mae"] < dp_gene_mae[g]:
                    dp_gene_mae[g] = s["mae"]
            genes = [n.split(".")[0].upper() for n in tf_names[:N]]
            deeppbs_pearson_r = np.array([
                dp_gene_r.get(g, float("nan")) for g in genes])
            deeppbs_mae_arr = np.array([
                dp_gene_mae.get(g, float("nan")) for g in genes])
            n_matched = np.sum(~np.isnan(deeppbs_pearson_r))
            print(f"Matched DeepPBS per-sample metrics for {n_matched}/{N} samples")

    # Figures
    figs = make_figures(gate, pwm_pred, pwm_targ, gt_mask, pred_mask,
                        pearson_r, ic_pearson, top1_acc, kl_div,
                        tf_names, args.out, deeppbs_preds=deeppbs_preds,
                        deeppbs_pearson_r=deeppbs_pearson_r,
                        deeppbs_mae=deeppbs_mae_arr)
    for name, path in figs.items():
        print(f"Figure [{name}] → {path}")


if __name__ == "__main__":
    main()
