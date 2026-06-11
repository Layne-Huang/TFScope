#!/usr/bin/env python
"""Plot IC-based sequence logos for the top-5 samples (by Pearson r) using
pwm_rosetta's makeLogo.

Usage:
    python scripts/plot_logo_top5.py \
        --ckpt  checkpoints/lofo_homeodomain/ckpt_epoch001.pt \
        --split data/processed/splits/lofo/Homeodomain.json \
        --data  data/processed/tf_pwm.parquet \
        --out   results/lofo_homeodomain
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
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pwm_rosetta"))

os.environ.setdefault("TORCH_HOME",
                      "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")

from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from pwm_hybrid.pwm.viz import makeLogo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--data",  default="data/processed/tf_pwm.parquet")
    p.add_argument("--out",   default="results/eval")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--topk", type=int, default=5,
                   help="Number of top samples to plot")
    return p.parse_args()


def information_content(pwm: np.ndarray) -> np.ndarray:
    """IC per position (bits). pwm shape: (4, L)."""
    pwm = np.clip(pwm, 1e-8, 1.0)
    entropy = -np.sum(pwm * np.log2(pwm), axis=0)
    return 2.0 - entropy


def pearson_per_sample(pred, target, mask):
    idx = mask.astype(bool)
    p = pred[:, idx].flatten()
    t = target[:, idx].flatten()
    if len(p) < 2:
        return float("nan")
    r, _ = pearsonr(p, t)
    return r


def ic_weighted_pearson(pred, target, mask):
    idx = mask.astype(bool)
    ic = information_content(target[:, idx])
    w = ic / (ic.sum() + 1e-8)
    p = pred[:, idx]; t = target[:, idx]
    w4 = np.tile(w, 4)
    p4 = p.flatten(); t4 = t.flatten()
    mp = np.sum(w4 * p4); mt = np.sum(w4 * t4)
    cov = np.sum(w4 * (p4 - mp) * (t4 - mt))
    sp = np.sqrt(np.sum(w4 * (p4 - mp)**2))
    st = np.sqrt(np.sum(w4 * (t4 - mt)**2))
    if sp < 1e-8 or st < 1e-8:
        return float("nan")
    return cov / (sp * st)


@torch.no_grad()
def run_inference(model, loader, device, threshold):
    model.eval()
    all_gate, all_pwm, all_target, all_mask = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device, dtype=torch.float32
                         if v.is_floating_point() else torch.long)
                 for k, v in batch.items()}
        gate_logits, pwm_logits, _ = model(
            batch["sequence_tokens"], batch["dbd_mask"], batch["family_id"])
        all_gate.append(gate_logits.sigmoid().cpu().numpy())
        all_pwm.append(F.softmax(pwm_logits, dim=1).cpu().numpy())
        all_target.append(batch["target_pwm"].cpu().numpy())
        all_mask.append(batch["pwm_mask"].cpu().numpy())
    return (np.concatenate(all_gate), np.concatenate(all_pwm),
            np.concatenate(all_target), np.concatenate(all_mask))


def plot_logo_panel(pwm_4xL, mask, ax, title, color):
    """Trim PWM to valid length, transpose to (L,4) for makeLogo."""
    L = int(mask.sum())
    ppm = pwm_4xL[:, :L].T          # (L, 4), ACGT order
    ppm = np.clip(ppm, 1e-8, 1.0)
    ppm = ppm / ppm.sum(axis=1, keepdims=True)  # renormalise rows
    makeLogo(ppm, ax)
    ax.set_title(title, fontsize=8, color=color, fontweight="bold")
    ax.set_ylabel("bits", fontsize=7)
    ax.set_xlabel("Position", fontsize=7)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.set_ylim(0, 2)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    cfg_path = os.path.join(os.path.dirname(args.ckpt), "config.json")
    config = TFScopeConfig()
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        for k, v in saved.items():
            if hasattr(config, k):
                try: setattr(config, k, type(getattr(config, k))(v))
                except: pass

    model = TFScopeModel(config, use_dummy_backbone=False).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, _ = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"Missing keys (loaded from ESM-2): {len(missing)}")
    print(f"Checkpoint: epoch={ckpt['epoch']}  "
          f"val_loss={ckpt.get('best_val_loss', 'N/A')}")

    test_ds = TFDataset(config, args.data, args.split, split="test")
    df_all  = pd.read_parquet(args.data)
    with open(args.split) as f:
        split_info = json.load(f)
    test_filenames = split_info["test"]
    tf_names = df_all[df_all["filename"].isin(test_filenames)]["tf_name"].tolist()

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_variable_length)
    print(f"Test samples: {len(test_ds)}")

    gate, pwm_pred, pwm_targ, gt_mask = run_inference(
        model, loader, device, args.threshold)
    pred_mask = (gate > args.threshold).astype(float)

    N = len(gate)
    pearson_r  = np.array([pearson_per_sample(pwm_pred[i], pwm_targ[i], gt_mask[i])
                            for i in range(N)])
    ic_pearson = np.array([ic_weighted_pearson(pwm_pred[i], pwm_targ[i], gt_mask[i])
                           for i in range(N)])

    top_idx = np.argsort(pearson_r)[::-1][:args.topk]

    fig, axes = plt.subplots(args.topk, 2,
                             figsize=(12, 3.5 * args.topk),
                             constrained_layout=True)

    TARGET_COLOR = "#D55E00"
    PRED_COLOR   = "#0072B2"

    for row, i in enumerate(top_idx):
        name = tf_names[i] if i < len(tf_names) else f"Sample {i}"

        plot_logo_panel(
            pwm_targ[i], gt_mask[i], axes[row, 0],
            title=f"{name}  —  Target  (len={int(gt_mask[i].sum())})",
            color=TARGET_COLOR,
        )
        plot_logo_panel(
            pwm_pred[i], pred_mask[i], axes[row, 1],
            title=(f"Predicted  (len={int(pred_mask[i].sum())})  "
                   f"r={pearson_r[i]:.3f}  IC-r={ic_pearson[i]:.3f}"),
            color=PRED_COLOR,
        )

    fig.suptitle(f"Top-{args.topk} PWM predictions — IC sequence logos",
                 fontsize=11, fontweight="bold")

    out_path = os.path.join(args.out, f"logo_top{args.topk}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
