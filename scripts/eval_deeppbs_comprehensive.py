#!/usr/bin/env python
"""Comprehensive metric panel for TFScope on the DeepPBS 130-structure blind test.

Extends scripts/eval_vs_deeppbs_blind.py with:
  - Probabilistic metrics: cross-entropy, KL(true||pred), top-1 base accuracy,
    per-base ROC-AUC (treating GT argmax as positive label)
  - Distribution statistics (mean / median / std / IQR / min / max) for every metric
  - Per-family breakdown of the headline metrics

Ground truth = Y_pwm from DeepPBS npz (same as DeepPBS's own evaluation).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_deeppbs_comprehensive.py \\
        --run-dir /data1/leihuang/project/TFScope/checkpoints/v19_deeppbs_benchmark/rag_seed42 \\
        --out results/v19_deeppbs/benchmark_comprehensive.json
"""
import argparse, json, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")

import pandas as pd
from tfscope.data.dataset import TFDataset, collate_variable_length

# reuse helpers + constants from the existing fair-comparison script
from eval_vs_deeppbs_blind import (
    DATA_DIR, FOLDS_DIR, DEFAULT_PARQUET, DEFAULT_SPLIT,
    MIN_POS, MAX_SHIFT,
    build_npz_to_parquet_map, deeppbs_metrics, oracle_metrics, load_model,
)


# ── Probabilistic metrics ────────────────────────────────────────────────────

def prob_metrics(pred_L4, true_L4):
    """Cross-entropy, KL(true||pred), top-1 base accuracy on the fixed overlap."""
    p = np.clip(pred_L4, 1e-8, 1.0); p = p / p.sum(axis=1, keepdims=True)
    t = np.clip(true_L4, 1e-8, 1.0); t = t / t.sum(axis=1, keepdims=True)
    ce  = float(np.mean(-(t * np.log(p)).sum(axis=1)))
    kl  = float(np.mean((t * (np.log(t) - np.log(p))).sum(axis=1)))
    top1 = float(np.mean(np.argmax(p, axis=1) == np.argmax(t, axis=1)))
    return ce, kl, top1


def auc_base(pred_L4, true_L4):
    """Per-base ROC-AUC: label = 1 if base is GT argmax at that position."""
    labels = (np.argmax(true_L4, axis=1)[:, None] ==
              np.arange(4)[None, :]).astype(int).ravel()
    scores = pred_L4.ravel()
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def dist_stats(arr):
    a = np.array(arr, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {k: float("nan") for k in
                ("mean", "median", "std", "q25", "q75", "min", "max", "n")}
    return {
        "mean":   float(np.mean(a)),
        "median": float(np.median(a)),
        "std":    float(np.std(a)),
        "q25":    float(np.percentile(a, 25)),
        "q75":    float(np.percentile(a, 75)),
        "min":    float(np.min(a)),
        "max":    float(np.max(a)),
        "n":      int(len(a)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="ckpt_best.pt")
    p.add_argument("--out", default="results/v19_deeppbs/benchmark_comprehensive.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--data",  default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--eval-split", default="test", choices=("train", "val", "test"))
    p.add_argument("--test-list", default=None,
                   help="Path to a file of npz names to evaluate (default: blind id.txt)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    parquet = args.data  if args.data  else DEFAULT_PARQUET
    split   = args.split if args.split else DEFAULT_SPLIT

    df = pd.read_parquet(parquet)
    fn2family = dict(zip(df["filename"], df.get("family_name", df["filename"])))
    npz_to_parquet = build_npz_to_parquet_map(df)

    test_list = args.test_list if args.test_list else os.path.join(FOLDS_DIR, "id.txt")
    with open(test_list) as fh:
        blind_npz = [l.strip() for l in fh if l.strip()]
    blind_pairs = [(n, npz_to_parquet[n]) for n in blind_npz if n in npz_to_parquet]
    print(f"Blind structures mapped: {len(blind_pairs)}/{len(blind_npz)}")

    model, cfg = load_model(args.run_dir, args.checkpoint, device)

    ds = TFDataset(cfg, parquet, split, split=args.eval_split, max_seq_len=1024)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_variable_length)
    preds_cache = {}
    row_offset = 0
    with torch.no_grad():
        for batch in loader:
            bs = batch["family_id"].shape[0]
            metadata = ds.df.iloc[row_offset: row_offset + bs]
            row_offset += bs
            batch = {k: v.to(device, dtype=torch.float32 if v.is_floating_point()
                             else torch.long) for k, v in batch.items()}
            gate_logits, pwm_logits, _ = model(
                batch["sequence_tokens"], batch["dbd_mask"], batch["family_id"],
                retrieved_pwms=batch.get("retrieved_pwms"),
                retrieved_masks=batch.get("retrieved_masks"),
                retrieved_sims=batch.get("retrieved_sims"),
                recog_prior=batch.get("recog_prior"),
            )
            pwm_prob = F.softmax(pwm_logits, dim=1).cpu().numpy()
            for i, meta in enumerate(metadata.itertuples(index=False)):
                preds_cache[str(meta.filename)] = pwm_prob[i]
    print(f"Cached predictions for {len(preds_cache)} test records\n")

    # accumulate metrics
    M = {k: [] for k in
         ("fixed_mae", "fixed_r", "fixed_ic_r",
          "panel_r", "ali_mae", "ali_r", "ali_ic_r",
          "ce", "kl", "top1", "auc")}
    per_family = {}   # family → {metric: [vals]}
    per_record = []

    for npz_name, pf in blind_pairs:
        npz_path = os.path.join(DATA_DIR, npz_name)
        if not os.path.exists(npz_path) or pf not in preds_cache:
            continue
        d = np.load(npz_path, allow_pickle=True)
        Y_pwm = d["Y_pwm"].astype(np.float64); pwm_mask = d["pwm_mask"]
        strand = 0 if int(pwm_mask[0].sum()) >= int(pwm_mask[1].sum()) else 1
        true_L4 = Y_pwm[strand][pwm_mask[strand]]
        if true_L4.shape[0] < MIN_POS:
            continue
        true_L4 = np.clip(true_L4, 1e-8, 1.0); true_L4 /= true_L4.sum(1, keepdims=True)

        pred_4L = preds_cache[pf]
        L = min(pred_4L.shape[1], true_L4.shape[0])
        pred_fixed = np.full((true_L4.shape[0], 4), 0.25, dtype=np.float64)
        pred_fixed[:L] = pred_4L[:, :L].T
        pred_fixed = np.clip(pred_fixed, 1e-8, 1.0); pred_fixed /= pred_fixed.sum(1, keepdims=True)

        f_mae, f_r, f_ic_r = deeppbs_metrics(pred_fixed, true_L4)
        ce, kl, top1 = prob_metrics(pred_fixed, true_L4)
        auc = auc_base(pred_fixed, true_L4)
        res = oracle_metrics(pred_4L, true_L4.T)
        panel_r, ali_mae, ali_r, ali_ic_r = res if res else (np.nan,)*4

        rec = dict(fixed_mae=f_mae, fixed_r=f_r, fixed_ic_r=f_ic_r,
                   panel_r=panel_r, ali_mae=ali_mae, ali_r=ali_r, ali_ic_r=ali_ic_r,
                   ce=ce, kl=kl, top1=top1, auc=auc)
        for k, v in rec.items():
            M[k].append(v)
        fam = fn2family.get(pf, "Unknown")
        per_family.setdefault(fam, {k: [] for k in M})
        for k, v in rec.items():
            per_family[fam][k].append(v)
        per_record.append({"npz": npz_name, "parquet": pf, "family": fam, **rec})

    # ── summary tables ───────────────────────────────────────────────────────
    summary = {k: dist_stats(v) for k, v in M.items()}

    def line(name, s, fmt="{:.4f}"):
        return (f"  {name:<12} "
                f"mean={fmt.format(s['mean'])}  med={fmt.format(s['median'])}  "
                f"std={fmt.format(s['std'])}  "
                f"IQR=[{fmt.format(s['q25'])},{fmt.format(s['q75'])}]")

    n = summary["fixed_r"]["n"]
    print(f"=== Comprehensive metrics — TFScope on DeepPBS blind test (n={n}) ===\n")
    print("[Fixed-frame — DeepPBS protocol, no alignment]")
    print(line("MAE",   summary["fixed_mae"]))
    print(line("per-pos r", summary["fixed_r"]))
    print(line("IC-r",  summary["fixed_ic_r"]))
    print("\n[Oracle-aligned — fair for sequence-only models, ±10 + RC]")
    print(line("panel-r",  summary["panel_r"]))
    print(line("ali-MAE",  summary["ali_mae"]))
    print(line("ali per-pos r", summary["ali_r"]))
    print(line("ali-IC-r", summary["ali_ic_r"]))
    print("\n[Probabilistic — fixed overlap]")
    print(line("cross-ent", summary["ce"]))
    print(line("KL(t||p)",  summary["kl"]))
    print(line("top-1 acc", summary["top1"]))
    print(line("base AUC",  summary["auc"]))

    print(f"\n[Per-family breakdown — oracle panel-r / fixed per-pos r / top-1]")
    fam_summary = {}
    print(f"  {'Family':<18} {'n':>3}  {'panel-r':>8} {'fixed-r':>8} {'top-1':>7} {'AUC':>7}")
    for fam in sorted(per_family, key=lambda f: -len(per_family[f]["fixed_r"])):
        fm = {k: dist_stats(v) for k, v in per_family[fam].items()}
        fam_summary[fam] = fm
        print(f"  {fam:<18} {fm['fixed_r']['n']:>3}  "
              f"{fm['panel_r']['mean']:>8.3f} {fm['fixed_r']['mean']:>8.3f} "
              f"{fm['top1']['mean']:>7.3f} {fm['auc']['mean']:>7.3f}")

    print(f"\n[DeepPBS 5-model ensemble, published, same 130 structures]")
    print(f"  per-position Pearson r ≈ 0.702 (fixed crystal frame)")

    out = {
        "checkpoint": os.path.join(args.run_dir, args.checkpoint),
        "test_set": "DeepPBS blind test (id.txt)",
        "n_structures": n,
        "ground_truth": "Y_pwm from DeepPBS npz",
        "summary": summary,
        "per_family": fam_summary,
        "deeppbs_published": {"r_mean": 0.702,
                              "note": "5-model ensemble, fixed crystal frame"},
        "per_record": per_record,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
