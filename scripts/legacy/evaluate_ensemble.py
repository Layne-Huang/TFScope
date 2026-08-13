#!/usr/bin/env python
"""Ensemble evaluation: average softmax outputs of 5 fold models on the blind benchmark.

Mirrors DeepPBS's 5-model ensemble protocol. For each test sample:
  - Forward through each of the 5 fold models (each with its own per-fold NN index)
  - Average the softmax(pwm_logits) across models
  - Compute Pearson r, MAE, IC-Pearson on the ensemble PWM

Also reports per-leakage-category breakdown (L0/L1/L2) for honest comparison
with DeepPBS, which performs equally on L0 and L2 (genuine generalization).
"""
import argparse, json, os, re, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, entropy as scipy_entropy

sys.path.insert(0, "src")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel


def per_position_pearson(pred, target):
    rs = []
    for i in range(pred.shape[1]):
        r = pearsonr(target[:, i], pred[:, i])[0]
        if not np.isnan(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def ic_corr_dp_formula(pred, target):
    """DeepPBS IC-Pearson: Pearson between per-position IC vectors."""
    BKG = [0.25, 0.25, 0.25, 0.25]
    ic_t = scipy_entropy(target.T, BKG, base=2, axis=1)
    ic_p = scipy_entropy(pred.T,   BKG, base=2, axis=1)
    if len(ic_t) < 2: return float("nan")
    r = pearsonr(ic_t, ic_p)[0]
    return float(r) if not np.isnan(r) else float("nan")


def parse_npz_source_id(entry_or_fn):
    """Return canonical source_id (matches our parquet column) from either an NPZ name
    or a parquet filename."""
    base = entry_or_fn.replace(".npz", "").replace(".txt", "")
    base = re.sub(r"\.v\d+$", "", base)
    m = re.search(r"(MA\d+)\.(\d+)$", base)
    if m: return f"{m.group(1)}.{m.group(2)}"
    m = re.search(r"\.([A-Z0-9][A-Z0-9\-]*_(?:HUMAN|MOUSE)\.H11MO\.\d+\.[A-Z])$", base)
    if m: return m.group(1)
    return None


def categorise(test_fn, df, cv_source_ids, cv_genes):
    """Classify a test row by leakage level vs CV training set."""
    row = df[df["filename"] == test_fn].iloc[0]
    src = row["source_id"]
    gene = str(row["gene_symbol"]).upper()
    if src in cv_source_ids: return "L2"
    if gene in cv_genes:     return "L1"
    return "L0"


def load_model(ckpt_path, retrieval_index_path, device, use_retrieval=True):
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    cfg = TFScopeConfig()
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        for k, v in saved.items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except: pass
    cfg.use_retrieval = use_retrieval
    if use_retrieval:
        cfg.retrieval_index_path = retrieval_index_path

    model = TFScopeModel(cfg, use_dummy_backbone=False).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    print(f"  Loaded {ckpt_path}  (epoch={ck.get('epoch','?')})  retrieval={use_retrieval}")
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default="data/processed/tf_pwm_deeppbs_only.parquet")
    ap.add_argument("--ckpt-dir-pattern",
                    default="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v8_fold{i}")
    ap.add_argument("--ckpt-file", default="ckpt_best.pt")
    ap.add_argument("--nn-index-pattern",
                    default="data/processed/nn_index_5fold/fold{i}.json")
    ap.add_argument("--out", default="results/tfscope_v8_ensemble")
    ap.add_argument("--no-retrieval", action="store_true",
                    help="Disable retrieval (use for v7 non-RAG ensemble)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(args.data)

    # The split files live at data/processed/splits/deeppbs_5fold/fold{i}.json
    def split_path(i):
        return f"data/processed/splits/deeppbs_5fold/fold{i}.json"

    # Test split (same across folds)
    with open(split_path(0)) as f:
        split0 = json.load(f)
    test_fns = sorted(split0["test"])
    print(f"Test set: {len(test_fns)} samples")

    # Compute leakage categories vs the union of all training data
    train_union = set()
    for i in range(5):
        with open(split_path(i)) as f:
            tr = json.load(f)["train"]
        train_union.update(tr)
    train_df = df[df["filename"].isin(train_union)]
    cv_src_ids = set(train_df["source_id"])
    cv_genes   = set(train_df["gene_symbol"].str.upper())
    print(f"Train union: {len(train_union)} structures, "
          f"{len(cv_src_ids)} unique source_ids, {len(cv_genes)} genes")

    # Forward through each fold model, accumulate softmax outputs per test sample
    ensemble_pwm = np.zeros((len(test_fns), 4, 20), dtype=np.float64)
    target_pwm   = np.zeros((len(test_fns), 4, 20), dtype=np.float32)
    pwm_mask     = np.zeros((len(test_fns), 20),    dtype=np.float32)

    for fold_i in range(5):
        ckpt_path = os.path.join(args.ckpt_dir_pattern.format(i=fold_i), args.ckpt_file)
        nn_idx    = args.nn_index_pattern.format(i=fold_i)
        if not os.path.exists(ckpt_path):
            print(f"  fold{fold_i}: missing {ckpt_path}, skipping")
            continue
        model, cfg = load_model(ckpt_path, nn_idx, device, use_retrieval=(not args.no_retrieval))

        split_path = f"data/processed/splits/deeppbs_5fold/fold{fold_i}.json"
        ds = TFDataset(cfg, args.data, split_path, split="test")
        # Lock the order to match test_fns
        order = [ds.filenames.index(fn) for fn in test_fns]
        from torch.utils.data import Subset
        ds = Subset(ds, order)
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                        collate_fn=collate_variable_length)
        idx = 0
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                         for k, v in batch.items()}
                _, pwm_logits, _ = model(
                    batch["sequence_tokens"], batch["dbd_mask"], batch["family_id"],
                    retrieved_pwms=batch.get("retrieved_pwms"),
                    retrieved_masks=batch.get("retrieved_masks"),
                    retrieved_sims=batch.get("retrieved_sims"),
                )
                pwm_prob = F.softmax(pwm_logits, dim=1).cpu().numpy()
                B = pwm_prob.shape[0]
                ensemble_pwm[idx:idx+B] += pwm_prob
                if fold_i == 0:
                    target_pwm[idx:idx+B] = batch["target_pwm"].cpu().numpy()
                    pwm_mask[idx:idx+B]   = batch["pwm_mask"].cpu().numpy()
                idx += B
        del model
        torch.cuda.empty_cache()

    # Average across folds (assume all 5 succeeded; otherwise normalise by count)
    n_models = sum(os.path.exists(os.path.join(args.ckpt_dir_pattern.format(i=i), args.ckpt_file))
                   for i in range(5))
    ensemble_pwm /= max(1, n_models)
    # re-normalise columns so they sum to 1 (after averaging across models)
    col_sums = ensemble_pwm.sum(axis=1, keepdims=True)
    ensemble_pwm = ensemble_pwm / np.clip(col_sums, 1e-8, None)
    print(f"\nEnsemble of {n_models} models")

    # Compute per-sample metrics on the TRUE length positions
    per_sample = []
    for i, fn in enumerate(test_fns):
        L = int(pwm_mask[i].sum())
        if L < 2: continue
        p = ensemble_pwm[i, :, :L].astype(np.float32)
        t = target_pwm[i, :, :L]
        per_sample.append({
            "filename":  fn,
            "category":  categorise(fn, df, cv_src_ids, cv_genes),
            "pearson_r": per_position_pearson(p, t),
            "mae":       float(np.abs(p - t).mean()),
            "ic_corr":   ic_corr_dp_formula(p, t),
        })

    # Aggregate overall + per-category
    def aggregate(samples):
        rs = np.array([s["pearson_r"] for s in samples])
        maes = np.array([s["mae"] for s in samples])
        ics = np.array([s["ic_corr"] for s in samples])
        return {
            "n": len(samples),
            "pearson_r_mean":   float(np.nanmean(rs)),
            "pearson_r_median": float(np.nanmedian(rs)),
            "mae_mean":         float(np.mean(maes)),
            "mae_median":       float(np.median(maes)),
            "mae_dp_x4":        float(np.mean(maes) * 4),
            "ic_corr_mean":     float(np.nanmean(ics)),
            "ic_corr_median":   float(np.nanmedian(ics)),
        }

    print(f"\nOverall (n={len(per_sample)}):")
    for k, v in aggregate(per_sample).items():
        print(f"  {k:<20} {v}")

    for cat in ["L2", "L1", "L0"]:
        subset = [s for s in per_sample if s["category"] == cat]
        if not subset: continue
        print(f"\n{cat} (n={len(subset)}):")
        for k, v in aggregate(subset).items():
            print(f"  {k:<20} {v}")

    # Save
    pd.DataFrame(per_sample).to_csv(os.path.join(args.out, "per_sample.csv"), index=False)
    summary = {
        "overall":     aggregate(per_sample),
        "by_category": {c: aggregate([s for s in per_sample if s["category"] == c])
                        for c in ["L2", "L1", "L0"]
                        if any(s["category"] == c for s in per_sample)},
        "n_models":    n_models,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
