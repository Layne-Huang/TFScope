#!/usr/bin/env python
"""Evaluate DeepPBS 5-model ensemble on the benchmark test set.

Reproduces the DeepPBS paper's evaluation exactly:
  - Applies dna_mask to model output, pwm_mask to ground truth
  - MAE   = mean over positions of sum(|pred-true|) across 4 bases
  - Pearson R = per-position Pearson R (4-element vectors), averaged
  - IC corr = Pearson R between IC profiles (using scipy KL-divergence IC)

Usage:
    python scripts/eval_deeppbs.py --out results/deeppbs_eval
"""

import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from os.path import join as ospj

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, entropy
from torch_geometric.data import DataLoader

DEEPPBS_DIR = "/n/home13/leihuang/project/DeepPBS"
sys.path.insert(0, DEEPPBS_DIR)
sys.path.insert(0, ospj(DEEPPBS_DIR, "run"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from deeppbs.nn.utils import loadDataset
from deeppbs.nn import processBatch
from models.model_v2 import Model
from make_deeppbs_splits import parse_tf_id

FOLDS_DIR    = ospj(DEEPPBS_DIR, "run/folds")
DATA_DIR     = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/deeppbsmar24/data/assembly2024"
RUN_DIR      = ospj(DEEPPBS_DIR, "run")
CKPT_NAMES   = [l.strip() for l in open(ospj(RUN_DIR, "plot_scripts/txts/DeepPBS.txt"))]
CKPT_PATHS   = [ospj(RUN_DIR, "output", c, "Model.best.tar") for c in CKPT_NAMES]
SCALER_PATHS = [ospj(RUN_DIR, "output", c, "scaler.pkl")     for c in CKPT_NAMES]
CONFIG       = {"nc": 4, "labels_key": "Y_pwm", "cache_dataset": False,
                "balance": "unmasked", "condition": "prot_shape", "readout": "all"}
BKG          = [0.25, 0.25, 0.25, 0.25]


def ic(pwm):
    """Per-position IC (bits) via KL divergence, matching DeepPBS's IC_corr."""
    return entropy(pwm, BKG, base=2, axis=1)


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """pred and true are (L×4) masked PWMs (already selected by dna_mask/pwm_mask).
    Formulas match DeepPBS evaluator.py exactly."""
    # MAE: sum |pred-true| across 4 bases, then mean over positions
    mae = float(np.mean(np.sum(np.abs(pred - true), axis=1)))
    # Pearson R: per-position (4-element), then averaged
    rs = [pearsonr(true[i], pred[i])[0] for i in range(len(true))]
    r  = float(np.nanmean(rs))
    # IC Pearson
    ic_true = ic(true)
    ic_pred = ic(pred)
    with np.errstate(invalid="ignore"):
        ic_r = float(pearsonr(ic_true, ic_pred)[0]) if len(ic_true) > 1 else float("nan")
    return {"mae": mae, "pearson_r": r, "ic_pearson": ic_r}


def load_npz_list(txt_paths):
    npzs = set()
    for p in txt_paths:
        with open(p) as f:
            npzs.update(l.strip() for l in f if l.strip())
    return sorted(npzs)


def gene_from_npz(name: str, df_j_lookup: dict) -> str:
    """Return uppercase gene symbol for an NPZ basename, or empty string."""
    kind, ident = parse_tf_id(name)
    if kind == "hocomoco":
        return ident.upper()
    elif kind == "jaspar":
        genes = df_j_lookup.get(ident.upper(), set())
        return next(iter(genes), "").upper()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/deeppbs_eval")
    parser.add_argument("--npz-list", default=None,
                        help="Text file with NPZ basenames (one per line). "
                             "Defaults to union of all 5 validation folds.")
    parser.add_argument("--parquet", default="data/processed/tf_pwm.parquet",
                        help="TFScope parquet (for JASPAR→gene mapping).")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build JASPAR MA→gene lookup
    df = pd.read_parquet(args.parquet)
    df_j = df[df["source"] == "JASPAR"].copy()
    df_j["ma"] = df_j["source_id"].str.extract(r"(MA\d+)", expand=False,
                                                flags=re.IGNORECASE).str.upper()
    jaspar_lookup = (df_j.groupby("ma")["gene_symbol"]
                     .apply(lambda x: set(x.str.upper())).to_dict())

    if args.npz_list:
        with open(args.npz_list) as f:
            npz_names = [l.strip() for l in f if l.strip()]
    else:
        valid_txts = [ospj(FOLDS_DIR, f"valid{i}.txt") for i in range(5)]
        npz_names  = load_npz_list(valid_txts)
    print(f"Test samples: {len(npz_names)}")

    # Load scalers and models
    scalers = [pickle.load(open(p, "rb")) for p in SCALER_PATHS]
    models = []
    for ckpt in CKPT_PATHS:
        m = Model(13, 14, condition=CONFIG["condition"], readout=CONFIG["readout"])
        m.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        m.to(device).eval()
        models.append(m)
    print(f"Loaded {len(models)} models")

    # Load datasets — one per scaler (each model was trained with its own scaler)
    DLs, loaded_names_list = [], []
    for i, scaler in enumerate(scalers):
        dataset, _, _, loaded_names = loadDataset(
            npz_names, CONFIG["nc"], CONFIG["labels_key"], DATA_DIR,
            cache_dataset=False, balance=CONFIG["balance"],
            remove_mask=False, scale=True, scaler=scaler)
        DL = list(DataLoader(dataset, batch_size=1, shuffle=False))
        DLs.append(DL)
        loaded_names_list.append(loaded_names)
        print(f"  scaler {i}: {len(DL)} samples")

    n_samples   = len(DLs[0])
    sample_names = [os.path.basename(n) for n in loaded_names_list[0]]

    per_sample = []
    for idx in range(n_samples):
        # ── collect all model outputs ──────────────────────────────────────
        model_outputs = []
        for dl_idx in range(len(DLs)):
            batch = DLs[dl_idx][idx].to(device)
            bd = processBatch(device, batch)
            with torch.no_grad():
                logits = models[dl_idx](bd["batch"])
                prob = torch.softmax(logits, dim=1).cpu().numpy()
            model_outputs.append(prob)

        # ── ensemble: simple average across 5 models ──────────────────────
        ensemble = np.mean(model_outputs, axis=0)   # (2*dna_len, 4)

        # ── retrieve masks from un-moved batch (first DL) ─────────────────
        batch0 = DLs[0][idx]
        dna_mask0 = batch0.dna_mask0.cpu().numpy()
        dna_mask1 = batch0.dna_mask1.cpu().numpy()
        pwm_mask0 = batch0.pwm_mask0.cpu().numpy()
        pwm_mask1 = batch0.pwm_mask1.cpu().numpy()
        y_pwm0    = batch0.y_pwm0.cpu().numpy()
        y_pwm1    = batch0.y_pwm1.cpu().numpy()

        # ── select masked positions — exactly as DeepPBS evaluator does ───
        dna_len  = dna_mask0.shape[0]
        out_fwd  = ensemble[:dna_len][dna_mask0]   # (n_valid_fwd, 4)
        out_rev  = ensemble[dna_len:][dna_mask1]   # (n_valid_rev, 4)
        true_fwd = y_pwm0[pwm_mask0]
        true_rev = y_pwm1[pwm_mask1]

        pred = np.concatenate([out_fwd, out_rev], axis=0)
        true = np.concatenate([true_fwd, true_rev], axis=0)

        m = compute_metrics(pred, true)
        m["name"] = sample_names[idx]
        per_sample.append(m)

        # Store raw forward-strand prediction (L×4) keyed by gene
        gene = gene_from_npz(sample_names[idx], jaspar_lookup)
        m["gene"] = gene
        # Use only the forward-strand masked prediction for per-gene averaging
        per_sample[-1]["_pred_fwd"] = out_fwd.tolist()
        # Per-STRUCTURE prediction + target, keyed by sample name (PDB_chain_motif)
        # so it can be matched to TFScope test filenames without gene-symbol loss.
        per_sample[-1]["_pred_struct"] = out_fwd
        per_sample[-1]["_true_struct"] = true_fwd

    # ── save per-gene predictions for figure use ──────────────────────────
    # Multiple PDB chains per gene can produce different-length masked arrays;
    # we average only when shapes match, otherwise keep the first prediction.
    gene_preds = defaultdict(list)
    for s in per_sample:
        g = s.get("gene", "")
        if g:
            gene_preds[g].append(np.array(s["_pred_fwd"]))
    gene_preds_avg = {}
    for g, arrs in gene_preds.items():
        shapes = [a.shape for a in arrs]
        if len(set(shapes)) == 1:
            gene_preds_avg[g] = np.mean(arrs, axis=0)
        else:
            gene_preds_avg[g] = arrs[0]   # use first if shapes differ
    np.savez_compressed(ospj(args.out, "gene_preds.npz"), **gene_preds_avg)

    # ── save PER-STRUCTURE predictions + targets (no gene-symbol loss) ────────
    struct_out = {}
    for s in per_sample:
        nm = s["name"]
        struct_out[nm + "::pred"] = s["_pred_struct"].astype(np.float32)
        struct_out[nm + "::true"] = s["_true_struct"].astype(np.float32)
    np.savez_compressed(ospj(args.out, "struct_preds.npz"), **struct_out)
    print(f"Saved per-structure predictions for {len(per_sample)} structures → struct_preds.npz")

    # Remove temporary arrays from JSON output
    for s in per_sample:
        s.pop("_pred_fwd", None)
        s.pop("_pred_struct", None)
        s.pop("_true_struct", None)

    # ── aggregate ─────────────────────────────────────────────────────────
    maes  = [s["mae"]       for s in per_sample]
    rs    = [s["pearson_r"] for s in per_sample]
    ic_rs = [s["ic_pearson"] for s in per_sample if not np.isnan(s["ic_pearson"])]

    aggregate = {
        "n":                n_samples,
        "mae_mean":         float(np.mean(maes)),
        "mae_median":       float(np.median(maes)),
        "pearson_r_mean":   float(np.mean(rs)),
        "pearson_r_median": float(np.median(rs)),
        "ic_pearson_mean":  float(np.mean(ic_rs)),
        "ic_pearson_median":float(np.median(ic_rs)),
        "n_ic_valid":       len(ic_rs),
    }

    print("\n=== DeepPBS benchmark results ===")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(ospj(args.out, "per_sample.json"), "w") as f:
        json.dump(per_sample, f, indent=2)
    with open(ospj(args.out, "metrics.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
