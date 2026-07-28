"""Shared Phase-I evaluator (plan §2 rules 3, 5, 8).

Scores a trained checkpoint (or the frozen v24 reference, B8) on the immutable
291-row structure test set with the identical coverage-aware, gene-balanced
protocol used for the training-free baselines, so every B0–B8 number is directly
comparable. Sequence-only inference (plan §7.9): no contacts/structure are fed.

    python -m iclr.evaluate --ckpt checkpoints/iclr_phase1/B5/seed42/ckpt_best.pt \
        --test-data data/processed/tf_pwm_training_v23.parquet \
        --test-split data/processed/splits/train_v22/split.json \
        --tag B5_seed42 --out checkpoints/iclr_phase1/B5

The model config is read from ``config.json`` next to the checkpoint (written by
scripts/train.py). Monomer vs multimer is read from the parquet's ``n_chains``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from iclr.baselines import _decode_pwm, _trimmed_core, _r_cov  # noqa: E402


def _load_model(ckpt: str):
    import torch
    from tfscope.config import TFScopeConfig
    from tfscope.models.tfscope import TFScopeModel

    cfg = TFScopeConfig()
    cfg_path = os.path.join(os.path.dirname(ckpt), "config.json")
    if os.path.exists(cfg_path):
        for k, v in json.load(open(cfg_path)).items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception:
                    setattr(cfg, k, v)
    else:
        print(f"[warn] no config.json next to {ckpt}; using default config", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TFScopeModel(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    return model, cfg, device


def _predict(model, cfg, device, test_data, test_split):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tfscope.data.dataset import TFDataset, collate_variable_length

    ds = TFDataset(cfg, test_data, test_split, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                    collate_fn=collate_variable_length)
    P, M = [], []
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            # sequence-only inference: no recog_prior / contact_override passed.
            _, pw, _ = model(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                             retrieved_pwms=b.get("retrieved_pwms"),
                             retrieved_masks=b.get("retrieved_masks"),
                             retrieved_sims=b.get("retrieved_sims"))
            P.append(F.softmax(pw, 1).cpu().numpy())
            M.append(b["pwm_mask"].cpu().numpy())
    P = np.concatenate(P); M = np.concatenate(M)
    return {fn: P[i][:, M[i].astype(bool)] for i, fn in enumerate(ds.filenames)}


def score(preds: dict, test_data, test_split, tag, out_dir, ic_thresh=0.25):
    df = pd.read_parquet(test_data)
    test_ids = set(json.load(open(test_split))["test"])
    df = df[df["filename"].astype(str).isin(test_ids)].copy()

    per_gene: dict[str, list] = {}
    per_sample = []
    for _, row in df.iterrows():
        fn = str(row["filename"])
        if fn not in preds:
            continue
        core = _trimmed_core(_decode_pwm(row["pwm"]), ic_thresh)
        if core is None:
            continue
        rc = _r_cov(preds[fn], core)
        gene = str(row.get("gene_symbol", fn))
        per_gene.setdefault(gene, []).append(rc)
        per_sample.append({"filename": fn, "gene": gene, "r_cov": rc,
                           "family": str(row.get("family_name", "NA")),
                           "n_chains": int(row.get("n_chains", 1))})

    gene_means = {g: float(np.nanmean(v)) for g, v in per_gene.items()}
    gene_covR = float(np.nanmean(list(gene_means.values()))) if gene_means else float("nan")
    row_covR = float(np.nanmean([s["r_cov"] for s in per_sample])) if per_sample else float("nan")
    mono = [s["r_cov"] for s in per_sample if s["n_chains"] <= 1]
    multi = [s["r_cov"] for s in per_sample if s["n_chains"] > 1]
    by_family: dict[str, list] = {}
    for s in per_sample:
        by_family.setdefault(s["family"], []).append(s["r_cov"])

    result = {
        "tag": tag,
        "gene_covR": gene_covR,
        "row_covR": row_covR,
        "monomer_gene_covR": float(np.nanmean(mono)) if mono else None,
        "multimer_gene_covR": float(np.nanmean(multi)) if multi else None,
        "n_test_scored": len(per_sample),
        "by_family_r_cov": {k: float(np.nanmean(v)) for k, v in by_family.items()},
        "by_gene_r_cov": gene_means,
        "per_sample": per_sample,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}_eval.json")
    json.dump(result, open(path, "w"), indent=2)
    print(f"[{tag}] gene_covR={gene_covR:.4f}  row_covR={row_covR:.4f}  "
          f"mono={result['monomer_gene_covR']}  multi={result['multimer_gene_covR']}  "
          f"(n={len(per_sample)}) -> {path}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-data", default="data/processed/tf_pwm_training_v23.parquet")
    ap.add_argument("--test-split", default="data/processed/splits/train_v22/split.json")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ic-thresh", type=float, default=0.25)
    args = ap.parse_args()

    model, cfg, device = _load_model(args.ckpt)
    preds = _predict(model, cfg, device, args.test_data, args.test_split)
    score(preds, args.test_data, args.test_split, args.tag, args.out, args.ic_thresh)


if __name__ == "__main__":
    main()
