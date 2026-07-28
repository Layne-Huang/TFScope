"""Training-free baselines B0 (family-average) and B1 (nearest training PWM).

These are the plan's floor / retrieval controls (§3): no learned parameters, so
they anchor how much of any model's score is a family prior or memorisation.
They are scored with the *identical* coverage-aware, gene-balanced protocol used
for every other variant (plan §2 rule 3), reusing the scoring primitives in
``scripts/eval_full_metrics.py`` so the numbers are directly comparable.

    python -m iclr.baselines --variant B0 \
        --train-data data/processed/tf_pwm_training_v23.parquet \
        --split data/processed/splits/train_v22/split.json \
        --test-data data/processed/tf_pwm_training_v23.parquet \
        --test-split data/processed/splits/train_v22/split.json \
        --out checkpoints/iclr_phase1/B0
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
from tfscope.models.alignment import align_pwm, revcomp_pwm_np  # noqa: E402


# ── PWM decode (matches tfscope.data.dataset) ──────────────────────────────────
def _decode_pwm(buf) -> np.ndarray:
    if isinstance(buf, (bytes, bytearray)):
        return np.frombuffer(buf, dtype=np.float32).reshape(4, -1).copy()
    arr = np.asarray(buf, dtype=np.float32)
    return arr.reshape(4, -1).copy()


def _ic_bits(p):
    p = np.clip(p, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)


def _trimmed_core(pwm: np.ndarray, thresh: float = 0.25) -> np.ndarray | None:
    """Trim to the informative core (same rule as eval_full_metrics.trimmed_core)."""
    t = np.clip(pwm, 1e-8, 1.0)
    t = t / t.sum(0, keepdims=True)
    ic = _ic_bits(t)
    inf = np.where(ic >= thresh)[0]
    if len(inf) == 0:
        return None
    c = t[:, inf[0]:inf[-1] + 1]
    return c / c.sum(0, keepdims=True)


def _r_cov(pred: np.ndarray, core: np.ndarray, max_shift=10, min_overlap=2) -> float:
    """Coverage-scaled overlap Pearson r (== eval_full_metrics r_cov)."""
    aligned, shift, orient, score = align_pwm(
        pred, core, max_shift=max_shift, consider_revcomp=True, min_overlap=min_overlap)
    if score <= -1.5:
        return 0.0
    o = revcomp_pwm_np(pred) if orient == "rc" else pred
    cols = [i + shift for i in range(o.shape[1]) if 0 <= i + shift < core.shape[1]]
    cols = np.array(sorted(cols), dtype=int)
    if len(cols) < 2:
        return 0.0
    t = core[:, cols]
    p = np.clip(aligned[:, cols], 1e-8, 1.0)
    p = p / p.sum(0, keepdims=True)
    from scipy.stats import pearsonr
    rs = [0.0 if (t[:, j].std() == 0 or p[:, j].std() == 0) else pearsonr(t[:, j], p[:, j])[0]
          for j in range(t.shape[1])]
    r_overlap = float(np.nanmean(rs))
    coverage = len(cols) / core.shape[1]
    return r_overlap * coverage


def _load(parquet: str, split: str, subset: str):
    df = pd.read_parquet(parquet)
    ids = set(json.load(open(split))[subset])
    sub = df[df["filename"].astype(str).isin(ids)].copy()
    sub["pwm_arr"] = sub["pwm"].map(_decode_pwm)
    return sub


def _family_average(train_df: pd.DataFrame) -> dict:
    """Left-anchored mean PWM per family_id (crude family prior floor)."""
    fam = {}
    for fid, g in train_df.groupby("family_id"):
        pwms = list(g["pwm_arr"])
        W = int(np.median([p.shape[1] for p in pwms]))
        W = max(W, 1)
        acc = np.full((4, W), 0.25, dtype=np.float64)
        n = np.zeros(W)
        for p in pwms:
            L = min(p.shape[1], W)
            acc[:, :L] += p[:, :L] / p[:, :L].sum(0, keepdims=True).clip(1e-8)
            n[:L] += 1
        avg = np.where(n > 0, acc / np.maximum(n, 1), 0.25)
        avg = avg / avg.sum(0, keepdims=True).clip(1e-8)
        fam[fid] = avg.astype(np.float32)
    return fam


def _nearest_pwm(test_seq: str, train_df: pd.DataFrame) -> np.ndarray:
    """Nearest training TF by DBD-sequence identity (difflib ratio proxy)."""
    from difflib import SequenceMatcher
    best_r, best_pwm = -1.0, None
    sm = SequenceMatcher()
    sm.set_seq2(test_seq)
    for seq, pwm in zip(train_df["sequence"], train_df["pwm_arr"]):
        sm.set_seq1(seq)
        r = sm.quick_ratio()
        if r > best_r:
            best_r, best_pwm = r, pwm
    return best_pwm if best_pwm is not None else np.full((4, 6), 0.25, np.float32)


def run(variant: str, train_parquet, split, test_parquet, test_split, out_dir,
        ic_thresh=0.25):
    train_df = _load(train_parquet, split, "train")
    test_df = _load(test_parquet, test_split, "test")
    print(f"[{variant}] train rows={len(train_df)}  test rows={len(test_df)}")

    fam_avg = _family_average(train_df) if variant == "B0" else None

    per_gene: dict[str, list] = {}
    per_sample = []
    for _, row in test_df.iterrows():
        core = _trimmed_core(row["pwm_arr"], ic_thresh)
        if core is None:
            continue
        if variant == "B0":
            pred = fam_avg.get(row["family_id"], np.full((4, core.shape[1]), 0.25, np.float32))
        elif variant == "B1":
            pred = _nearest_pwm(row["sequence"], train_df)
        else:
            raise ValueError(f"unknown training-free variant {variant}")
        rc = _r_cov(pred, core)
        gene = str(row.get("gene_symbol", row["filename"]))
        per_gene.setdefault(gene, []).append(rc)
        per_sample.append({"filename": str(row["filename"]), "gene": gene, "r_cov": rc,
                           "n_chains": int(row.get("n_chains", 1))})

    gene_means = {g: float(np.nanmean(v)) for g, v in per_gene.items()}
    gene_covR = float(np.nanmean(list(gene_means.values()))) if gene_means else float("nan")
    row_covR = float(np.nanmean([s["r_cov"] for s in per_sample])) if per_sample else float("nan")
    mono = [s["r_cov"] for s in per_sample if s["n_chains"] <= 1]
    multi = [s["r_cov"] for s in per_sample if s["n_chains"] > 1]

    result = {
        "variant": variant,
        "gene_covR": gene_covR,
        "row_covR": row_covR,
        "monomer_row_covR": float(np.nanmean(mono)) if mono else None,
        "multimer_row_covR": float(np.nanmean(multi)) if multi else None,
        "n_test_scored": len(per_sample),
        "by_gene_r_cov": gene_means,
        "per_sample": per_sample,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{variant}_result.json")
    json.dump(result, open(path, "w"), indent=2)
    print(f"[{variant}] gene_covR={gene_covR:.4f}  row_covR={row_covR:.4f}  "
          f"(n={len(per_sample)})  -> {path}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=["B0", "B1"])
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--test-split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ic-thresh", type=float, default=0.25)
    args = ap.parse_args()
    run(args.variant, args.train_data, args.split, args.test_data, args.test_split,
        args.out, args.ic_thresh)


if __name__ == "__main__":
    main()
