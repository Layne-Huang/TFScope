#!/usr/bin/env python
"""Frozen ESM-2 DBD-embedding family-organization diagnostic.

Tests whether mean-pooled DBD residue embeddings from a *frozen* ESM-2
(esm2_t33_650M_UR50D) organize transcription factors by DNA-binding-domain
family. No training of the protein encoder is involved.

Pipeline
--------
1. Load esm2_t33_650M_UR50D (fair-esm), frozen, eval mode.
2. Per TF, run the (DBD-windowed) sequence through ESM-2 and take the
   final-layer (layer 33) residue representations.
3. Slice out residues [dbd_start, dbd_end) and mean-pool -> one vector per TF.
4. UMAP (n_neighbors=30, min_dist=0.25, metric="cosine") for visualization.
5. Nature-Methods-style scatter colored by family.
6. kNN family purity (k=10, cosine) in the original embedding space.
7. Logistic-regression linear probe, 5-fold stratified CV.

Long sequences (> ESM-2's ~1022-residue budget) are cropped to a flanking
window around the DBD before encoding, so the DBD keeps local structural
context while staying within the model's length limit. The DBD itself is
truncated only if it alone exceeds the budget (a handful of multi-domain
C2H2/Other proteins).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ESM_MAX_RESIDUES = 1022  # esm2_t33 positional budget minus BOS/EOS
DEFAULT_FLANK = 64       # residues of context added on each side of the DBD


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def _crop_window(seq: str, start: int, end: int, flank: int):
    """Return (cropped_seq, dbd_start_in_crop, dbd_end_in_crop).

    Adds `flank` residues on each side of the DBD, then clips the whole window
    to ESM_MAX_RESIDUES keeping the DBD as centered as possible.
    """
    L = len(seq)
    w_start = max(0, start - flank)
    w_end = min(L, end + flank)
    # If the window is too long, shrink flanks symmetrically around the DBD.
    if w_end - w_start > ESM_MAX_RESIDUES:
        dbd_len = end - start
        if dbd_len >= ESM_MAX_RESIDUES:
            # DBD alone overflows: keep its first ESM_MAX_RESIDUES residues.
            w_start = start
            w_end = start + ESM_MAX_RESIDUES
        else:
            spare = ESM_MAX_RESIDUES - dbd_len
            left = spare // 2
            w_start = max(0, start - left)
            w_end = min(L, w_start + ESM_MAX_RESIDUES)
            w_start = max(0, w_end - ESM_MAX_RESIDUES)
    return seq[w_start:w_end], start - w_start, min(end, w_end) - w_start


def compute_dbd_embeddings(df: pd.DataFrame, flank: int, batch_tokens: int,
                           device: str, use_fp16: bool) -> np.ndarray:
    import torch
    import esm

    print("[esm] loading esm2_t33_650M_UR50D ...", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device).eval()
    if use_fp16 and device.startswith("cuda"):
        model = model.half()
    for p in model.parameters():
        p.requires_grad_(False)

    # Pre-crop windows and remember DBD offsets within each crop.
    records = []  # (row_index, cropped_seq, dbd_s, dbd_e)
    for i, r in df.iterrows():
        cseq, ds, de = _crop_window(r["sequence"], int(r["dbd_start"]),
                                    int(r["dbd_end"]), flank)
        records.append((i, cseq, ds, de))

    # Length-sorted greedy batching by token budget (keeps padding small).
    order = sorted(range(len(records)), key=lambda k: len(records[k][1]))
    dim = model.embed_dim
    emb = np.zeros((len(records), dim), dtype=np.float32)

    batch, batch_max = [], 0
    done = 0

    def flush(batch):
        nonlocal done
        if not batch:
            return
        data = [(str(rec[0]), rec[1]) for rec in batch]
        _, _, toks = batch_converter(data)
        toks = toks.to(device)
        with torch.no_grad():
            out = model(toks, repr_layers=[33], return_contacts=False)
        rep = out["representations"][33].float()  # (B, Lpad+2, D)
        for bi, rec in enumerate(batch):
            row_idx, _, ds, de = rec
            # residue j -> token index j+1 (BOS prepended)
            vec = rep[bi, ds + 1:de + 1].mean(dim=0)
            emb[row_idx] = vec.cpu().numpy()
        done += len(batch)
        print(f"[esm] embedded {done}/{len(records)}", flush=True)

    for k in order:
        rec = records[k]
        seq_len = len(rec[1]) + 2
        new_max = max(batch_max, seq_len)
        if batch and new_max * (len(batch) + 1) > batch_tokens:
            flush(batch)
            batch, batch_max = [], 0
        batch.append(rec)
        batch_max = max(batch_max, seq_len)
    flush(batch)
    return emb


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def knn_family_purity(emb: np.ndarray, families: np.ndarray, k: int = 10):
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
    _, idx = nn.kneighbors(emb)
    idx = idx[:, 1:]  # drop self
    neigh_fam = families[idx]
    same = (neigh_fam == families[:, None]).mean(axis=1)

    per_family = {}
    for fam in np.unique(families):
        m = families == fam
        per_family[fam] = {"n": int(m.sum()),
                           "mean_purity": float(same[m].mean())}
    # Chance baseline: sum of squared family proportions.
    _, counts = np.unique(families, return_counts=True)
    p = counts / counts.sum()
    chance = float((p ** 2).sum())
    return {
        "k": k,
        "overall_mean_purity": float(same.mean()),
        "chance_baseline": chance,
        "per_family": per_family,
        "per_tf_purity": same,
    }


def linear_probe(emb: np.ndarray, families: np.ndarray, n_splits: int = 5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (accuracy_score, f1_score,
                                 balanced_accuracy_score, classification_report)

    # Drop families too small to stratify across folds.
    fams, counts = np.unique(families, return_counts=True)
    keep_fams = set(fams[counts >= n_splits])
    mask = np.array([f in keep_fams for f in families])
    X, y = emb[mask], families[mask]

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    return {
        "n_splits": n_splits,
        "n_samples": int(mask.sum()),
        "families_included": sorted(keep_fams),
        "families_excluded_too_small": sorted(set(fams) - keep_fams),
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted")),
        "per_family_report": report,
    }


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
# Paul Tol "muted" qualitative palette (colorblind-safe). Gray is deliberately
# excluded here and reserved for the generic "Other" catch-all class.
_PALETTE = [
    "#332288", "#CC6677", "#117733", "#DDCC77", "#88CCEE",
    "#AA4499", "#44AA99", "#882255", "#999933", "#661100",
]


def plot_umap(coords: np.ndarray, families: np.ndarray, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Order families by size (largest first) but push generic "Other" to gray.
    fam_order = [f for f, _ in sorted(
        ((f, int((families == f).sum())) for f in np.unique(families)),
        key=lambda t: -t[1])]
    color_map = {}
    ci = 0
    for f in fam_order:
        if f.lower() == "other":
            color_map[f] = "#CFCFCF"
        else:
            color_map[f] = _PALETTE[ci % len(_PALETTE)]
            ci += 1

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "figure.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    # Draw "Other" first (background), real families on top.
    for f in sorted(fam_order, key=lambda x: x.lower() != "other"):
        m = families == f
        ax.scatter(coords[m, 0], coords[m, 1], s=6, linewidths=0,
                   c=color_map[f], alpha=0.85 if f.lower() != "other" else 0.45,
                   label=f"{f} (n={int(m.sum())})", rasterized=True)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Frozen ESM-2 DBD embeddings", fontsize=9, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=False, fontsize=6.5, handletextpad=0.3,
                    labelspacing=0.35, borderaxespad=0.0)
    for h in leg.legend_handles:
        h.set_sizes([18])

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_png} (+ .pdf)", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="CSV: tf_id, sequence, dbd_start, dbd_end, family")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--flank", type=int, default=DEFAULT_FLANK)
    ap.add_argument("--batch-tokens", type=int, default=16000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--umap-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-emb", action="store_true",
                    help="reuse dbd_embeddings.npy in outdir if present")
    args = ap.parse_args()

    import torch
    device = args.device if (args.device.startswith("cuda")
                             and torch.cuda.is_available()) else "cpu"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    for col in ["tf_id", "sequence", "dbd_start", "dbd_end", "family"]:
        if col not in df.columns:
            raise ValueError(f"input missing required column: {col}")
    df["dbd_start"] = df["dbd_start"].astype(int)
    df["dbd_end"] = df["dbd_end"].astype(int)
    df = df.reset_index(drop=True)
    print(f"[data] {len(df)} TFs, {df['family'].nunique()} families "
          f"on device={device}", flush=True)

    emb_path = outdir / "dbd_embeddings.npy"
    if args.cache_emb and emb_path.exists():
        emb = np.load(emb_path)
        print(f"[esm] loaded cached embeddings {emb.shape}", flush=True)
    else:
        emb = compute_dbd_embeddings(df, args.flank, args.batch_tokens,
                                     device, args.fp16)
        np.save(emb_path, emb)
        print(f"[esm] saved embeddings -> {emb_path} {emb.shape}", flush=True)

    families = df["family"].to_numpy()

    # --- UMAP ---
    import umap
    reducer = umap.UMAP(n_neighbors=args.umap_neighbors,
                        min_dist=args.umap_min_dist, metric="cosine",
                        random_state=args.seed)
    coords = reducer.fit_transform(emb)

    umap_df = pd.DataFrame({
        "tf_id": df["tf_id"], "family": families,
        "umap1": coords[:, 0], "umap2": coords[:, 1],
    })
    umap_csv = outdir / "esm_dbd_umap.csv"
    umap_df.to_csv(umap_csv, index=False)
    print(f"[umap] saved {umap_csv}", flush=True)

    plot_umap(coords, families, outdir / "esm_dbd_umap.png")

    # --- kNN purity ---
    purity = knn_family_purity(emb, families, k=args.knn_k)
    per_tf = purity.pop("per_tf_purity")
    umap_df["knn_purity"] = per_tf
    umap_df.to_csv(umap_csv, index=False)  # augment with purity column

    # --- Linear probe ---
    probe = linear_probe(emb, families, n_splits=5)

    metrics = {
        "input": os.path.abspath(args.input),
        "n_tfs": int(len(df)),
        "n_families": int(df["family"].nunique()),
        "embedding_dim": int(emb.shape[1]),
        "esm_model": "esm2_t33_650M_UR50D",
        "esm_layer": 33,
        "pooling": "mean over DBD residues",
        "device": device,
        "knn_family_purity": purity,
        "linear_probe": probe,
    }
    metrics_path = outdir / "family_probe_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[metrics] saved {metrics_path}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"kNN(k={purity['k']}) family purity: "
          f"{purity['overall_mean_purity']:.3f} "
          f"(chance {purity['chance_baseline']:.3f})")
    print(f"Linear probe: acc={probe['accuracy']:.3f}  "
          f"balanced_acc={probe['balanced_accuracy']:.3f}  "
          f"macro_F1={probe['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
