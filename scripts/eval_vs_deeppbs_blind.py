#!/usr/bin/env python
"""Fair comparison of TFScope vs DeepPBS on DeepPBS's 130-structure blind test.

Uses the SAME ground truth (Y_pwm from DeepPBS npz files) and SAME test set
(id.txt from DeepPBS folds/) as DeepPBS's own evaluation.

Computes:
  1. DeepPBS-style fixed-frame metrics (no alignment freedom):
       MAE = mean_pos(sum_bases |pred - true|)
       per-position Pearson r (4-element vectors), mean over positions
       IC Pearson r
     NOTE: fixed-frame is biased towards structure-based models (DeepPBS knows
     the crystal frame; TFScope's canonical frame ≠ crystal frame in general).

  2. Oracle-aligned metrics (±10 shift + RC) — fair to sequence-only models:
       panel-r (oracle-aligned Pearson r on trimmed core)
       aligned MAE
       aligned per-position r
       aligned IC-r

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_vs_deeppbs_blind.py \\
        --run-dir /data1/leihuang/project/TFScope/checkpoints/v19_deeppbs_rag/rag_seed42 \\
        --out results/v19_deeppbs/blind_vs_deeppbs.json
"""
import argparse, json, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")

from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from tfscope.data.dataset import TFDataset, collate_variable_length

DEEPPBS_DIR   = "/afs/csail.mit.edu/u/l/leihuang/project/DeepPBS"
DATA_DIR      = "/data1/leihuang/DeepPBS/deeppbsmar24/data/assembly2024"
FOLDS_DIR     = os.path.join(DEEPPBS_DIR, "run/folds")
DEFAULT_PARQUET  = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
DEFAULT_SPLIT    = "data/processed/splits/deeppbs_cluster_val/split.json"
DEFAULT_NN_INDEX = "data/processed/tf_nn_index_deeppbs_cluster_val.json"
BKG           = np.array([0.25, 0.25, 0.25, 0.25])
IC_THRESH     = 0.25
MIN_POS       = 4
MAX_SHIFT     = 10


# ── NPZ filename → parquet filename mapping ──────────────────────────────────

import pandas as pd

def build_npz_to_parquet_map(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["pdb_chain"] = df["filename"].str.extract(r"^([^_]+_[^_]+)_")
    df["source_id_up"] = df["source_id"].str.upper()

    def parse_npz(npz):
        name = npz.replace(".npz", "")
        if name.endswith(".jaspar"):
            name = name[:-len(".jaspar")]
            m = re.match(r"^([^_]+_[^_]+)_(MA\d+\.\d+)$", name, re.IGNORECASE)
            if m:
                return m.group(1), m.group(2).upper()
        m = re.match(r"^([^_]+_[^_]+)_([\w]+\.H\d+MO[\w.]+)$", name)
        if m:
            return m.group(1), m.group(2)
        return None, None

    mapping = {}
    for npz in os.listdir(DATA_DIR):
        if not npz.endswith(".npz"):
            continue
        pdb_chain, src_id = parse_npz(npz)
        if pdb_chain is None:
            continue
        sub = df[df["pdb_chain"] == pdb_chain]
        hits = sub[sub["source_id_up"] == src_id.upper()]
        if len(hits) == 0:
            hits = sub[sub["source_id"].str.contains(
                src_id.split(".")[0], case=False, na=False)]
        if len(hits) > 0:
            mapping[npz] = hits["filename"].iloc[0]
    return mapping


# ── Metric helpers ────────────────────────────────────────────────────────────

def ic_bits(pwm):
    p = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(axis=1)


def trimmed_core(pwm_L4, thresh=IC_THRESH, min_pos=MIN_POS):
    """Return columns of pwm_L4 (L×4) where IC >= thresh, if span >= min_pos."""
    p = np.clip(pwm_L4, 1e-8, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    ic = 2.0 + (p * np.log2(p)).sum(axis=1)
    keep = ic >= thresh
    if keep.sum() < min_pos:
        return None
    return p[keep]                             # (k×4)


def deeppbs_metrics(pred_L4, true_L4):
    """DeepPBS protocol: (L,4) arrays."""
    mae = float(np.mean(np.sum(np.abs(pred_L4 - true_L4), axis=1)))
    rs  = [pearsonr(true_L4[i], pred_L4[i])[0] for i in range(len(true_L4))]
    r   = float(np.nanmean(rs))
    ic_t = ic_bits(true_L4); ic_p = ic_bits(pred_L4)
    with np.errstate(invalid="ignore"):
        ic_r = float(pearsonr(ic_t, ic_p)[0]) if len(ic_t) > 1 else float("nan")
    return mae, r, ic_r


def oracle_metrics(pred_4L, true_4L, max_shift=MAX_SHIFT):
    """Oracle-aligned panel-r, aligned MAE, aligned per-pos r, aligned IC-r."""
    core = trimmed_core(true_4L.T)             # need (L,4) for helper
    if core is None:
        return None
    aligned, _, _, _ = align_pwm(
        pred_4L, true_4L, max_shift=max_shift, consider_revcomp=True
    )
    # Pearson r on oracle-aligned overlap
    a = aligned.ravel(); b = true_4L.ravel()
    n = min(len(a), len(b)); a = a[:n]; b = b[:n]
    denom = np.sqrt(((a - a.mean())**2).sum() * ((b - b.mean())**2).sum())
    panel_r = float(np.dot(a - a.mean(), b - b.mean()) / denom) if denom > 0 else 0.0
    # aligned per-pos r and MAE on overlap (take common span)
    al_L4 = aligned.T  # (L,4) aligned
    tr_L4 = true_4L.T
    L = min(al_L4.shape[0], tr_L4.shape[0])
    al_L4 = al_L4[:L]; tr_L4 = tr_L4[:L]
    ali_mae, ali_r, ali_ic_r = deeppbs_metrics(al_L4, tr_L4)
    return panel_r, ali_mae, ali_r, ali_ic_r


# ── Load model ───────────────────────────────────────────────────────────────

def load_model(run_dir, ckpt_name="ckpt_best.pt", device="cuda"):
    cfg = TFScopeConfig()
    with open(os.path.join(run_dir, "config.json")) as fh:
        saved = json.load(fh)
    for k, v in saved.items():
        if hasattr(cfg, k):
            try:    setattr(cfg, k, type(getattr(cfg, k))(v))
            except: setattr(cfg, k, v)
    model = TFScopeModel(cfg).to(device).eval()
    ckpt = torch.load(os.path.join(run_dir, ckpt_name),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded {ckpt_name} (epoch {epoch})")
    return model, cfg


def make_dataloader(cfg, parquet, split, split_name="test", batch=16):
    ds = TFDataset(cfg, parquet, split, split=split_name, max_seq_len=1024)
    return DataLoader(ds, batch_size=batch, shuffle=False,
                      num_workers=2, collate_fn=collate_variable_length)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="ckpt_best.pt")
    p.add_argument("--out", default="results/v19_deeppbs/blind_vs_deeppbs.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--data",     default=None, help="Override parquet path")
    p.add_argument("--split",    default=None, help="Override split JSON path")
    p.add_argument("--eval-split", default="test", choices=("train","val","test"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    parquet = args.data  if args.data  else DEFAULT_PARQUET
    split   = args.split if args.split else DEFAULT_SPLIT

    df = pd.read_parquet(parquet)
    npz_to_parquet = build_npz_to_parquet_map(df)

    # Load blind test list
    with open(os.path.join(FOLDS_DIR, "id.txt")) as fh:
        blind_npz = [l.strip() for l in fh if l.strip()]
    print(f"Blind test structures: {len(blind_npz)}")

    # Map to parquet filenames
    blind_pairs = []   # [(npz_name, parquet_filename)]
    missing = []
    for npz in blind_npz:
        pf = npz_to_parquet.get(npz)
        if pf:
            blind_pairs.append((npz, pf))
        else:
            missing.append(npz)
    print(f"Mapped: {len(blind_pairs)}, Missing: {len(missing)}")
    if missing:
        print("  Missing:", missing[:5])

    # Load TFScope model
    model, cfg = load_model(args.run_dir, args.checkpoint, device)

    # Run inference over the full split test set to get all predictions cached
    ds = TFDataset(cfg, parquet, split, split=args.eval_split, max_seq_len=1024)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_variable_length)
    preds_cache = {}   # parquet_filename → (pred_4L, gate_pred_4L)
    row_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = batch["family_id"].shape[0]
            metadata = ds.df.iloc[row_offset : row_offset + batch_size]
            row_offset += batch_size
            batch = {k: v.to(device,
                             dtype=torch.float32 if v.is_floating_point()
                             else torch.long)
                     for k, v in batch.items()}
            gate_logits, pwm_logits, _ = model(
                batch["sequence_tokens"], batch["dbd_mask"], batch["family_id"],
                retrieved_pwms=batch.get("retrieved_pwms"),
                retrieved_masks=batch.get("retrieved_masks"),
                retrieved_sims=batch.get("retrieved_sims"),
                recog_prior=batch.get("recog_prior"),
            )
            pwm_prob  = F.softmax(pwm_logits,  dim=1).cpu().numpy()  # (B,4,L)
            gate_prob = torch.sigmoid(gate_logits).cpu().numpy()
            for i, meta in enumerate(metadata.itertuples(index=False)):
                preds_cache[str(meta.filename)] = (pwm_prob[i], gate_prob[i])

    print(f"Cached predictions for {len(preds_cache)} test records")

    # Evaluate each blind structure
    fixed_rows   = []   # (mae, r, ic_r) fixed-frame
    aligned_rows = []   # (panel_r, ali_mae, ali_r, ali_ic_r) oracle-aligned
    per_record   = []

    for npz_name, pf in blind_pairs:
        # Ground truth from npz (DeepPBS's exact Y_pwm)
        npz_path = os.path.join(DATA_DIR, npz_name)
        if not os.path.exists(npz_path):
            print(f"  Missing npz: {npz_path}")
            continue
        d = np.load(npz_path, allow_pickle=True)
        Y_pwm    = d["Y_pwm"].astype(np.float64)     # (2, L, 4)
        pwm_mask = d["pwm_mask"]                       # (2, L) bool

        # Use the strand with more informative positions
        n0 = int(pwm_mask[0].sum()); n1 = int(pwm_mask[1].sum())
        strand = 0 if n0 >= n1 else 1
        true_L4 = Y_pwm[strand][pwm_mask[strand]]     # (L, 4)
        if true_L4.shape[0] < MIN_POS:
            continue
        # Normalize GT
        true_L4 = np.clip(true_L4, 1e-8, 1.0)
        true_L4 = true_L4 / true_L4.sum(axis=1, keepdims=True)

        # TFScope prediction
        if pf not in preds_cache:
            print(f"  No prediction cached for {pf}")
            continue
        pred_4L, _ = preds_cache[pf]                  # (4, L_full)
        L_true = true_L4.shape[0]
        L_pred = pred_4L.shape[1]

        # --- Fixed-frame (canonical left-anchor, no alignment) ---
        pred_fixed = np.full((L_true, 4), 0.25, dtype=np.float64)
        L = min(L_pred, L_true)
        pred_fixed[:L] = pred_4L[:, :L].T
        pred_fixed = np.clip(pred_fixed, 1e-8, 1.0)
        pred_fixed = pred_fixed / pred_fixed.sum(axis=1, keepdims=True)
        f_mae, f_r, f_ic_r = deeppbs_metrics(pred_fixed, true_L4)
        fixed_rows.append((f_mae, f_r, f_ic_r))

        # --- Oracle-aligned ---
        true_4L = true_L4.T                            # (4, L)
        result = oracle_metrics(pred_4L, true_4L)
        if result is not None:
            panel_r, ali_mae, ali_r, ali_ic_r = result
            aligned_rows.append((panel_r, ali_mae, ali_r, ali_ic_r))
        else:
            panel_r = ali_mae = ali_r = ali_ic_r = float("nan")

        per_record.append({
            "npz": npz_name, "parquet": pf,
            "fixed_mae": f_mae, "fixed_r": f_r, "fixed_ic_r": f_ic_r,
            "panel_r": panel_r, "aligned_mae": ali_mae,
            "aligned_r": ali_r, "aligned_ic_r": ali_ic_r,
        })

    def nanm(arr): return float(np.nanmean(arr))
    def nanmed(arr): return float(np.nanmedian(arr))

    fa = np.array(fixed_rows)
    aa = np.array(aligned_rows)

    print(f"\n=== TFScope on DeepPBS blind test (n={len(fixed_rows)}) ===")
    print(f"  [Fixed-frame — DeepPBS protocol, NO alignment freedom]")
    print(f"    MAE   mean={nanm(fa[:,0]):.4f}  median={nanmed(fa[:,0]):.4f}")
    print(f"    r     mean={nanm(fa[:,1]):.4f}  median={nanmed(fa[:,1]):.4f}")
    print(f"    IC-r  mean={nanm(fa[:,2]):.4f}")
    print(f"  [Oracle-aligned — fair for sequence-only models, ±{MAX_SHIFT} + RC]")
    print(f"    panel-r  mean={nanm(aa[:,0]):.4f}  median={nanmed(aa[:,0]):.4f}")
    print(f"    ali-MAE  mean={nanm(aa[:,1]):.4f}")
    print(f"    ali-r    mean={nanm(aa[:,2]):.4f}")
    print(f"    ali-IC-r mean={nanm(aa[:,3]):.4f}")

    print(f"\n  [DeepPBS 5-model ensemble, published (same 130 structures)]")
    print(f"    Pearson r ≈ 0.702  (per-position, fixed crystal frame)")
    print(f"    See DeepPBS paper Table 1 / eval_deeppbs_per_fold.py comment")

    result = {
        "checkpoint": os.path.join(args.run_dir, args.checkpoint),
        "test_set": "DeepPBS blind test (id.txt)",
        "n_structures": len(fixed_rows),
        "ground_truth": "Y_pwm from DeepPBS npz (crystal-structure-aligned JASPAR/HOCOMOCO)",
        "fixed_frame": {
            "note": "No alignment freedom; TFScope canonical frame ≠ crystal frame → lower bound",
            "mae_mean":   nanm(fa[:,0]), "mae_median":   nanmed(fa[:,0]),
            "r_mean":     nanm(fa[:,1]), "r_median":     nanmed(fa[:,1]),
            "ic_r_mean":  nanm(fa[:,2]),
        },
        "oracle_aligned": {
            "note": f"±{MAX_SHIFT} shift + RC alignment — fair metric for sequence-only models",
            "panel_r_mean":   nanm(aa[:,0]), "panel_r_median": nanmed(aa[:,0]),
            "ali_mae_mean":   nanm(aa[:,1]),
            "ali_r_mean":     nanm(aa[:,2]),
            "ali_ic_r_mean":  nanm(aa[:,3]),
        },
        "deeppbs_published": {
            "note": "5-model ensemble, fixed crystal frame, same 130 structures",
            "r_mean": 0.702,
        },
        "per_record": per_record,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
