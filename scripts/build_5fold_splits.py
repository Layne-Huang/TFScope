#!/usr/bin/env python
"""Build 5-fold cross-validation splits matching DeepPBS exactly.

For each fold i in 0..4:
  train = parquet filenames of NPZ entries in DeepPBS's train{i}.txt
  val   = parquet filenames of NPZ entries in DeepPBS's valid{i}.txt
  test  = parquet filenames of NPZ entries in DeepPBS's id.txt (blind benchmark)

Also builds per-fold NN indices:
  donor pool = ONLY the train set of that fold (val and test retrieve from this pool)
  → prevents the model from peeking at val/test PWMs through retrieval
"""
import json, os, re
import numpy as np
import pandas as pd

FOLD_DIR  = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/deeppbsmar24/run/folds"
PARQUET   = "data/processed/tf_pwm_deeppbs_only.parquet"
EMB_PATH  = "data/processed/tf_dbd_embeddings.npz"
OUT_DIR   = "data/processed/splits/deeppbs_5fold"
NN_DIR    = "data/processed/nn_index_5fold"
K         = 5  # top-K neighbours

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(NN_DIR,  exist_ok=True)


def load_lines(fp):
    with open(fp) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    df = pd.read_parquet(PARQUET)
    embs_npz = np.load(EMB_PATH)
    all_fns  = list(df["filename"])

    # Build a lookup from (pdb, chain, source_id_canonical) → parquet filename
    # Parquet filename pattern: {pdb}_{chain}_{gene}.{source_id}.txt  (or .vN.txt)
    parquet_by_pdb_chain_src = {}   # (pdb, chain, src) → filename
    for fn in all_fns:
        # Strip .txt and optional .vN suffix
        stem = re.sub(r"\.v\d+\.txt$|\.txt$", "", fn)
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        pdb, chain = parts[0], parts[1]
        # Source_id is everything after the second "_<gene>." marker; we recover the
        # canonical (HOCOMOCO or JASPAR) form below
        # Easier: parse the source_id from the underlying NPZ name pattern we used
        # The NPZ original is {pdb}_{chain}_{rest}.npz where rest is e.g.
        # MA1951.1.jaspar (jaspar) or FOS_HUMAN.H11MO.0.A (hocomoco)
        # In our parquet, filename = {pdb}_{chain}_{gene}.{source_id}.{maybe vN}.txt
        # source_id (parquet col) is exactly the post-gene fragment
        pass

    # Use the source_id column directly for reliable matching
    df["pdb"]   = df["filename"].str.split("_", n=2).str[0]
    df["chain"] = df["filename"].str.split("_", n=2).str[1]
    # Build (pdb, chain, source_id) → parquet filename
    key_to_fn = {}
    for _, row in df.iterrows():
        key = (row["pdb"], row["chain"], row["source_id"])
        # If duplicate keys (rare — happens if multiple .vN versions exist),
        # keep the first (no suffix preferred)
        if key not in key_to_fn or not row["filename"].count(".v"):
            key_to_fn[key] = row["filename"]

    # NPZ entry → (pdb, chain, source_id) parse
    def parse_npz(entry):
        base = entry.replace(".npz", "")
        parts = base.split("_")
        if len(parts) < 3:
            return None
        pdb, chain = parts[0], parts[1]
        # JASPAR: MA####.V.jaspar
        m = re.search(r"(MA\d+)\.(\d+)\.jaspar$", base)
        if m:
            return (pdb, chain, f"{m.group(1)}.{m.group(2)}")
        # HOCOMOCO: {GENE}_{SPECIES}.H11MO.X.Y → source_id stored as e.g. "FOS_HUMAN.H11MO.0.A"
        m = re.search(r"_([A-Z0-9][A-Z0-9\-]*_(?:HUMAN|MOUSE)\.H11MO\.\d+\.[A-Z])$", base)
        if m:
            return (pdb, chain, m.group(1))
        return None

    def npz_to_parquet(entry):
        key = parse_npz(entry)
        return key_to_fn.get(key) if key else None

    # Load DeepPBS fold files and map to parquet filenames
    blind_npz = load_lines(os.path.join(FOLD_DIR, "id.txt"))
    blind_fns = [npz_to_parquet(e) for e in blind_npz]
    blind_fns = [f for f in blind_fns if f is not None]

    folds_train, folds_val = [], []
    for i in range(5):
        tr_npz = load_lines(os.path.join(FOLD_DIR, f"train{i}.txt"))
        va_npz = load_lines(os.path.join(FOLD_DIR, f"valid{i}.txt"))
        tr_fns = sorted(set(f for f in (npz_to_parquet(e) for e in tr_npz) if f is not None))
        va_fns = sorted(set(f for f in (npz_to_parquet(e) for e in va_npz) if f is not None))
        folds_train.append(tr_fns); folds_val.append(va_fns)
        print(f"  fold{i}: train={len(tr_fns)} (from {len(tr_npz)} NPZ), "
              f"val={len(va_fns)} (from {len(va_npz)} NPZ)")

    print(f"  blind test: {len(blind_fns)} (from {len(blind_npz)} NPZ)")

    # Write fold splits
    for i in range(5):
        split = {
            "train": folds_train[i],
            "val":   folds_val[i],
            "test":  sorted(blind_fns),
            "metadata": {
                "description": f"DeepPBS fold {i} — train{i}.txt + valid{i}.txt + id.txt",
                "n_train": len(folds_train[i]),
                "n_val":   len(folds_val[i]),
                "n_test":  len(blind_fns),
                "deeppbs_fold": i,
            },
        }
        path = os.path.join(OUT_DIR, f"fold{i}.json")
        with open(path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"  wrote {path}")

    # ── Build per-fold NN indices ────────────────────────────────────────────
    # For each fold, donor pool = train{i} only. Train/val/test queries all
    # retrieve from this pool.
    fn_to_idx = {fn: i for i, fn in enumerate(all_fns)}
    M = np.stack([embs_npz[fn] for fn in all_fns]).astype(np.float32)
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

    for fold_i in range(5):
        donor_fns = folds_train[fold_i]
        donor_idx = np.array([fn_to_idx[fn] for fn in donor_fns])
        index = {}
        for fn in all_fns:
            qi = fn_to_idx[fn]
            sims = (M_norm[qi:qi+1] @ M_norm[donor_idx].T).flatten()
            # Exclude self if it's a training query for this fold
            try:
                self_pos = donor_fns.index(fn)
                sims[self_pos] = -2.0
            except ValueError:
                pass
            top_k_idx = np.argsort(-sims)[:K]
            index[fn] = [
                {"nn_filename": donor_fns[int(di)], "cos_sim": float(sims[di])}
                for di in top_k_idx if sims[di] > -1.5
            ]
        path = os.path.join(NN_DIR, f"fold{fold_i}.json")
        with open(path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"  wrote {path}  (K={K}, donors={len(donor_fns)})")


if __name__ == "__main__":
    main()
