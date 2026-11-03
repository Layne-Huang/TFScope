#!/usr/bin/env python
"""Build the two derived inputs the v24 ablation study needs.

Everything else in the study is a pure flag delta off `scripts/run_v24_contact_ddp.sh`
and needs no new files. Two ablations do need data:

A1 `structonly` — "train only on the data regime DeepPBS lives in"
    Drop every sequence-only (`seq_*`) row from train and val; keep the *identical*
    291-row structure test set. Answers the standing reviewer objection that TFScope
    beats DeepPBS only because it also trains on 4 641 sequence-only PWMs that a
    structure model cannot consume.
        train 4794 -> 936 rows (211 genes) | val 658 -> 144 rows (32 genes)
    Scale check against DeepPBS's own gene-disjoint universe
    (/data1/leihuang/DeepPBS/deeppbsmar24/run/iclr_folds, 477 structures / 217 TF ids):
    comparable, 211 vs 217 TF ids. It is a *regime* match, not a byte-identical copy
    of their fold files — their NPZ names are half JASPAR matrix ids, so an exact
    row-level mapping is lossy and would only shrink the set further.

A5 `shufcontact` — permuted-contact control
    Same contact losses, same number of supervised residues and columns, same weights
    — only *which* residue each PWM column is told to attend to is randomised. If the
    contact block's gain survives this, the block is a regulariser and not structural
    knowledge. That is a live hypothesis here: v23's contact indices are known to be
    frame-shifted for 192/273 structure rows, so a large share of the "structure"
    supervision already points at the wrong residue.

    Permutation design (per row, so the control is exactly matched):
      n1 = len(chain-1 sequence) = the token span the index is *supposed* to address.
      pi = a random permutation of [0, n1).
      residue r  ->  pi(r) if r < n1, else r unchanged.
    Leaving r >= n1 alone preserves the original clipping / partner-block spill
    pattern, so the shuffled model receives exactly as much surviving supervision as
    the real one. Both files are permuted with the SAME pi per row, so the 1-D
    recognition prior and the 2-D distillation target stay mutually consistent.

Out:
    data/processed/splits/ablation_v24/structonly.json
    data/contact_maps/recognition_residues_v23_shuffled.json
    data/contact_maps/contact_targets_v23_shuffled.json
    data/processed/splits/ablation_v24/_manifest.json

    scripts/ablation_v24/build_ablation_assets.py [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
RECOG_IN = "data/contact_maps/recognition_residues_v23.json"
CT_IN = "data/contact_maps/contact_targets_v23.json"

SPLIT_DIR = "data/processed/splits/ablation_v24"
RECOG_OUT = "data/contact_maps/recognition_residues_v23_shuffled.json"
CT_OUT = "data/contact_maps/contact_targets_v23_shuffled.json"

MAX_SEQ_LEN = 1024  # scripts/train.py --max-seq-len default, and what TFDataset tokenises to


def build_structonly_split(df: pd.DataFrame, split: dict) -> dict:
    """train/val restricted to `_set == 'struct'`; test untouched."""
    kind = dict(zip(df.filename, df["_set"]))
    gene = dict(zip(df.filename, df.gene_key))

    keep = {k: [f for f in split[k] if kind.get(f) == "struct"] for k in ("train", "val")}
    dropped = {k: [f for f in split[k] if kind.get(f) != "struct"] for k in ("train", "val")}
    test = list(split["test"])

    # The canonical split is already gene-disjoint; restricting a side can only help,
    # but assert it rather than trust it.
    tg = {gene[f] for f in test}
    for k in ("train", "val"):
        leak = {gene[f] for f in keep[k]} & tg
        assert not leak, f"{k}/test gene leak after restriction: {sorted(leak)[:5]}"

    out = {
        "train": keep["train"],
        "val": keep["val"],
        "test": test,
        # everything the ablation removes, kept explicit so the split is self-describing
        "excluded": list(split.get("excluded", [])) + dropped["train"] + dropped["val"],
        "metadata": {
            "derived_from": SPLIT,
            "rule": "train and val restricted to _set == 'struct' (structure-backed rows)",
            "n_train_rows": len(keep["train"]),
            "n_val_rows": len(keep["val"]),
            "n_test_rows": len(test),
            "n_train_genes": len({gene[f] for f in keep['train']}),
            "n_val_genes": len({gene[f] for f in keep['val']}),
            "n_test_genes": len(tg),
            "n_dropped_train_rows": len(dropped["train"]),
            "n_dropped_val_rows": len(dropped["val"]),
        },
    }
    return out


def _chain1_len(seq: str) -> int:
    return min(len(seq), MAX_SEQ_LEN)


def build_shuffled_contacts(df: pd.DataFrame, rng: np.random.Generator):
    recog = json.load(open(RECOG_IN))
    ct = json.load(open(CT_IN))
    seqlen = {f: _chain1_len(s) for f, s in zip(df.filename, df.sequence)}

    # one permutation per row, shared by both files
    perms: dict[str, np.ndarray] = {}

    def pi(fn: str, r: int) -> int:
        n1 = seqlen.get(fn)
        if not n1 or r >= n1 or r < 0:
            return r                      # out of chain-1: leave it (clipping preserved)
        p = perms.get(fn)
        if p is None:
            p = rng.permutation(n1)
            perms[fn] = p
        return int(p[r])

    recog_out = {fn: sorted({pi(fn, int(r)) for r in idx}) if idx else []
                 for fn, idx in recog.items()}

    ct_out = {}
    for fn, e in ct.items():
        cols = {}
        for col, rows in e["cols"].items():
            # collapse duplicates defensively (a permutation is injective on [0,n1),
            # and identity outside, so collisions are impossible -- assert instead)
            new = [[pi(fn, int(r)), float(w)] for r, w in rows]
            assert len({r for r, _ in new}) == len(new), f"{fn}:{col} permutation collided"
            cols[col] = new
        ct_out[fn] = {"L": e["L"], "cols": cols}
        for extra in ("partner_cols",):
            if extra in e:
                ct_out[fn][extra] = e[extra]   # partner space untouched (none in v23)

    # sanity: counts and weights are conserved, positions are not
    assert sum(len(v) for v in recog_out.values()) == sum(len(v) for v in recog.values())
    same = sum(1 for fn in recog if recog_out[fn] == recog[fn] and recog[fn])
    n_nonempty = sum(1 for v in recog.values() if v)
    print(f"  recognition prior: {n_nonempty} non-empty rows, {same} unchanged by chance "
          f"({same / max(n_nonempty, 1):.1%})")
    w_in = sum(w for e in ct.values() for rows in e["cols"].values() for _, w in rows)
    w_out = sum(w for e in ct_out.values() for rows in e["cols"].values() for _, w in rows)
    assert abs(w_in - w_out) < 1e-6, (w_in, w_out)
    print(f"  contact targets:   {len(ct_out)} rows, total weight {w_out:.2f} conserved")
    return recog_out, ct_out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="permutation seed for the A5 control")
    args = ap.parse_args()

    for p in (DATA, SPLIT, RECOG_IN, CT_IN):
        if not os.path.exists(p):
            sys.exit(f"ABORT: missing input {p}")

    df = pd.read_parquet(DATA)
    df["filename"] = df.filename.astype(str)
    split = json.load(open(SPLIT))

    print("=== A1 structonly split ===")
    so = build_structonly_split(df, split)
    m = so["metadata"]
    print(f"  train {m['n_train_rows']:>5} rows / {m['n_train_genes']} genes "
          f"(dropped {m['n_dropped_train_rows']} seq rows)")
    print(f"  val   {m['n_val_rows']:>5} rows / {m['n_val_genes']} genes "
          f"(dropped {m['n_dropped_val_rows']} seq rows)")
    print(f"  test  {m['n_test_rows']:>5} rows / {m['n_test_genes']} genes  (unchanged)")
    os.makedirs(SPLIT_DIR, exist_ok=True)
    json.dump(so, open(f"{SPLIT_DIR}/structonly.json", "w"), indent=1)
    print(f"  -> {SPLIT_DIR}/structonly.json")

    print(f"=== A5 permuted-contact control (seed {args.seed}) ===")
    rng = np.random.default_rng(args.seed)
    recog_out, ct_out = build_shuffled_contacts(df, rng)
    json.dump(recog_out, open(RECOG_OUT, "w"))
    json.dump(ct_out, open(CT_OUT, "w"))
    print(f"  -> {RECOG_OUT}\n  -> {CT_OUT}")

    json.dump({
        "structonly_split": f"{SPLIT_DIR}/structonly.json",
        "structonly_metadata": so["metadata"],
        "shuffled_recognition": RECOG_OUT,
        "shuffled_contact_targets": CT_OUT,
        "shuffle_seed": args.seed,
        "shuffle_rule": "per-row permutation of [0, len(chain1)); indices >= len(chain1) "
                        "left unchanged so clipping/partner-spill is identical to the real file",
    }, open(f"{SPLIT_DIR}/_manifest.json", "w"), indent=1)
    print(f"  -> {SPLIT_DIR}/_manifest.json")


if __name__ == "__main__":
    main()
