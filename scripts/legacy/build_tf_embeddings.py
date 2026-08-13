#!/usr/bin/env python
"""Compute DBD-masked mean-pooled ESM2 embeddings for every TF in the dataset.

For each row of `tf_pwm_deeppbs_only.parquet`:
  1. Tokenize the sequence (ESM2 alphabet)
  2. Forward through ESM2-650M (frozen, no LoRA)
  3. Take per-residue embeddings (layer 33), restricted to DBD positions
  4. Mean-pool over DBD → (1280,) embedding
  5. Save all to data/processed/tf_dbd_embeddings.npz, keyed by filename
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/tf_pwm_deeppbs_only.parquet")
    ap.add_argument("--out",  default="data/processed/tf_dbd_embeddings.npz")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--max-len", type=int, default=1024,
                    help="Cap sequence length before tokenizing (memory safety)")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import esm
    model_loader = getattr(esm.pretrained, args.esm_model)
    esm_model, alphabet = model_loader()
    esm_model = esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    repr_layer = esm_model.num_layers
    print(f"Loaded {args.esm_model} ({repr_layer} layers)")

    df = pd.read_parquet(args.data)
    print(f"Computing embeddings for {len(df)} TFs")

    embeddings = {}
    filenames  = df["filename"].tolist()
    sequences  = df["sequence"].tolist()
    dbd_starts = df["dbd_start"].tolist()
    dbd_ends   = df["dbd_end"].tolist()

    with torch.no_grad():
        for i in range(0, len(df), args.batch_size):
            batch_seqs = sequences[i:i+args.batch_size]
            batch_starts = dbd_starts[i:i+args.batch_size]
            batch_ends   = dbd_ends[i:i+args.batch_size]
            batch_fns    = filenames[i:i+args.batch_size]

            # Cap sequence length (long DBD misannotations would OOM ESM2)
            batch_seqs = [s[: args.max_len] for s in batch_seqs]
            batch_ends = [min(int(e), args.max_len) for e in batch_ends]

            # ESM batch converter expects [(label, sequence)]
            data = [(f, s) for f, s in zip(batch_fns, batch_seqs)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            out = esm_model(tokens, repr_layers=[repr_layer], return_contacts=False)
            reps = out["representations"][repr_layer]    # (B, L+2, 1280) with cls/eos

            for j in range(len(batch_seqs)):
                seq_len = len(batch_seqs[j])
                ds = max(0, int(batch_starts[j]))
                de = min(seq_len, int(batch_ends[j]))
                if de <= ds:
                    ds, de = 0, seq_len
                # +1 to skip <cls>; representation positions 1..seq_len correspond to residues
                dbd_reps = reps[j, 1+ds:1+de, :]
                emb = dbd_reps.mean(dim=0).cpu().numpy().astype(np.float32)
                embeddings[batch_fns[j]] = emb

            if (i // args.batch_size) % 10 == 0:
                print(f"  [{min(i+args.batch_size, len(df))}/{len(df)}] processed")

    print(f"Total embeddings: {len(embeddings)}")
    np.savez_compressed(args.out, **embeddings)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
