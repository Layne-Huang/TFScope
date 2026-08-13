#!/usr/bin/env python
"""Step 1 of the DPAC-augmented retrieval diagnostic.

Reads data/raw/dpac/DNA_train.csv (Protein, Motif), dedups proteins, and:
  - writes a parquet (DPAC_<i> filename, protein sequence, dbd_start/end = full)
    so scripts/build_tf_embeddings.py can ESM2-embed it in the SAME space as our TFs
  - builds a per-protein PSEUDO-PWM from its bound DNA sequence(s):
        modal-length motifs -> averaged one-hot + small pseudocount -> (4,L)
    saved to an npz keyed by DPAC_<i>.
"""
import os, numpy as np, pandas as pd
from collections import Counter

CSV = "data/raw/dpac/DNA_train.csv"
OUT_PARQUET = "data/processed/dpac_proteins.parquet"
OUT_PWM_NPZ = "data/processed/dpac_pseudo_pwms.npz"
B2I = {"A": 0, "C": 1, "G": 2, "T": 3}
MAXL = 30
ALPHA = 0.02          # pseudocount to avoid degenerate one-hot columns


def onehot(seq):
    L = min(len(seq), MAXL)
    m = np.zeros((4, L), np.float32)
    for j, b in enumerate(seq[:L]):
        i = B2I.get(b.upper())
        if i is not None: m[i, j] = 1.0
    return m


def pseudo_pwm(motifs):
    lens = [len(s) for s in motifs if all(b.upper() in B2I for b in s)]
    if not lens: return None
    modal = Counter(lens).most_common(1)[0][0]
    sel = [s for s in motifs if len(s) == modal][:50]
    mats = [onehot(s) for s in sel]
    pwm = np.mean(mats, axis=0)                       # (4, modal_clipped)
    pwm = pwm + ALPHA
    pwm = pwm / pwm.sum(0, keepdims=True)
    return pwm.astype(np.float32)


def main():
    df = pd.read_csv(CSV)
    groups = df.groupby("Protein")["Motif"].apply(list)
    print(f"{len(df)} pairs -> {len(groups)} unique proteins")

    rows, pwms = [], {}
    for i, (prot, motifs) in enumerate(groups.items()):
        pp = pseudo_pwm([str(m) for m in motifs])
        if pp is None: continue
        fn = f"DPAC_{i}"
        rows.append({"filename": fn, "sequence": str(prot),
                     "dbd_start": 0, "dbd_end": len(str(prot)),
                     "n_motifs": len(motifs), "pwm_len": pp.shape[1]})
        pwms[fn] = pp

    pdf = pd.DataFrame(rows)
    pdf.to_parquet(OUT_PARQUET)
    np.savez_compressed(OUT_PWM_NPZ, **pwms)
    print(f"wrote {OUT_PARQUET} ({len(pdf)} proteins) + {OUT_PWM_NPZ}")
    print(f"pseudo-PWM len: median={int(pdf.pwm_len.median())} "
          f"min={pdf.pwm_len.min()} max={pdf.pwm_len.max()}; "
          f"single-motif proteins={int((pdf.n_motifs==1).sum())} ({100*(pdf.n_motifs==1).mean():.0f}%)")


if __name__ == "__main__":
    main()
