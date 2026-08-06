#!/usr/bin/env python
"""Build a FLANK-AUGMENTED training table from v23: re-attach +/-F flanking
residues around each DBD, recovered from the original full-length sequences
(tf_pwm.parquet). The DBD is marked via dbd_start:dbd_end WITHIN the flanked
sequence (dbd_start now > 0), so the model sees the DBD in its native protein
context. Rows whose full sequence is unavailable stay DBD-only (dbd_start=0).

Crystal chains carry tags/point-mutations, so we locate the crop by exact
substring first, else by local alignment (PairwiseAligner). Partner_seqs / PWM /
family_id / all other columns are carried through unchanged.

  python scripts/build_flank_dataset.py --flank 20 \
      --out data/processed/tf_pwm_training_v25flank.parquet
"""
import argparse, numpy as np, pandas as pd
from Bio.Align import PairwiseAligner

V23 = "data/processed/tf_pwm_training_v23.parquet"
ORIG = "data/processed/tf_pwm.parquet"

aln = PairwiseAligner()
aln.mode = "local"; aln.match_score = 2; aln.mismatch_score = -1
aln.open_gap_score = -5; aln.extend_gap_score = -0.5


def locate(crop, full):
    """Return (start,end) of the DBD crop within full, or None."""
    i = full.find(crop)
    if i >= 0:
        return i, i + len(crop)
    if len(crop) < 6 or len(full) < 6:
        return None
    try:
        a = aln.align(full, crop)[0]
    except Exception:
        return None
    # aligned target (full) coords of the crop's span
    tblocks = a.aligned[0]
    if len(tblocks) == 0:
        return None
    s = int(tblocks[0][0]); e = int(tblocks[-1][1])
    # require the alignment to cover most of the crop (guard against junk)
    if (e - s) < 0.7 * len(crop):
        return None
    return s, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flank", type=int, default=20, help="residues added on EACH side of the DBD")
    ap.add_argument("--max-len", type=int, default=1000, help="clip total sequence to ESM budget")
    ap.add_argument("--out", default="data/processed/tf_pwm_training_v25flank.parquet")
    a = ap.parse_args()

    v = pd.read_parquet(V23).reset_index(drop=True)
    orig = pd.read_parquet(ORIG); orig["G"] = orig.gene_symbol.astype(str).str.upper()
    fullmap = {}
    for r in orig.itertuples():
        fullmap.setdefault(r.G, str(r.sequence))     # gene -> full-length seq

    new_seq, new_ds, new_de, src = [], [], [], []
    n_flanked = n_dbd = 0
    for r in v.itertuples():
        g = str(r.gene_symbol).upper(); crop = str(r.sequence); full = fullmap.get(g)
        loc = locate(crop, full) if full else None
        if loc is None:
            new_seq.append(crop); new_ds.append(0); new_de.append(len(crop)); src.append("dbd_only"); n_dbd += 1
            continue
        s, e = loc
        ws = max(0, s - a.flank); we = min(len(full), e + a.flank)
        seq = full[ws:we]
        ds = s - ws; de = ds + (e - s)
        if len(seq) > a.max_len:                       # clip flanks symmetrically to budget
            over = len(seq) - a.max_len; l = over // 2
            seq = seq[l:l + a.max_len]; ds = max(0, ds - l); de = min(len(seq), de - l)
        new_seq.append(seq); new_ds.append(int(ds)); new_de.append(int(de)); src.append("flanked"); n_flanked += 1

    v["sequence"] = new_seq; v["dbd_start"] = new_ds; v["dbd_end"] = new_de
    v["seq_length"] = [len(s) for s in new_seq]; v["flank_source"] = src
    v.to_parquet(a.out, index=False)
    nf = sum(s == "flanked" for s in src)
    print(f"flank={a.flank}  flanked={nf}/{len(v)} ({100*nf/len(v):.0f}%)  dbd_only={n_dbd}")
    print(f"median seq_length: {int(np.median(v.seq_length))}  (v23 was 83)")
    print(f"median DBD span:   {int(np.median(v.dbd_end - v.dbd_start))}")
    print("wrote", a.out)
    # sanity: MyoD1
    m = v[v.gene_symbol.astype(str).str.upper() == "MYOD1"]
    if len(m):
        r = m.iloc[0]; print(f"MYOD1: len={len(r.sequence)} dbd=[{r.dbd_start}:{r.dbd_end}] src={r.flank_source}")
        print("       seq:", r.sequence)


if __name__ == "__main__":
    main()
