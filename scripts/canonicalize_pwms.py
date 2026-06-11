#!/usr/bin/env python
"""Canonical PWM registration preprocessor (v16).

Re-registers every target PWM to ONE consistent convention so the model can
learn a fixed motif placement (fixes the offset problem: 70/130 v14 predictions
were shifted, costing ~0.20 mean r).

Deterministic, content-based transform (no external info → applies uniformly to
all sources, train and test, no leakage):

  1. LEFT-ANCHOR the informative core: trim leading/trailing positions whose
     information content < IC_THRESH bits, so the first informative column sits
     at position 0 and the motif is left-aligned.
  2. CANONICAL STRAND: choose the orientation (forward vs reverse-complement)
     whose IC-weighted position centroid is in the first half of the core
     (i.e. the informative mass leans left). Tie-break by a fixed base-priority
     rule on the highest-IC column. This gives every motif one canonical strand.

Outputs new parquets with the `pwm` column replaced by the canonical PWM and
`motif_length` updated. Filenames/splits are unchanged, so existing split JSONs
and NN indices still apply.
"""
import argparse, numpy as np, pandas as pd

IC_THRESH = 0.25   # bits; columns below this at the ends are flank → trimmed
RC_ROWS = [3, 2, 1, 0]


def pwm_ic(pwm):
    """Per-column IC in bits. pwm: (4, L)."""
    p = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)        # (L,)


def revcomp(pwm):
    return pwm[RC_ROWS][:, ::-1]


def trim_core(pwm, ic):
    """Trim leading/trailing low-IC flank; return core (>=1 col)."""
    informative = np.where(ic >= IC_THRESH)[0]
    if len(informative) == 0:
        return pwm                               # all flat → keep as-is
    lo, hi = informative[0], informative[-1] + 1
    return pwm[:, lo:hi]


def canonical_strand(core):
    """Pick orientation whose informative mass leans left.

    Score = IC-weighted centroid position (normalised). Lower centroid = mass
    on the left = canonical. Deterministic tie-break on highest-IC column base.
    """
    def centroid(c):
        ic = pwm_ic(c)
        w = ic / (ic.sum() + 1e-8)
        return (w * np.arange(len(ic))).sum() / max(len(ic) - 1, 1)

    fwd, rc = core, revcomp(core)
    cf, cr = centroid(fwd), centroid(rc)
    if abs(cf - cr) > 1e-6:
        return fwd if cf <= cr else rc
    # tie-break: orientation whose strongest column prefers the lower base index
    ic_f = pwm_ic(fwd); jf = int(ic_f.argmax())
    ic_r = pwm_ic(rc);  jr = int(ic_r.argmax())
    return fwd if fwd[:, jf].argmax() <= rc[:, jr].argmax() else rc


def canonicalize(pwm):
    ic = pwm_ic(pwm)
    core = trim_core(pwm, ic)
    return canonical_strand(core)


def process(in_path, out_path, max_len=20):
    df = pd.read_parquet(in_path)
    new_pwm, new_len = [], []
    n_trim = n_rc = 0
    for _, r in df.iterrows():
        if isinstance(r["pwm"], bytes):
            pwm = np.frombuffer(r["pwm"], dtype=np.float32).reshape(4, -1)
        else:
            new_pwm.append(r["pwm"]); new_len.append(r.get("motif_length", 0)); continue
        L0 = pwm.shape[1]
        can = canonicalize(pwm)
        if can.shape[1] != L0:
            n_trim += 1
        # detect rc (compare to forward-trimmed)
        if not np.array_equal(can, trim_core(pwm, pwm_ic(pwm))):
            n_rc += 1
        can = can[:, :max_len].astype(np.float32)
        # renormalise columns
        can = can / can.sum(0, keepdims=True).clip(1e-8)
        new_pwm.append(can.tobytes())
        new_len.append(can.shape[1])
    df = df.copy()
    df["pwm"] = new_pwm
    df["motif_length"] = new_len
    df.to_parquet(out_path)
    print(f"{in_path} -> {out_path}")
    print(f"  {len(df)} TFs | trimmed: {n_trim} | reverse-complemented: {n_rc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=[
        "data/processed/tf_pwm_deeppbs_only.parquet:data/processed/tf_pwm_deeppbs_only_canon.parquet",
        "data/processed/tf_pwm_aug_dbd.parquet:data/processed/tf_pwm_aug_dbd_canon.parquet",
    ])
    args = ap.parse_args()
    for pair in args.pairs:
        i, o = pair.split(":")
        process(i, o)
