#!/usr/bin/env python
"""Mine within-family 'natural mutation pairs' from the TRAIN split: pairs of
same-family proteins with 60-99% sequence identity, each with its own measured
PWM. For each pair we store the oracle-aligned ΔPWM magnitude so the pairwise
fine-tune can up-weight specificity-SWITCHING pairs (large ΔPWM) while still
keeping neutral pairs (teach invariance). Per-protein neighbour cap keeps big
families (C2H2) from dominating.

Out: data/processed/mut_pairs_v23.json  (list of {a,b,ident,dpwm})
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from tfscope.models.alignment import align_pwm

DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
OUT = "data/processed/mut_pairs_v23.json"
K_PER_PROT = 8          # neighbours kept per protein (top-ΔPWM + random neutral)
ID_LO, ID_HI = 0.55, 0.999


def decode(raw):
    a = raw.astype(np.float32) if isinstance(raw, np.ndarray) else np.frombuffer(raw, dtype=np.float32)
    return a.reshape(4, -1)


def core(pwm, ic=0.2):
    ic_col = 2.0 + np.sum(np.clip(pwm, 1e-6, 1) * np.log2(np.clip(pwm, 1e-6, 1)), 0)
    keep = np.where(ic_col > ic)[0]
    return pwm[:, keep.min():keep.max() + 1] if len(keep) else pwm


def ident(a, b):
    L = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:L], b[:L])) / L if L else 0.0


def dpwm(ca, cb):
    aligned, _, _, _ = align_pwm(cb, ca, max_shift=10, consider_revcomp=True, min_overlap=3)
    return float(np.mean(np.abs(ca - aligned)))


def main():
    df = pd.read_parquet(DATA)
    tr = df[df.filename.isin(set(json.load(open(SPLIT))["train"]))].reset_index(drop=True)
    tr["seq"] = tr.sequence.astype(str)
    cores = {r.filename: core(decode(r.pwm)) for r in tr.itertuples()}
    pairs = []
    fams = tr.groupby("family_name")
    rng = np.random.default_rng(0)
    seen = set()
    for fi, (fam, grp) in enumerate(fams):
        rows = grp.reset_index(drop=True)
        n = len(rows)
        if n < 2:
            continue
        fns = rows.filename.tolist(); seqs = rows.seq.tolist()
        for i in range(n):
            cand = []
            for j in range(n):
                if i == j:
                    continue
                idv = ident(seqs[i], seqs[j])
                if ID_LO <= idv <= ID_HI:
                    cand.append((j, idv))
            if not cand:
                continue
            # score neighbours by ΔPWM; keep top-K/2 switching + K/2 random neutral
            scored = []
            for j, idv in cand:
                d = dpwm(cores[fns[i]], cores[fns[j]])
                scored.append((j, idv, d))
            scored.sort(key=lambda t: -t[2])
            top = scored[:max(1, K_PER_PROT // 2)]
            rest = scored[max(1, K_PER_PROT // 2):]
            rnd = [rest[k] for k in rng.choice(len(rest), min(len(rest), K_PER_PROT // 2), replace=False)] if rest else []
            for j, idv, d in top + rnd:
                key = tuple(sorted((fns[i], fns[j])))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"a": fns[i], "b": fns[j], "ident": round(idv, 3), "dpwm": round(d, 4)})
        print(f"[{fi+1}] {fam:22s} n={n:4d} cum_pairs={len(pairs)}", flush=True)
    json.dump(pairs, open(OUT, "w"))
    dv = np.array([p["dpwm"] for p in pairs])
    print(f"\nmined {len(pairs)} pairs | dpwm: median={np.median(dv):.3f} "
          f"switching(>0.15)={int((dv>0.15).sum())} neutral(<0.05)={int((dv<0.05).sum())}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
