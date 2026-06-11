#!/usr/bin/env python
"""Build 2D contact-distillation targets (PWM-column x DBD-residue) keyed by training filename.

Each TFScope training example whose filename begins with `<pdb>_<chain>_` corresponds directly to a
co-crystal structure. For those, we re-parse the structure (gemmi), compute residue->base contacts,
and align both axes onto the model's index space:
  * protein residues -> the training sequence positions (Needleman-Wunsch of the structure chain
    sequence vs the parquet `sequence`), restricted to the DBD window;
  * DNA bases -> PWM columns (best offset+orientation match of the crystal strand to the target-PWM
    consensus; the antiparallel partner base maps to the same base-pair column).
Output: data/contact_maps/contact_targets.json  { filename: {"L": seqlen, "cols": {col: [[res_idx, w], ...]}} }
which the dataset turns into contact_target (Lq x Lk) + contact_base_mask.

Run from repo root in the `tfscope` env (gemmi + biopython).
"""
import os, sys, json, glob, re
import numpy as np, pandas as pd
import gemmi
from Bio import Align
sys.path.insert(0, "src")
from tfscope.models.alignment import align_pwm  # noqa  (kept for parity; we use our own base map)

PARQUET = "data/processed/tf_pwm_aug_dbd_canon_trim_rebin34.parquet"
SPLIT = "data/processed/splits/cluster40/split.json"
CACHE = "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/pdb"
OUT = "data/contact_maps/contact_targets.json"
THRESH, TAU, MAXSHIFT = 4.5, 2.0, 8
AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H",
          "ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
          "TYR":"Y","VAL":"V"}
DNA1 = {"DA":"A","DC":"C","DG":"G","DT":"T"}
COMP = {"A":"T","T":"A","C":"G","G":"C"}
_aligner = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                                 open_gap_score=-5, extend_gap_score=-0.5)

def load_struct_chains(pid):
    st = gemmi.read_structure(f"{CACHE}/{pid}.cif"); st.setup_entities(); m = st[0]
    prot, dna = {}, {}
    for chain in m:
        for res in chain:
            rn = res.name.strip()
            xyz = [[a.pos.x, a.pos.y, a.pos.z] for a in res
                   if a.element.name != "H" and a.altloc in ("\x00", "", " ", "A")]
            if not xyz: continue
            if rn in AA3to1:
                prot.setdefault(chain.name, []).append((AA3to1[rn], np.asarray(xyz)))
            elif rn in DNA1:
                dna.setdefault(chain.name, []).append((DNA1[rn], np.asarray(xyz)))
    return prot, dna

def min_d(a, b):
    return float(np.sqrt(((a[:,None]-b[None])**2).sum(-1)).min())

def nearest_two_dna(pres, dna):
    pc = np.concatenate([x for _, x in pres], 0)
    order = sorted(dna.items(), key=lambda kv: min_d(pc, np.concatenate([x for _, x in kv[1]], 0)))
    return order[:2]

def best_offset(d1, cons):
    """Return (offset, orient) mapping strand1 index -> column for the best match to consensus."""
    best = (-1e9, 0, "fwd")
    for orient, seq in (("fwd", d1), ("rev", "".join(COMP[c] for c in reversed(d1)))):
        for s in range(-len(seq), len(cons) + 1):
            sc = sum(1 for i, ch in enumerate(seq) if 0 <= i + s < len(cons) and cons[i + s] == ch)
            if sc > best[0]:
                best = (sc, s, orient)
    return best[1], best[2]

def build_one(pid, pchain, seq, dbd_start, dbd_end, target_pwm):
    prot, dna = load_struct_chains(pid)
    if pchain not in prot or len(dna) < 1:
        return None
    pres = prot[pchain]
    struct_seq = "".join(c for c, _ in pres)
    # protein residue -> training-sequence index via global alignment
    aln = _aligner.align(seq, struct_seq)[0]
    # map struct index -> seq index
    s2q = {}
    qi = ti = 0
    for (q0, q1), (t0, t1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(q1 - q0):
            s2q[t0 + k] = q0 + k
    # DNA strands
    two = nearest_two_dna(pres, dna)
    strand1 = two[0][1]
    d1 = "".join(c for c, _ in strand1)
    cons = "".join("ACGT"[i] for i in target_pwm.argmax(0))
    off, orient = best_offset(d1, cons)
    L = target_pwm.shape[1]
    def col_of_strand1(i):
        c = i + off
        if orient == "rev":
            c = (len(d1) - 1 - i) + off
        return c if 0 <= c < L else None
    # base -> column for each strand; strand2 base j pairs strand1[len2-1-j]
    base_cols = {}   # (chain_idx, base_idx) -> column
    for i in range(len(strand1)):
        base_cols[(0, i)] = col_of_strand1(i)
    if len(two) > 1:
        strand2 = two[1][1]; n2 = len(strand2)
        for j in range(n2):
            partner = len(d1) - 1 - j
            base_cols[(1, j)] = col_of_strand1(partner) if 0 <= partner < len(d1) else None
    # contacts -> target rows
    strands = [strand1] + ([two[1][1]] if len(two) > 1 else [])
    cols = {}
    for si, strand in enumerate(strands):
        for bi, (_, bxyz) in enumerate(strand):
            col = base_cols.get((si, bi))
            if col is None: continue
            for ri, (_, rxyz) in enumerate(pres):
                qidx = s2q.get(ri)
                if qidx is None or not (dbd_start <= qidx < dbd_end): continue
                d = min_d(rxyz, bxyz)
                if d <= THRESH:
                    w = float(np.exp(-d / TAU))
                    cols.setdefault(col, {}).setdefault(qidx, 0.0)
                    cols[col][qidx] += w
    # normalize each column row
    out_cols = {}
    for col, rowd in cols.items():
        tot = sum(rowd.values())
        if tot <= 0: continue
        out_cols[int(col)] = [[int(k), float(v / tot)] for k, v in rowd.items()]
    if not out_cols:
        return None
    return {"L": len(seq), "cols": out_cols}

def main():
    df = pd.read_parquet(PARQUET)
    sp = json.load(open(SPLIT))
    keep = set(sp["train"]) | set(sp.get("val", []))
    df = df[df["filename"].isin(keep)].copy()
    teach = set(os.path.basename(f)[:4].lower() for f in glob.glob(f"{CACHE}/*.cif"))
    targets = {}; ok = miss = fail = 0
    homeo_check = []
    for _, r in df.iterrows():
        m = re.match(r'^([0-9a-zA-Z]{4})_([A-Za-z0-9]+)_', r["filename"])
        if not m: continue
        pid, pch = m.group(1).lower(), m.group(2)
        if pid not in teach: miss += 1; continue
        pwm = np.frombuffer(r["pwm"], np.float32).reshape(4, -1)
        try:
            t = build_one(pid, pch, r["sequence"], int(r["dbd_start"]), int(r["dbd_end"]), pwm)
        except Exception:
            t = None
        if t is None: fail += 1; continue
        targets[r["filename"]] = t; ok += 1
        if r["family_name"] == "Homeodomain" and len(homeo_check) < 3:
            homeo_check.append((r["filename"], t))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(targets, open(OUT, "w"))
    print(f"built 2D contact targets: ok={ok}  no-structure={miss}  align-fail={fail}")
    print(f"  saved -> {OUT}  ({len(targets)} training files)")
    n_cols = [len(t["cols"]) for t in targets.values()]
    print(f"  per-file contacted columns: mean {np.mean(n_cols):.1f}  median {int(np.median(n_cols))}")
    for fn, t in homeo_check:
        print(f"\n  HOMEODOMAIN check {fn}: {len(t['cols'])} columns")
        for c in sorted(t["cols"])[:4]:
            res = sorted(t["cols"][c], key=lambda x: -x[1])[:4]
            print(f"    col {c}: residues " + ", ".join(f"{i}({w:.2f})" for i, w in res))

if __name__ == "__main__":
    main()
