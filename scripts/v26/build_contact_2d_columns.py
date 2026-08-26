#!/usr/bin/env python
"""v26 Phase-2 step 3b: assign a PWM column to every empirical contact.

Completes the 2-D (PWM column x residue) contact map. Kept strictly separate from the rule-based
recognition prior — v24 conflated their roles and, worse, weighted the rule-based prior (0.3)
ABOVE the empirical distillation (0.2). v26 inverts that.

Method, per (example, DNA duplex):
  1. Re-read the duplex's FULL DNA strand sequences from the mmCIF (cheap: no distance matrices).
  2. Pick the longer strand as reference; pair the other strand by aligning its reverse complement,
     so a base on either strand maps to the same base-PAIR column.
  3. Align the reference strand to the target PWM by scanning every offset and both orientations,
     scoring with PWM log-odds (background 0.25) — the same criterion as the legacy builder.
  4. Each contacted base inherits the PWM column of its aligned position.

Contacts whose base falls outside the aligned PWM window get pwm_column=None and
column_status='outside_pwm_window' — recorded, never clipped (audit Finding C).

Input : data/contacts_v26/projected_{dataset}.parquet, data/processed/v26/v26_{dataset}.parquet
Output: data/contacts_v26/contacts2d_{dataset}.parquet
        results/v26/contact_2d_report.csv / contact_2d_summary.json

  python scripts/v26/build_contact_2d_columns.py --datasets core flank20 flank32
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CIF = "data/raw/pdb_cif_cache"
CD = "data/contacts_v26"
V26D = "data/processed/v26"
RESD = "results/v26"
DNA1 = {"DA": "A", "DC": "C", "DG": "G", "DT": "T"}
B2I = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
PROGRESS_EVERY = 100


def revcomp(s: str) -> str:
    return "".join(COMP.get(c, "N") for c in reversed(s))


def dna_strands(pdb_id: str) -> dict[str, list[tuple[int, str]]]:
    """chain -> [(auth_resid, base), ...] ordered by auth_resid. Cached per process."""
    from Bio.PDB import MMCIFParser
    path = os.path.join(CIF, f"{pdb_id.lower()}.cif")
    if not os.path.exists(path):
        return {}
    st = MMCIFParser(QUIET=True).get_structure(pdb_id, path)
    model = next(iter(st))
    out = {}
    for ch in model:
        res = []
        for r in ch:
            nm = r.get_resname().strip().upper()
            if nm in DNA1:
                res.append((int(r.id[1]), DNA1[nm]))
        if res:
            out[ch.id] = sorted(res, key=lambda x: x[0])
    return out


def pwm_of(row) -> np.ndarray | None:
    b = row.pwm
    if not isinstance(b, (bytes, bytearray)):
        return None
    a = np.frombuffer(b, dtype=np.float32)
    if a.size % 4:
        return None
    m = a.reshape(4, -1).astype(np.float64)
    L = int(row.motif_length)
    return m[:, :L] if 0 < L <= m.shape[1] else m


def align_strand_to_pwm(strand: str, pwm: np.ndarray):
    """Best (offset, orientation) placing the PWM onto the strand.

    Three changes over the first version, which produced an IC-enrichment ratio of only 1.11
    (i.e. contacts landed on informative columns barely above chance):

      1. IC-WEIGHTED scoring. Plain summed log-odds lets many low-information columns dominate
         by sheer count. Each column's log-odds is now weighted by its information content and
         the total is normalised by sum(IC), so the placement is driven by the columns that
         actually carry specificity.
      2. MARGIN against the runner-up. Short crystallised oligos admit several near-equal
         offsets, and palindromic motifs admit both orientations, so the arg-max alone is often
         arbitrary. We return the gap to the best ALTERNATIVE placement (excluding offsets within
         +/-2 of the winner in the same orientation, which are the same alignment jittered).
      3. Both score and margin are returned so a threshold can be chosen from the data
         (see diagnose_pwm_column_alignment.py --sweep) instead of guessed.

    Returns (offset, revcomp_used, score, margin). Column j sits at offset + j in the
    ORIENTED strand. offset is None when the strand is shorter than the PWM.
    """
    lo = np.log2(np.clip(pwm, 1e-6, 1.0) / 0.25)
    icw = np.clip(2.0 + (np.clip(pwm, 1e-9, 1.0)
                         * np.log2(np.clip(pwm, 1e-9, 1.0))).sum(0), 0.0, None)
    W = pwm.shape[1]
    denom = float(icw.sum()) or 1.0
    cand = []
    for rc in (False, True):
        s_or = revcomp(strand) if rc else strand
        if len(s_or) < W:
            continue
        idx = np.array([B2I.get(c, -1) for c in s_or])
        for off in range(0, len(s_or) - W + 1):
            win = idx[off:off + W]
            ok = win >= 0
            if not ok.any():
                continue
            cols = np.arange(W)[ok]
            sc = float((icw[cols] * lo[win[ok], cols]).sum()) / denom
            cand.append((sc, off, rc))
    if not cand:
        return None, False, -np.inf, 0.0
    cand.sort(key=lambda x: -x[0])
    best_sc, best_off, best_rc = cand[0]
    margin = 0.0
    for sc, off, rc in cand[1:]:
        if rc == best_rc and abs(off - best_off) <= 2:
            continue                      # same alignment, jittered
        margin = best_sc - sc
        break
    return best_off, best_rc, best_sc, float(margin)


def pair_strands(a: str, b: str):
    """Map index in strand b -> index in strand a via revcomp alignment. Absent if unpaired."""
    from Bio.Align import PairwiseAligner
    al = PairwiseAligner()
    al.mode = "global"
    al.match_score, al.mismatch_score = 2, -1
    al.open_gap_score, al.extend_gap_score = -5, -0.5
    al.target_end_gap_score = al.query_end_gap_score = 0.0
    rb = revcomp(b)                     # rb[k] pairs with b[len(b)-1-k]
    try:
        aln = al.align(a, rb)[0]
    except Exception:
        return {}
    out = {}
    for (a0, a1), (r0, r1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(a1 - a0):
            out[len(b) - 1 - (r0 + k)] = a0 + k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["core", "flank20", "flank32"])
    a = ap.parse_args()
    os.makedirs(RESD, exist_ok=True)

    strand_cache: dict[str, dict] = {}
    report, summary = [], {}

    for ds in a.datasets:
        pp = f"{CD}/projected_{ds}.parquet"
        if not os.path.exists(pp):
            continue
        proj = pd.read_parquet(pp)
        ex = pd.read_parquet(f"{V26D}/v26_{ds}.parquet")
        exi = {r.example_id: r for r in ex.itertuples()}

        print(f"[{ds}] {len(proj)} projected contacts, "
              f"{proj.example_id.nunique()} examples", flush=True)
        col_of = np.full(len(proj), -1, dtype=np.int64)
        status = np.array(["no_alignment"] * len(proj), dtype=object)
        align_score = np.full(len(proj), np.nan)
        align_margin = np.full(len(proj), np.nan)

        t0 = time.time()
        groups = list(proj.groupby(["example_id", "duplex_id"], dropna=False))
        for gi, ((eid, dup), g) in enumerate(groups, 1):
            e = exi.get(eid)
            if e is None:
                continue
            pwm = pwm_of(e)
            if pwm is None or pwm.shape[1] == 0:
                status[g.index] = "no_pwm"
                continue
            pdb = str(g.pdb_id.iloc[0])
            if pdb not in strand_cache:
                strand_cache[pdb] = dna_strands(pdb)
            strands = strand_cache[pdb]
            chains = sorted(set(g.dna_chain.dropna()))
            chains = [c for c in chains if c in strands]
            if not chains:
                status[g.index] = "no_dna_strand"
                continue
            # reference = longest strand of this duplex
            chains.sort(key=lambda c: -len(strands[c]))
            ref = chains[0]
            ref_seq = "".join(b for _, b in strands[ref])
            ref_pos = {auth: i for i, (auth, _) in enumerate(strands[ref])}

            off, rc, ascore, amargin = align_strand_to_pwm(ref_seq, pwm)
            if off is None:
                status[g.index] = "strand_shorter_than_pwm"
                continue
            align_score[g.index] = ascore
            align_margin[g.index] = amargin
            W = pwm.shape[1]
            n = len(ref_seq)

            def ref_index_to_column(i: int):
                """i = index in the FORWARD reference strand."""
                oi = (n - 1 - i) if rc else i          # index in the oriented strand
                j = oi - off
                return int(j) if 0 <= j < W else None

            # pairing maps for the complementary strand(s)
            pairing = {}
            for c in chains[1:]:
                cseq = "".join(b for _, b in strands[c])
                pairing[c] = (pair_strands(ref_seq, cseq),
                              {auth: i for i, (auth, _) in enumerate(strands[c])})

            for r in g.itertuples():
                dc, auth = r.dna_chain, r.dna_auth_resid
                if dc == ref:
                    i = ref_pos.get(auth)
                else:
                    pm, aidx = pairing.get(dc, ({}, {}))
                    i = pm.get(aidx.get(auth)) if aidx.get(auth) is not None else None
                if i is None:
                    status[r.Index] = "base_unpaired"
                    continue
                col = ref_index_to_column(i)
                if col is None:
                    status[r.Index] = "outside_pwm_window"
                else:
                    col_of[r.Index] = col
                    status[r.Index] = "assigned"
            if gi % PROGRESS_EVERY == 0 or gi == len(groups):
                el = time.time() - t0
                print(f"  [{ds}] {gi}/{len(groups)} duplex-groups  "
                      f"{el/60:.1f}m elapsed  eta={(len(groups)-gi)/max(gi/max(el,1e-9),1e-9)/60:.1f}m",
                      flush=True)

        proj = proj.copy()
        proj["pwm_column"] = np.where(col_of >= 0, col_of, None)
        proj["column_status"] = status
        proj["align_score"] = align_score
        proj["align_margin"] = align_margin
        outp = f"{CD}/contacts2d_{ds}.parquet"
        proj.to_parquet(outp, index=False)

        st = pd.Series(status).value_counts().to_dict()
        assigned = int((proj.column_status == "assigned").sum())
        inc = proj[proj.in_crop]
        assigned_inc = int((inc.column_status == "assigned").sum())
        summary[ds] = {
            "rows": int(len(proj)),
            "assigned_pwm_column": assigned,
            "frac_assigned": round(assigned / max(len(proj), 1), 4),
            "in_crop_rows": int(len(inc)),
            "in_crop_assigned": assigned_inc,
            "frac_in_crop_assigned": round(assigned_inc / max(len(inc), 1), 4),
            "status_counts": st,
            "distinct_columns_used": int(proj.pwm_column.dropna().nunique()),
            "path": outp,
        }
        report.append({"dataset": ds, **{k: v for k, v in summary[ds].items()
                                         if k not in ("status_counts", "path")}})
        print(f"  [{ds}] assigned={assigned}/{len(proj)} "
              f"({100*assigned/max(len(proj),1):.1f}%), "
              f"in-crop assigned={assigned_inc}/{len(inc)}", flush=True)
        print(f"  [{ds}] status: {st}", flush=True)

    pd.DataFrame(report).to_csv(f"{RESD}/contact_2d_report.csv", index=False)
    json.dump(summary, open(f"{RESD}/contact_2d_summary.json", "w"), indent=2)
    print(f"\nwrote {RESD}/contact_2d_report.csv")


if __name__ == "__main__":
    main()
