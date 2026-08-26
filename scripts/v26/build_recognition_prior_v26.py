#!/usr/bin/env python
"""v26 Phase-2 step 4: rule-based recognition prior in canonical coordinates.

Kept STRICTLY SEPARATE from empirical contacts. v24 conflated the two roles and weighted the
rule-based prior (v18_contact_weight 0.3) ABOVE the empirical 2-D distillation
(contact_distill_weight 0.2). v26 inverts that: this prior is a LOW-weight auxiliary objective only.

Also fixes the indexing defect behind audit Finding C at its source: in v23,
77/228 recognition-prior entries already pointed beyond their own crop, and the loader dropped
them with `if 0 <= p < len(recog)`. Here the prior is computed in UniProt coordinates from the
DBD span itself, so out-of-range is impossible by construction; anything unresolvable is REPORTED.

Rules are derived from established structural biology per DBD family (not from any test structure),
so there is no benchmark leakage. Only families with a well-defined recognition element get a
prior; everything else is reported as `no_rule` and simply has no auxiliary target.

Output: data/contacts_v26/recognition_prior_{dataset}.parquet
          example_id, crop_residue_idx, unp_residue_index, family, rule, weight
        results/v26/recognition_prior_report.csv

  python scripts/v26/build_recognition_prior_v26.py
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

CD = "data/contacts_v26"
V26D = "data/processed/v26"
RESD = "results/v26"

# C2H2: C-x(2-4)-C-x(3)-[FY]-x(5)-[hydrophobic]-x(2)-H-x(3-5)-H
# The recognition helix runs from ~2 residues after the 2nd Cys to the 2nd His.
C2H2_RE = re.compile(r"C.{2,4}C(.{8,18})H.{3,6}H")


def _frac_span(n: int, lo: float, hi: float) -> list[int]:
    """Residue offsets covering the fractional window [lo, hi) of a domain of length n."""
    a, b = int(round(lo * n)), int(round(hi * n))
    return list(range(max(0, a), min(n, max(a + 1, b))))


def prior_offsets(dbd: str, families: str):
    """Return (offsets_into_dbd, rule_name). Offsets are 0-based within the DBD span."""
    n = len(dbd)
    if n < 8:
        return [], "too_short"
    fams = set(str(families).split(";"))

    if fams & {"C2H2", "C2H2_short", "C2H2_medium", "C2H2_long"}:
        off = []
        for m in C2H2_RE.finditer(dbd):
            # the captured group spans 2nd-Cys+1 .. 1st-His-1; the recognition helix
            # positions -1,2,3,6 are canonical, so mark the whole helix generously
            s, e = m.start(1), m.end(1)
            off.extend(range(s, e))
        if off:
            return sorted(set(off)), "C2H2_recognition_helix_regex"
        return [], "C2H2_regex_no_match"

    if "Homeodomain" in fams or "Homeo_prospero" in fams:
        # helix 3 (recognition helix) is ~residues 42-58 of a 60-aa homeodomain
        return _frac_span(n, 0.68, 0.98), "homeodomain_helix3"
    if "bZIP" in fams:
        # basic region N-terminal to the leucine zipper
        return _frac_span(n, 0.05, 0.45), "bZIP_basic_region"
    if "bHLH" in fams:
        # basic region at the N-terminus
        return _frac_span(n, 0.0, 0.28), "bHLH_basic_region"
    if "Forkhead" in fams:
        return _frac_span(n, 0.35, 0.62), "forkhead_helix3"
    if "ETS" in fams:
        return _frac_span(n, 0.55, 0.85), "ETS_helix3"
    if "Nuclear_Receptor" in fams:
        return _frac_span(n, 0.10, 0.40), "NR_P_box_helix1"
    if "HMG_box" in fams:
        return _frac_span(n, 0.0, 0.35), "HMG_box_helix1"
    if "Rel_homology" in fams or "p53" in fams or "STAT" in fams or "T_box" in fams \
            or "Runt" in fams or "NDT80_p53like" in fams:
        # immunoglobulin-like / p53-like folds read DNA through loops distributed across the
        # domain; no compact single element -> deliberately no prior rather than a guess.
        return [], "no_compact_recognition_element"
    return [], "no_rule"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["core", "flank20", "flank32"])
    a = ap.parse_args()
    os.makedirs(RESD, exist_ok=True)

    report = []
    for ds in a.datasets:
        p = f"{V26D}/v26_{ds}.parquet"
        if not os.path.exists(p):
            continue
        ex = pd.read_parquet(p)
        rows, rules = [], {}
        for e in ex.itertuples():
            seq = str(e.sequence)
            d0, d1 = int(e.dbd_start), int(e.dbd_end)     # crop-local, 0-based half-open
            dbd = seq[d0:d1]
            offs, rule = prior_offsets(dbd, e.dbd_families_for_analysis_only)
            rules[rule] = rules.get(rule, 0) + 1
            for o in offs:
                ci = d0 + o
                if not (0 <= ci < len(seq)):
                    # impossible by construction; assert loudly rather than drop silently
                    raise AssertionError(
                        f"prior index {ci} outside crop for {e.example_id} (len {len(seq)})")
                rows.append({
                    "example_id": e.example_id,
                    "dataset": ds,
                    "crop_residue_idx": ci,
                    "unp_residue_index": int(e.crop_unp_start) + ci,
                    "family": str(e.dbd_families_for_analysis_only),
                    "rule": rule,
                    "weight": 1.0,
                })
        df = pd.DataFrame(rows)
        outp = f"{CD}/recognition_prior_{ds}.parquet"
        df.to_parquet(outp, index=False)
        covered = df.example_id.nunique() if len(df) else 0
        print(f"  {ds:9s} examples={len(ex)} with_prior={covered} "
              f"({100*covered/max(len(ex),1):.1f}%) marked_residues={len(df)}", flush=True)
        print(f"    rules: {dict(sorted(rules.items(), key=lambda x: -x[1]))}", flush=True)
        report.append({"dataset": ds, "examples": len(ex), "examples_with_prior": covered,
                       "marked_residues": len(df),
                       "frac_with_prior": round(covered / max(len(ex), 1), 4),
                       "rules": json.dumps(rules)})

    pd.DataFrame(report).to_csv(f"{RESD}/recognition_prior_report.csv", index=False)
    print(f"\n  NOTE: this prior is auxiliary and must be weighted BELOW the empirical 2-D "
          f"contact loss (v24 had it the other way round: 0.3 vs 0.2).")
    print(f"  wrote {RESD}/recognition_prior_report.csv")


if __name__ == "__main__":
    main()
