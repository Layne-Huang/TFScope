#!/usr/bin/env python
"""Phase-1 invariants for sequence-derived DBD spans.

  pytest tests/v26/test_dbd_spans.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import shutil
import tempfile

import pandas as pd

OUTD = "data/processed/v26"
WL = "data/annotations_v26/dbd_pfam_whitelist.json"
WL_PRE = "data/annotations_v26/dbd_pfam_whitelist.pre_triage.json"
PY = "/data1/leihuang/miniconda3/envs/tfscope/bin/python"


def _cand():
    return pd.read_parquet(f"{OUTD}/dbd_candidates.parquet")


def test_spans_are_within_protein():
    c = _cand()
    assert (c.start >= 1).all(), "1-based coordinates required"
    ok = c.protein_len.notna()
    assert (c.loc[ok, "end"] <= c.loc[ok, "protein_len"]).all(), "span past protein end"
    assert (c.end >= c.start).all()
    assert (c.span_len == c.end - c.start + 1).all()


def test_no_contact_derived_columns():
    """A DBD span must carry no trace of DNA-contact information (audit Finding A)."""
    c = _cand()
    banned = {"n_contact_residues", "contact_start", "contact_end", "pdb_id", "chain_id",
              "min_distance", "dna_contact"}
    assert not (banned & set(c.columns)), f"contact-derived columns present: {banned & set(c.columns)}"


def test_no_family_oracle_columns():
    """Selection must not have used the motif-database family (audit Finding I)."""
    c = _cand()
    banned = {"motif_source", "motif_family", "family_id", "pwm", "gene_symbol"}
    assert not (banned & set(c.columns)), f"oracle columns present: {banned & set(c.columns)}"


def test_tier2_is_rescue_only():
    """Rescue entries may only apply to accessions with no tier-1 DBD."""
    c = _cand()
    t1 = set(c[c.tier == "tier1"].accession)
    t2 = set(c[c.tier == "tier2_rescue"].accession)
    assert not (t1 & t2), f"{len(t1 & t2)} accessions have both tiers; rescue leaked"


def test_rescue_entries_cause_zero_tier1_drift():
    """Rebuilding with the pre-triage whitelist must reproduce tier-1 spans EXACTLY.

    Guards the regression found on 2026-08-14: applying IPR013087 globally lengthened
    534/1357 spans (up to +429 aa on ZNF142), silently redefining the C2H2 DBD.
    """
    if not os.path.exists(WL_PRE):
        return                                    # nothing to compare against
    new = _cand()
    with tempfile.TemporaryDirectory() as td:
        keep = os.path.join(td, "wl.json")
        shutil.copy(WL, keep)
        try:
            shutil.copy(WL_PRE, WL)
            subprocess.run([PY, "scripts/v26/build_dbd_spans.py"], check=True,
                           capture_output=True)
            old = _cand()
        finally:
            shutil.copy(keep, WL)
            subprocess.run([PY, "scripts/v26/build_dbd_spans.py"], check=True,
                           capture_output=True)

    cols = ["accession", "candidate_idx", "start", "end", "span_len"]
    o = old[cols]
    n = new[new.tier == "tier1"][cols]
    # The invariant is NO CHANGE to spans of accessions that already had one. Gaining spans for
    # NEW accessions is legitimate (round-3 added Zn2Cys6/WRKY/B3/Brinker/BZR1/AFT for non-human
    # orthologs). So compare only the accessions present in BOTH builds.
    shared = set(o.accession) & set(n.accession)
    a = o[o.accession.isin(shared)].sort_values(cols[:2]).reset_index(drop=True)
    b = n[n.accession.isin(shared)].sort_values(cols[:2]).reset_index(drop=True)
    if not a.equals(b):
        # Any boundary change on a pre-existing accession must be explicitly reviewed and
        # listed in accepted_span_changes.json with a biological justification.
        acc_path = os.path.join(os.path.dirname(__file__), "accepted_span_changes.json")
        accepted = set()
        if os.path.exists(acc_path):
            for e in json.load(open(acc_path))["accepted"]:
                accepted.add((e["accession"], int(e["candidate_idx"])))
        m = a.merge(b, on=cols[:2], suffixes=("_old", "_new"), how="outer", indicator=True)
        changed = m[(m._merge != "both") | (m.start_old != m.start_new)
                    | (m.end_old != m.end_new)]
        unreviewed = [(r.accession, int(r.candidate_idx)) for r in changed.itertuples()
                      if (r.accession, int(r.candidate_idx)) not in accepted]
        assert not unreviewed, (
            f"{len(unreviewed)} UNREVIEWED span changes on pre-existing accessions "
            f"(first 10: {unreviewed[:10]}). Review each and add it to "
            f"tests/v26/accepted_span_changes.json with a justification, or revert the "
            f"whitelist change.")
        print(f"    ({len(changed)} span change(s), all reviewed and accepted)")
    gained = set(n.accession) - set(o.accession)
    print(f"    (zero drift on {len(shared)} shared accessions; "
          f"+{len(gained)} newly covered)")


def test_residual_missing_dbd_is_explicit():
    """Every accession without a DBD must carry a recorded decision, never a silent drop."""
    p = "results/v26/missing_dbd_triage.csv"
    assert os.path.exists(p), "triage table missing"
    t = pd.read_csv(p)
    assert t.decision.notna().all()
    unclassified = t[t.decision == "unclassified"]
    assert len(unclassified) <= 5, (
        f"{len(unclassified)} unclassified accessions -- triage each before proceeding:\n"
        f"{unclassified[['accession', 'gene', 'v23_rows']].to_string(index=False)}")


if __name__ == "__main__":
    for fn in [test_spans_are_within_protein, test_no_contact_derived_columns,
               test_no_family_oracle_columns, test_tier2_is_rescue_only,
               test_rescue_entries_cause_zero_tier1_drift,
               test_residual_missing_dbd_is_explicit]:
        fn()
        print(f"PASS {fn.__name__}")
