#!/usr/bin/env python
"""Phase-2 invariants: canonical contact coordinates and crop projection.

These encode the fixes for audit Findings C/D — v24 silently destroyed 1,034/10,232 contact links
and emptied 122 PWM columns; v25flank relocated 680 onto flank residues.

  pytest tests/v26/test_contact_projection.py -q
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

CD = "data/contacts_v26"
V26D = "data/processed/v26"
DATASETS = ["core", "flank20", "flank32"]


def _canon():
    return pd.read_parquet(f"{CD}/contacts_canonical.parquet")


def _proj(ds):
    return pd.read_parquet(f"{CD}/projected_{ds}.parquet")


def test_canonical_has_full_coordinate_chain():
    """Every contact must carry each link of PDB auth -> chain-local -> UniProt."""
    c = _canon()
    for col in ["pdb_auth_resid", "chain_local_idx", "unp_residue_index",
                "chain_entity_id", "chain_role", "duplex_id", "dna_auth_resid",
                "base", "min_distance", "mapping_status"]:
        assert col in c.columns, f"missing coordinate column: {col}"
    assert c.pdb_auth_resid.notna().all()
    assert c.chain_local_idx.notna().all()
    assert (c.chain_local_idx >= 0).all()
    assert (c.min_distance <= 4.5 + 1e-6).all(), "contact beyond the 4.5 A cutoff"


def test_unmapped_contacts_are_recorded_not_dropped():
    """A contact without a UniProt index must survive with its status recorded."""
    c = _canon()
    unmapped = c[c.unp_residue_index.isna()]
    assert (unmapped.mapping_status != "mapped").all(), \
        "contact has no UniProt index but is marked mapped"
    # and the converse
    mapped = c[c.mapping_status == "mapped"]
    assert mapped.unp_residue_index.notna().all()


def test_primary_chains_are_fully_accession_covered():
    """Primary chains drive the model input; every one must resolve to an accession."""
    c = _canon()
    prim = c[c.chain_role == "primary"]
    assert prim.accession.notna().all(), \
        f"{int(prim.accession.isna().sum())} primary contacts have no accession"


def test_projection_never_moves_a_contact():
    """crop_residue_idx must equal unp_residue_index - crop_unp_start, exactly."""
    for ds in DATASETS:
        if not os.path.exists(f"{CD}/projected_{ds}.parquet"):
            continue
        p = _proj(ds)
        ex = pd.read_parquet(f"{V26D}/v26_{ds}.parquet")[["example_id", "crop_unp_start"]]
        m = p[p.unp_residue_index.notna()].merge(ex, on="example_id", how="left")
        expect = m.unp_residue_index.astype(int) - m.crop_unp_start.astype(int)
        bad = int((m.crop_residue_idx.astype(int) != expect).sum())
        assert bad == 0, f"{ds}: {bad} contacts were MOVED rather than offset"


def test_out_of_crop_contacts_are_masked_not_deleted():
    """Contacts outside the crop must be present with in_crop=False."""
    for ds in DATASETS:
        if not os.path.exists(f"{CD}/projected_{ds}.parquet"):
            continue
        p = _proj(ds)
        out = p[~p.in_crop]
        assert len(out) > 0, f"{ds}: no masked contacts at all -- were they dropped?"
        assert (out.projection_status != "mapped").all(), \
            f"{ds}: a contact is out of crop but marked mapped"
        # every projected row must carry a decision
        assert p.projection_status.notna().all()


def test_in_crop_indices_are_within_the_sequence():
    for ds in DATASETS:
        if not os.path.exists(f"{CD}/projected_{ds}.parquet"):
            continue
        p = _proj(ds)
        ex = pd.read_parquet(f"{V26D}/v26_{ds}.parquet")[["example_id", "sequence"]]
        ex["seq_len"] = ex.sequence.str.len()
        m = p[p.in_crop].merge(ex[["example_id", "seq_len"]], on="example_id", how="left")
        assert (m.crop_residue_idx >= 0).all()
        assert (m.crop_residue_idx < m.seq_len).all(), \
            f"{ds}: in_crop contact indexes past the end of the crop"


def test_flanking_only_increases_visible_contacts():
    """core ⊂ flank20 ⊂ flank32 means in-crop counts must be monotone non-decreasing."""
    counts = {}
    for ds in DATASETS:
        if os.path.exists(f"{CD}/projected_{ds}.parquet"):
            counts[ds] = int(_proj(ds).in_crop.sum())
    if len(counts) == 3:
        assert counts["core"] <= counts["flank20"] <= counts["flank32"], \
            f"non-monotone in-crop counts: {counts}"
        print(f"    in-crop counts: {counts}")


def test_same_contact_set_across_datasets():
    """The three variants must project the SAME canonical contacts -- only the crop differs.

    This is what makes the flank ablation controlled, which the v24-vs-v25flank comparison
    could not be (v23 clipped 1,034 links to zero; v25flank relocated 680 onto flanks).
    """
    totals = {ds: len(_proj(ds)) for ds in DATASETS
              if os.path.exists(f"{CD}/projected_{ds}.parquet")}
    assert len(set(totals.values())) == 1, \
        f"datasets project different contact counts: {totals}"


def test_test_structures_are_tagged_eval_only():
    for ds in DATASETS:
        if not os.path.exists(f"{CD}/projected_{ds}.parquet"):
            continue
        p = _proj(ds)
        assert "eval_only" in p.columns
        assert int(p.eval_only.sum()) > 0, "no contacts tagged eval_only"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
