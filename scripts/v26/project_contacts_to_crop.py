#!/usr/bin/env python
"""v26 Phase-2 step 3: project canonical contacts into a crop's coordinate frame.

This is the step that replaces v24's silent clipping (audit Finding C: 1,034/10,232 contact links
destroyed, 122 PWM columns emptied, all with no record) with explicit MASKING.

Rules, enforced here and asserted in tests/v26/test_contact_projection.py:
  1. A contact is never MOVED. Its UniProt index is authoritative; the crop offset is arithmetic.
  2. A contact outside the crop is kept with in_crop=False so its loss can be masked, not deleted.
  3. A contact with no UniProt index is kept with mapping_status preserved and in_crop=False.
  4. Chain identity is asserted: a contact only projects onto the example whose accession it matches.
  5. Test-structure contacts are tagged eval_only and are excluded from training target files.

Coordinate chain completed here:
    PDB auth residue -> chain-local idx -> UniProt idx -> crop idx (-> tensor idx at collate)

Input : data/contacts_v26/contacts_canonical.parquet
        data/processed/v26/v26_{core,flank20,flank32}.parquet
Output: data/contacts_v26/projected_{dataset}.parquet   one row per (example, contact)
        results/v26/contact_projection_report.csv       per-dataset coverage
        results/v26/contact_projection_summary.json

  python scripts/v26/project_contacts_to_crop.py
  python scripts/v26/project_contacts_to_crop.py --datasets core flank20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

CD = "data/contacts_v26"
V26D = "data/processed/v26"
RESD = "results/v26"


def project(ds: str, contacts: pd.DataFrame, split_of: dict | None):
    ex = pd.read_parquet(f"{V26D}/v26_{ds}.parquet")
    ex = ex[ex.structure_id.notna() & ex.structure_chain.notna()].copy()
    ex["seq_len"] = ex.sequence.str.len()

    # Index contacts by the chain they were observed on.
    by_chain = {k: g for k, g in contacts.groupby(["pdb_id", "protein_chain"])}

    rows = []
    for e in ex.itertuples():
        pdb, chain = str(e.structure_id).upper(), str(e.structure_chain)
        # primary chain contacts
        groups = [(by_chain.get((pdb, chain)), "primary")]
        # partner chains: every OTHER protein chain of the same structure
        for (p, c), g in by_chain.items():
            if p == pdb and c != chain:
                groups.append((g, "partner"))

        for g, role in groups:
            if g is None or not len(g):
                continue
            for c in g.itertuples():
                # RULE 4: chain identity. A primary contact must belong to this example's accession.
                if role == "primary" and c.accession is not None \
                        and c.accession != e.primary_accession:
                    continue
                unp = c.unp_residue_index
                if unp is None or (isinstance(unp, float) and np.isnan(unp)):
                    crop_idx, in_crop, status = None, False, str(c.mapping_status)
                else:
                    crop_idx = int(unp) - int(e.crop_unp_start)     # 0-based crop coordinate
                    in_crop = 0 <= crop_idx < int(e.seq_len)
                    status = "mapped" if in_crop else "outside_crop"
                rows.append({
                    "example_id": e.example_id,
                    "target_unit_id": e.target_unit_id,
                    "dataset": ds,
                    "chain_entity_id": c.chain_entity_id,
                    "chain_role": role,
                    "accession": c.accession,
                    "pdb_id": c.pdb_id,
                    "pdb_auth_resid": c.pdb_auth_resid,
                    "chain_local_idx": c.chain_local_idx,
                    "unp_residue_index": (None if unp is None else
                                          (None if (isinstance(unp, float) and np.isnan(unp))
                                           else int(unp))),
                    "crop_residue_idx": crop_idx,
                    "in_crop": bool(in_crop),
                    "projection_status": status,
                    "aa": c.aa,
                    "duplex_id": c.duplex_id,
                    "dna_chain": c.dna_chain,
                    "dna_auth_resid": c.dna_auth_resid,
                    "dna_chain_index": c.dna_chain_index,
                    "base": c.base,
                    "min_distance": c.min_distance,
                    "eval_only": (split_of or {}).get(e.legacy_filename) == "test",
                })
    df = pd.DataFrame(rows)
    p = f"{CD}/projected_{ds}.parquet"
    df.to_parquet(p, index=False)
    return df, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["core", "flank20", "flank32"])
    a = ap.parse_args()
    os.makedirs(RESD, exist_ok=True)

    contacts = pd.read_parquet(f"{CD}/contacts_canonical.parquet")
    print(f"canonical contacts: {len(contacts)}", flush=True)

    # legacy split, used ONLY to tag eval_only until the v26 split exists (Phase 3)
    split_of = None
    sp = "data/processed/splits/train_v22/split.json"
    if os.path.exists(sp):
        j = json.load(open(sp))
        split_of = {fn: s for s, fns in j.items() for fn in fns}

    report, summary = [], {}
    for ds in a.datasets:
        df, p = project(ds, contacts, split_of)
        tot = len(df)
        inc = int(df.in_crop.sum())
        st = df.projection_status.value_counts().to_dict()
        # per-role coverage
        roles = {}
        for role in ("primary", "partner"):
            s = df[df.chain_role == role]
            roles[role] = {"contacts": int(len(s)),
                           "in_crop": int(s.in_crop.sum()),
                           "frac_in_crop": round(float(s.in_crop.mean()) if len(s) else 0.0, 4)}
        summary[ds] = {
            "examples_with_structure": int(df.example_id.nunique()),
            "projected_rows": tot,
            "in_crop": inc,
            "masked_not_dropped": tot - inc,
            "frac_in_crop": round(inc / max(tot, 1), 4),
            "status_counts": st,
            "by_role": roles,
            "eval_only_rows": int(df.eval_only.sum()),
            "path": p,
        }
        report.append({"dataset": ds, "rows": tot, "in_crop": inc,
                       "masked": tot - inc, "frac_in_crop": round(inc / max(tot, 1), 4),
                       "primary_frac": roles["primary"]["frac_in_crop"],
                       "partner_frac": roles["partner"]["frac_in_crop"]})
        print(f"  {ds:9s} rows={tot:7d} in_crop={inc:7d} "
              f"({100*inc/max(tot,1):.1f}%) masked={tot-inc:6d} "
              f"primary={roles['primary']['frac_in_crop']:.3f} "
              f"partner={roles['partner']['frac_in_crop']:.3f}", flush=True)

    pd.DataFrame(report).to_csv(f"{RESD}/contact_projection_report.csv", index=False)
    json.dump(summary, open(f"{RESD}/contact_projection_summary.json", "w"), indent=2)

    print("\n=== projection summary ===")
    for ds, s in summary.items():
        print(f"  {ds}: {s['in_crop']}/{s['projected_rows']} in crop, "
              f"{s['masked_not_dropped']} MASKED (never dropped), "
              f"{s['eval_only_rows']} eval_only")
        print(f"      status: {s['status_counts']}")
    print("\n  NOTE: 2-D PWM-column x residue assignment is step 3b "
          "(needs base->column alignment); bp coordinates are preserved here for it.")
    print(f"  wrote {RESD}/contact_projection_report.csv")


if __name__ == "__main__":
    main()
