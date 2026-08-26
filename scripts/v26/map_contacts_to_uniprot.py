#!/usr/bin/env python
"""v26 Phase-2 step 2: add UniProt coordinates to every parsed contact.

Completes the traceable chain
    PDB author residue -> chain-local index -> UniProt index
by aligning each observed PDB chain sequence to its UniProt sequence. Crystal constructs carry
expression tags, point mutations and disorder gaps, so the mapping is genuinely partial for some
chains -- every unmapped residue is REPORTED, never silently dropped (audit Finding C).

Accession per chain, in order of preference:
  1. SIFTS (pdb_id, chain_id) -> accession, highest coverage
  2. the v26_core row's primary_accession when this chain is that row's structure_chain

chain_role is 'primary' when the chain is the one the example is built from, else 'partner'
(partners participate in the Phase-3 Assembly-OOD audit).

Input : data/contacts_v26/contacts_raw.jsonl.gz
Output: data/contacts_v26/contacts_canonical.parquet
        results/v26/contact_mapping_report.csv       per-chain coverage
        results/v26/contact_mapping_summary.json

  python scripts/v26/map_contacts_to_uniprot.py
"""
from __future__ import annotations

import gzip
import json
import os
import time

import numpy as np
import pandas as pd

OUTD = "data/contacts_v26"
RESD = "results/v26"
V26D = "data/processed/v26"
PROGRESS_EVERY = 50


def _aligner():
    from Bio.Align import PairwiseAligner
    al = PairwiseAligner()
    al.mode = "global"
    al.open_gap_score = -10
    al.extend_gap_score = -0.5
    al.target_end_gap_score = 0.0
    al.query_end_gap_score = 0.0
    try:
        from Bio.Align import substitution_matrices
        al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    except Exception:
        al.match_score, al.mismatch_score = 2, -1
    return al


def align_map(chain_seq: str, unp_seq: str, al):
    """chain-local idx (0-based) -> UniProt idx (1-based). Returns (map, identity_on_mapped)."""
    if not chain_seq or not unp_seq:
        return {}, 0.0
    try:
        aln = al.align(unp_seq, chain_seq)[0]
    except Exception:
        return {}, 0.0
    m, match, tot = {}, 0, 0
    for (u0, u1), (c0, c1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(u1 - u0):
            ui, ci = u0 + k, c0 + k
            m[ci] = ui + 1
            tot += 1
            if unp_seq[ui] == chain_seq[ci]:
                match += 1
    return m, (match / tot if tot else 0.0)


def main():
    os.makedirs(RESD, exist_ok=True)
    al = _aligner()

    acc_df = pd.read_parquet(f"{V26D}/accessions.parquet")
    unp_seq = dict(zip(acc_df.accession, acc_df.sequence))
    sif = pd.read_parquet(f"{V26D}/sifts_mappings.parquet")
    sif_best = {}
    for r in sif.itertuples():
        k = (str(r.pdb_id).upper(), str(r.chain_id))
        cov = float(r.coverage or 0)
        if k not in sif_best or cov > sif_best[k][1]:
            sif_best[k] = (r.accession, cov)

    core = pd.read_parquet(f"{V26D}/v26_core.parquet")
    # (pdb, chain) -> the accession the example was built from
    example_acc = {}
    for r in core.itertuples():
        if r.structure_id and r.structure_chain:
            example_acc[(str(r.structure_id).upper(), str(r.structure_chain))] = r.primary_accession

    raw = f"{OUTD}/contacts_raw.jsonl.gz"
    rows, report = [], []
    n_pdb = 0
    t0 = time.time()
    with gzip.open(raw, "rt") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            n_pdb += 1
            pdb = rec["pdb_id"]
            chains = rec.get("chains", {})
            cmaps = {}
            for ch, meta in chains.items():
                k = (pdb, ch)
                accn = None
                src = None
                if k in sif_best and sif_best[k][0] in unp_seq:
                    accn, src = sif_best[k][0], "sifts"
                if accn is None and k in example_acc and example_acc[k] in unp_seq:
                    accn, src = example_acc[k], "example_primary"
                cseq = meta["sequence"]
                if accn is None:
                    cmaps[ch] = ({}, None, "no_accession")
                    report.append({"pdb_id": pdb, "chain": ch, "accession": None,
                                   "accession_source": None, "n_residues": len(cseq),
                                   "n_mapped": 0, "frac_mapped": 0.0, "identity": None,
                                   "status": "no_accession"})
                    continue
                m, ident = align_map(cseq, unp_seq[accn], al)
                cmaps[ch] = (m, accn, "ok" if m else "align_failed")
                report.append({"pdb_id": pdb, "chain": ch, "accession": accn,
                               "accession_source": src, "n_residues": len(cseq),
                               "n_mapped": len(m),
                               "frac_mapped": round(len(m) / max(len(cseq), 1), 4),
                               "identity": round(ident, 4),
                               "status": "ok" if m else "align_failed"})
            for c in rec.get("contacts", []):
                ch = c["protein_chain"]
                m, accn, st = cmaps.get(ch, ({}, None, "no_chain_meta"))
                li = int(c["chain_local_idx"])
                unp = m.get(li)
                is_primary = (pdb, ch) in example_acc
                rows.append({
                    "pdb_id": pdb,
                    "protein_chain": ch,
                    "chain_entity_id": f"{pdb}_{ch}",
                    "chain_role": "primary" if is_primary else "partner",
                    "accession": accn,
                    "pdb_auth_resid": c["pdb_auth_resid"],
                    "pdb_icode": c.get("pdb_icode", ""),
                    "chain_local_idx": li,
                    "unp_residue_index": unp,             # 1-based; None = UNMAPPED (reported)
                    "aa": c["aa"],
                    "dna_chain": c["dna_chain"],
                    "duplex_id": c["duplex_id"],
                    "dna_auth_resid": c["dna_auth_resid"],
                    "dna_chain_index": c["dna_chain_index"],
                    "base": c["base"],
                    "min_distance": c["min_distance"],
                    "mapping_status": ("mapped" if unp is not None
                                       else ("unmapped_residue" if st == "ok" else st)),
                })
            if n_pdb % PROGRESS_EVERY == 0:
                el = time.time() - t0
                print(f"  mapped {n_pdb} structures, {len(rows)} contacts, "
                      f"{el/60:.1f}m elapsed", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUTD}/contacts_canonical.parquet", index=False)
    rep = pd.DataFrame(report)
    rep.to_csv(f"{RESD}/contact_mapping_report.csv", index=False)

    total = len(df)
    mapped = int((df.mapping_status == "mapped").sum())
    summary = {
        "structures": n_pdb,
        "contacts_total": total,
        "contacts_mapped_to_uniprot": mapped,
        "contacts_unmapped": total - mapped,
        "frac_mapped": round(mapped / max(total, 1), 4),
        "mapping_status_counts": df.mapping_status.value_counts().to_dict(),
        "chain_role_counts": df.chain_role.value_counts().to_dict(),
        "chains_total": int(len(rep)),
        "chains_ok": int((rep.status == "ok").sum()),
        "chains_no_accession": int((rep.status == "no_accession").sum()),
        "chains_align_failed": int((rep.status == "align_failed").sum()),
        "chain_frac_mapped_median": float(rep.frac_mapped.median()),
        "chain_identity_median": float(rep.identity.dropna().median()) if rep.identity.notna().any() else None,
        "chains_below_50pct_mapped": int((rep.frac_mapped < 0.5).sum()),
    }
    json.dump(summary, open(f"{RESD}/contact_mapping_summary.json", "w"), indent=2)

    print("\n=== UniProt mapping summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    worst = rep[rep.status == "ok"].nsmallest(8, "frac_mapped")
    if len(worst):
        print("\n  worst-mapped chains:")
        print(worst[["pdb_id", "chain", "accession", "n_residues", "n_mapped",
                     "frac_mapped", "identity"]].to_string(index=False))
    print(f"\n  wrote {OUTD}/contacts_canonical.parquet, {RESD}/contact_mapping_report.csv")


if __name__ == "__main__":
    main()
