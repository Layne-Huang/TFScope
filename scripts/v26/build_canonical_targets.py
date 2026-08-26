#!/usr/bin/env python
"""v26 Phase-1 step 2: parse the frozen annotation snapshot into canonical tables.

Reads ONLY data/annotations_v26/ (no network). Emits:

  data/processed/v26/accessions.parquet   acc -> canonical UniProt sequence + hash + gene + organism
  data/processed/v26/domains.parquet      every InterPro hit: acc, entry, db, type, start, end (1-based, UniProt coords)
  data/processed/v26/sifts_mappings.parquet  pdb, chain, entity, acc, unp_start/end, pdb resnum start/end, identity, coverage
  data/processed/v26/row_resolution.parquet  every v23 row -> primary_accession + how it was resolved
  results/v26/accession_ambiguity.csv        rows needing a gene-symbol fallback or left unresolved

Resolution policy (audit Finding A/I fix):
  * str_ rows  -> accession from SIFTS via (pdb_id, chain_id).  STRUCTURE-INDEPENDENT of DNA contacts.
  * seq_ rows  -> accession from the tf_pwm.parquet gene->uniprot seed, verified against the snapshot.
  * fallback   -> gene-symbol search result from gene_resolution.jsonl.gz, logged as ambiguous.
  * unresolved -> reported, never silently dropped.

  python scripts/v26/build_canonical_targets.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os

import numpy as np
import pandas as pd

SNAP = "data/annotations_v26"
OUTD = "data/processed/v26"
RESD = "results/v26"
V23 = "data/processed/tf_pwm_training_v23.parquet"
ORIG = "data/processed/tf_pwm.parquet"
STRUCT = "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"

PROGRESS_EVERY = 200


def _iter(path):
    """Yield (key, payload) for successful responses in a snapshot jsonl.gz."""
    if not os.path.exists(path):
        return
    with gzip.open(path, "rt") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == 200 and r.get("payload") is not None:
                yield r["key"], r["payload"]


def seq_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ accessions
def build_accessions():
    rows = []
    for acc, p in _iter(f"{SNAP}/uniprot.jsonl.gz"):
        seq = (p.get("sequence") or {}).get("value")
        if not seq:
            continue
        genes = [g.get("geneName", {}).get("value") for g in (p.get("genes") or [])]
        genes = [g for g in genes if g]
        rows.append({
            "accession": acc,
            "uniprot_id": p.get("uniProtkbId"),
            "primary_accession": p.get("primaryAccession", acc),
            "gene": (genes[0].upper() if genes else None),
            "all_genes": ";".join(g.upper() for g in genes),
            "organism_taxid": (p.get("organism") or {}).get("taxonId"),
            "organism": (p.get("organism") or {}).get("scientificName"),
            "reviewed": str(p.get("entryType", "")).startswith("UniProtKB reviewed"),
            "sequence": seq,
            "seq_len": len(seq),
            "sequence_hash": seq_hash(seq),
        })
    df = pd.DataFrame(rows).drop_duplicates("accession")
    print(f"  accessions: {len(df)}  reviewed={int(df.reviewed.sum())}", flush=True)
    return df


# --------------------------------------------------------------------- domains
def build_domains():
    """Flatten InterPro entry-all responses to one row per (acc, entry, fragment)."""
    rows = []
    n = 0
    for acc, p in _iter(f"{SNAP}/interpro.jsonl.gz"):
        n += 1
        for res in p.get("results", []):
            md = res.get("metadata") or {}
            for prot in res.get("proteins", []):
                plen = prot.get("protein_length")
                for loc in prot.get("entry_protein_locations") or []:
                    for frag in loc.get("fragments") or []:
                        s, e = frag.get("start"), frag.get("end")
                        if s is None or e is None:
                            continue
                        rows.append({
                            "accession": acc,
                            "entry_accession": md.get("accession"),
                            "entry_name": md.get("name"),
                            "source_database": md.get("source_database"),
                            "entry_type": md.get("type"),
                            "integrated": md.get("integrated"),
                            "protein_length": plen,
                            "start": int(s), "end": int(e),          # 1-based inclusive
                            "dc_status": frag.get("dc-status"),
                            "score": loc.get("score"),
                        })
        if n % PROGRESS_EVERY == 0:
            print(f"  interpro parsed {n} accessions, {len(rows)} fragments", flush=True)
    df = pd.DataFrame(rows)
    print(f"  domains: {len(df)} fragments over {df.accession.nunique()} accessions "
          f"({df.source_database.nunique()} databases)", flush=True)
    return df


# ------------------------------------------------------------------ sifts
def build_sifts():
    rows = []
    for pdb, p in _iter(f"{SNAP}/sifts.jsonl.gz"):
        for pdb_key, block in (p or {}).items():
            for acc, ent in (block.get("UniProt") or {}).items():
                for m in ent.get("mappings", []):
                    st, en = m.get("start") or {}, m.get("end") or {}
                    rows.append({
                        "pdb_id": pdb_key.upper(),
                        "chain_id": m.get("chain_id"),
                        "struct_asym_id": m.get("struct_asym_id"),
                        "entity_id": m.get("entity_id"),
                        "accession": acc,
                        "unp_start": m.get("unp_start"), "unp_end": m.get("unp_end"),
                        "pdb_auth_start": st.get("author_residue_number"),
                        "pdb_auth_end": en.get("author_residue_number"),
                        "pdb_resnum_start": st.get("residue_number"),
                        "pdb_resnum_end": en.get("residue_number"),
                        "identity": m.get("identity"), "coverage": m.get("coverage"),
                    })
    df = pd.DataFrame(rows)
    print(f"  sifts: {len(df)} mappings over {df.pdb_id.nunique()} PDB ids", flush=True)
    return df


# ------------------------------------------------------------- row resolution
def resolve_rows(acc_df, sifts_df):
    v = pd.read_parquet(V23)
    # pandas itertuples mangles leading-underscore names; rename before iterating
    v = v.rename(columns={"_set": "set_label"})
    st = pd.read_parquet(STRUCT).reset_index(drop=True)
    orig = pd.read_parquet(ORIG)[["gene_symbol", "uniprot_id"]].dropna()
    orig["G"] = orig.gene_symbol.astype(str).str.upper()
    gene2acc = {}
    for r in orig.itertuples():
        gene2acc.setdefault(r.G, str(r.uniprot_id))

    known_acc = set(acc_df.accession)
    # gene -> accession, from the snapshot, for disambiguating multi-accession chains
    acc_gene = {r.accession: (str(r.gene).upper() if r.gene else None)
                for r in acc_df.itertuples()}
    sif = {}
    for r in sifts_df.itertuples():
        sif.setdefault((r.pdb_id, str(r.chain_id)), []).append(
            (r.accession, r.coverage if r.coverage is not None else 0.0))

    gene_search = {}
    for g, p in _iter(f"{SNAP}/gene_resolution.jsonl.gz"):
        hits = [h.get("primaryAccession") for h in (p.get("results") or [])]
        gene_search[g] = [h for h in hits if h]

    out = []
    for r in v.itertuples():
        fn = str(r.filename)
        gene = str(r.gene_symbol).upper()
        acc = None
        how = None
        cands = []
        if fn.startswith("str_"):
            idx = int(fn.replace("str_", ""))
            if idx < len(st):
                pdb = str(st.iloc[idx]["pdb_id"]).upper()
                chain = str(st.iloc[idx]["chain_id"])
                cands = sif.get((pdb, chain), [])
                if cands:
                    # Prefer the accession whose UniProt gene matches this row's gene symbol.
                    # Without this, SIFTS returns CRYSTALLISATION FUSION PARTNERS as the chain's
                    # accession -- maltose-binding protein (MALE, 14 rows) and GFP were being
                    # picked instead of the TF, which then has no DBD. Gene symbol is provenance
                    # metadata used for preprocessing only, never a model input.
                    cands = sorted(
                        cands,
                        key=lambda x: (0 if acc_gene.get(x[0]) == gene else 1,
                                       -float(x[1] or 0)))
                    acc = cands[0][0]
                    how = ("sifts_pdb_chain_gene_matched"
                           if acc_gene.get(acc) == gene else "sifts_pdb_chain")
        if acc is None and gene in gene2acc and gene2acc[gene] in known_acc:
            acc, how = gene2acc[gene], "gene_seed_uniprot_id"
        if acc is None and gene in gene_search and gene_search[gene]:
            acc, how = gene_search[gene][0], "gene_symbol_search_FALLBACK"
        if acc is None and gene in gene2acc:
            acc, how = gene2acc[gene], "gene_seed_unverified"
        if acc is None:
            how = "UNRESOLVED"
        out.append({
            "filename": fn, "gene_symbol": gene, "set_label": str(r.set_label),
            "primary_accession": acc, "resolution_method": how,
            "n_sifts_candidates": len(cands),
            "sifts_candidates": ";".join(a for a, _ in cands[:4]),
        })
    df = pd.DataFrame(out)
    print("  resolution:", df.resolution_method.value_counts().to_dict(), flush=True)
    return df


def main():
    os.makedirs(OUTD, exist_ok=True)
    os.makedirs(RESD, exist_ok=True)

    print("[1/4] accessions ...", flush=True)
    acc_df = build_accessions()
    acc_df.to_parquet(f"{OUTD}/accessions.parquet", index=False)

    print("[2/4] domains ...", flush=True)
    dom = build_domains()
    dom.to_parquet(f"{OUTD}/domains.parquet", index=False)

    print("[3/4] sifts ...", flush=True)
    sif = build_sifts()
    sif.to_parquet(f"{OUTD}/sifts_mappings.parquet", index=False)

    print("[4/4] row resolution ...", flush=True)
    res = resolve_rows(acc_df, sif)
    res.to_parquet(f"{OUTD}/row_resolution.parquet", index=False)

    amb = res[res.resolution_method.isin(
        ["gene_symbol_search_FALLBACK", "gene_seed_unverified", "UNRESOLVED"])
        | (res.n_sifts_candidates > 1)]
    amb.to_csv(f"{RESD}/accession_ambiguity.csv", index=False)

    summary = {
        "accessions": int(len(acc_df)),
        "domain_fragments": int(len(dom)),
        "sifts_mappings": int(len(sif)),
        "rows": int(len(res)),
        "resolution_methods": res.resolution_method.value_counts().to_dict(),
        "ambiguous_or_fallback_rows": int(len(amb)),
        "unresolved_rows": int((res.resolution_method == "UNRESOLVED").sum()),
        "unresolved_genes": sorted(res[res.resolution_method == "UNRESOLVED"]
                                   .gene_symbol.unique().tolist()),
    }
    json.dump(summary, open(f"{RESD}/canonical_targets_summary.json", "w"), indent=2)

    print("\n=== summary ===")
    for k, v in summary.items():
        if k != "unresolved_genes":
            print(f"  {k}: {v}")
    print(f"  unresolved genes ({len(summary['unresolved_genes'])}): "
          f"{summary['unresolved_genes'][:20]}")
    print(f"  wrote {OUTD}/ and {RESD}/")


if __name__ == "__main__":
    main()
