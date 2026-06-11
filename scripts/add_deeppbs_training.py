#!/usr/bin/env python
"""Add DeepPBS CV training TFs that are absent from tf_pwm.parquet.

For each missing entry:
  - Protein sequence: extracted from the PDB/mmCIF crystal structure chain
    (PDB ID + chain from NPZ filename, downloaded from RCSB)
  - PWM: from NPZ Y_pwm (HOCOMOCO) or JASPAR REST API (JASPAR entries)
  - DBD: the entire crystal chain = the binding domain; dbd_start=0, dbd_end=len(seq)
  - Family: from InterPro via the UniProt ID returned by JASPAR API / resolve_uniprot

Outputs:
  data/processed/tf_pwm_augmented.parquet
  data/processed/splits/deeppbs_blind/benchmark_augmented.json

Usage:
    python scripts/add_deeppbs_training.py
    python scripts/add_deeppbs_training.py --dry-run
"""

import argparse
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from map_tf_annotations import (
    make_session,
    resolve_uniprot,
    fetch_interpro_domains,
    assign_family_label,
    FAMILY_NAMES,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
FOLD_DIR   = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/deeppbsmar24/run/folds"
NPZ_DIR    = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/deeppbsmar24/data/assembly2024"
PARQUET_IN = "data/processed/tf_pwm.parquet"
PARQUET_OUT= "data/processed/tf_pwm_augmented.parquet"
SPLIT_IN   = "data/processed/splits/deeppbs_blind/benchmark.json"
SPLIT_OUT  = "data/processed/splits/deeppbs_blind/benchmark_augmented.json"
PDB_CACHE  = "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/pdb"

ALIASES = {
    "GCR":"NR3C1","TF65":"RELA","BMAL1":"ARNTL","PO2F1":"POU2F1","PO5F1":"POU5F1",
    "NFAC1":"NFATC1","NFAC2":"NFATC2","KAISO":"ZBTB33","SUH":"RBPJ","HXA13":"HOXA13",
    "HXA9":"HOXA9","HXB13":"HOXB13","NDF1":"NEUROD1","PRGR":"PGR","ZBT7A":"ZBTB7A",
    "TFE2":"TCF4","ITF2":"TCF4","BRAC":"T","STF1":"NR5A1","NKX25":"NKX2-5",
    "PO3F1":"POU3F1","COE1":"EBF1",
}


# ── PDB helpers ────────────────────────────────────────────────────────────────

def fetch_cif(pdb_id: str, cache_dir: str, session: requests.Session) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id.lower()}.cif")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id.lower()}.cif"
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        with open(path, "w") as f:
            f.write(resp.text)
    return path


def extract_chain_sequence(cif_path: str, chain_id: str) -> str:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import PPBuilder
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("x", cif_path)
    ppb = PPBuilder()
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                seq = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(chain))
                if seq:
                    return seq
    return ""


# ── JASPAR helpers ─────────────────────────────────────────────────────────────

def fetch_jaspar_matrix(ma_id: str, session: requests.Session) -> dict | None:
    """Fetch matrix + metadata from JASPAR REST API. Returns dict with keys:
    gene_name, uniprot_ids, species, pfms (4×L raw counts), version.
    """
    # Try version-agnostic search first: get the latest version
    url = f"https://jaspar.elixir.no/api/v1/matrix/{ma_id}/?format=json"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            # Try without version (base ID search)
            url2 = f"https://jaspar.elixir.no/api/v1/matrix/?base_id={ma_id}&format=json"
            resp = session.get(url2, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None
            # Pick highest version
            results.sort(key=lambda x: int(x.get("version", 0)), reverse=True)
            entry = results[0]
        else:
            resp.raise_for_status()
            entry = resp.json()

        pfms = entry.get("pfm", entry.get("pfms", None))
        if pfms is None:
            return None

        # pfms may be dict {A:[], C:[], G:[], T:[]} or list of 4 lists
        if isinstance(pfms, dict):
            counts = np.array([pfms["A"], pfms["C"], pfms["G"], pfms["T"]], dtype=np.float32)
        else:
            counts = np.array(pfms, dtype=np.float32)  # shape (4, L)

        return {
            "gene_name": entry.get("name", ""),
            "uniprot_ids": entry.get("uniprot_ids", []),
            "species": entry.get("species", []),
            "counts": counts,   # (4, L) raw counts
            "version": str(entry.get("version", "1")),
            "matrix_id": entry.get("matrix_id", f"{ma_id}.1"),
        }
    except Exception as e:
        print(f"  [JASPAR] fetch failed for {ma_id}: {e}")
        return None


def counts_to_pwm(counts: np.ndarray) -> np.ndarray:
    """Normalize raw counts to column-wise probabilities. Shape (4, L)."""
    col_sums = counts.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    pwm = counts / col_sums
    # Clip to max_motif_length=20
    if pwm.shape[1] > 20:
        pwm = pwm[:, :20]
    return pwm.astype(np.float32)


# ── NPZ PWM extraction ─────────────────────────────────────────────────────────

def extract_npz_pwm(npz_path: str) -> np.ndarray | None:
    """Extract forward-strand PWM from NPZ. Returns (4, L) float32 or None."""
    try:
        d = np.load(npz_path, allow_pickle=True)
        y = d["Y_pwm"]       # (2, L, 4)
        mask = d["pwm_mask"] # (2, L)
        fwd = y[0][mask[0]]  # (n_valid, 4)  ACGT
        if len(fwd) == 0:
            return None
        pwm = fwd.T.astype(np.float32)  # (4, L)
        if pwm.shape[1] > 20:
            pwm = pwm[:, :20]
        return pwm
    except Exception as e:
        print(f"  [NPZ] failed to extract PWM from {npz_path}: {e}")
        return None


# ── Entry parsing ──────────────────────────────────────────────────────────────

def parse_npz_entry(entry: str):
    """Parse NPZ filename. Returns (kind, ma_or_gene, pdb_id, chain, species, ma_versioned).
    kind = 'jaspar' | 'hocomoco' | 'unknown'
    ma_versioned: e.g. 'MA0094.2' for JASPAR (empty for HOCOMOCO)
    """
    base = os.path.basename(entry).replace(".npz", "")
    parts = base.split("_")
    if len(parts) < 2:
        return ("unknown", base, "", "", "", "")
    pdb_id = parts[0]
    chain  = parts[1]

    # JASPAR: {pdb}_{chain}_{MA####.V}.jaspar
    m = re.search(r"(MA\d+)\.(\d+)\.jaspar$", base)
    if m:
        ma_base = m.group(1)
        ma_ver  = f"{ma_base}.{m.group(2)}"
        return ("jaspar", ma_base, pdb_id, chain, "", ma_ver)

    # HOCOMOCO: {pdb}_{chain}_{GENE}_{SPECIES}.H11MO.*
    m = re.search(r"_([A-Z0-9][A-Z0-9\-]*)_(HUMAN|MOUSE)\.H11MO", base)
    if m:
        gene    = ALIASES.get(m.group(1), m.group(1))
        species = m.group(2)
        return ("hocomoco", gene, pdb_id, chain, species, "")

    return ("unknown", base, pdb_id, chain, "", "")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be added without writing files")
    args = parser.parse_args()

    session = make_session()
    df = pd.read_parquet(PARQUET_IN)
    print(f"Loaded {len(df)} rows from {PARQUET_IN}")

    # Build lookup sets
    tfscope_genes   = set(df["gene_symbol"].str.upper())
    tfscope_jaspar  = set(
        df[df["source"] == "JASPAR"]["source_id"]
        .str.extract(r"(MA\d+)", expand=False)
        .dropna()
        .str.upper()
    )

    # Load blind benchmark (test set — must NOT be added to train)
    with open(SPLIT_IN) as f:
        split_orig = json.load(f)
    test_files   = set(split_orig["test"])
    test_genes   = set(df[df["filename"].isin(test_files)]["gene_symbol"].str.upper())
    print(f"Blind test genes (protected): {len(test_genes)}")

    # Collect all CV training entries (union of train0-4, excluding blind benchmark)
    with open(os.path.join(FOLD_DIR, "id.txt")) as f:
        blind_entries = {l.strip() for l in f if l.strip()}

    train_union: set[str] = set()
    for i in range(5):
        with open(os.path.join(FOLD_DIR, f"train{i}.txt")) as f:
            train_union.update(l.strip() for l in f if l.strip())
    train_union -= blind_entries
    print(f"CV training entries (excl. blind): {len(train_union)}")

    # Deduplicate: one representative entry per (kind, identifier)
    seen_ids: dict[tuple, str] = {}  # (kind, id) → first entry seen
    seen_ma_ver: dict[str, str] = {}  # ma_base → ma_versioned (e.g. MA0094 → MA0094.2)
    for entry in sorted(train_union):
        kind, ident, pdb_id, chain, species, ma_ver = parse_npz_entry(entry)
        key = (kind, ident.upper())
        if key not in seen_ids:
            seen_ids[key] = entry
            if ma_ver:
                seen_ma_ver[ident.upper()] = ma_ver

    print(f"Unique TF identities in CV train: {len(seen_ids)}")

    # Identify which are missing from TFScope
    to_add: list[tuple] = []
    for (kind, ident), entry in seen_ids.items():
        if kind == "hocomoco":
            if ident in tfscope_genes:
                continue  # already present
            if ident in test_genes:
                print(f"  [SKIP] {ident} is in blind test (HOCOMOCO)")
                continue
            to_add.append((kind, ident, entry))
        elif kind == "jaspar":
            if ident in tfscope_jaspar:
                continue
            # Check if gene for this JASPAR entry is in test (we'll find out during fetch)
            to_add.append((kind, ident, entry))

    print(f"\nEntries to add: {len(to_add)}")

    if args.dry_run:
        for kind, ident, entry in to_add:
            print(f"  [{kind.upper()}] {ident}  ←  {entry}")
        return

    # ── Process each missing entry ────────────────────────────────────────────
    new_rows = []
    skipped  = []
    failed   = []

    for idx, (kind, ident, entry) in enumerate(to_add):
        _, _, pdb_id, chain, species, _ = parse_npz_entry(entry)
        print(f"\n[{idx+1}/{len(to_add)}] {kind.upper()} {ident}  (pdb={pdb_id} chain={chain})")

        # ── 1. Get protein sequence from PDB ─────────────────────────────────
        try:
            cif_path = fetch_cif(pdb_id, PDB_CACHE, session)
            seq = extract_chain_sequence(cif_path, chain)
        except Exception as e:
            print(f"  [FAIL] PDB fetch/extract failed: {e}")
            failed.append((kind, ident, f"PDB: {e}"))
            continue

        if not seq or len(seq) < 5:
            print(f"  [FAIL] sequence too short ({len(seq)} aa)")
            failed.append((kind, ident, "seq too short"))
            continue

        print(f"  seq len={len(seq)}: {seq[:30]}...")

        # ── 2. Get PWM ────────────────────────────────────────────────────────
        if kind == "hocomoco":
            npz_path = os.path.join(NPZ_DIR, entry)
            pwm = extract_npz_pwm(npz_path)
            if pwm is None:
                print(f"  [FAIL] NPZ PWM extraction failed")
                failed.append((kind, ident, "NPZ PWM failed"))
                continue
            source     = "HOCOMOCO"
            source_id  = re.search(r"_([A-Z0-9\-]+)_(HUMAN|MOUSE)\.H11MO\.\S+", entry)
            source_id  = source_id.group(1) + "_" + source_id.group(2) + ".H11MO" if source_id else ident
            gene_name  = ident
            organism   = "Homo sapiens" if species == "HUMAN" else "Mus musculus"
            uniprot_id = ""

        else:  # jaspar
            # Use versioned MA ID (e.g. MA0094.2) for direct API lookup when available
            ma_versioned = seen_ma_ver.get(ident.upper(), ident)
            jdata = fetch_jaspar_matrix(ma_versioned, session)
            if jdata is None:
                print(f"  [FAIL] JASPAR fetch failed")
                failed.append((kind, ident, "JASPAR fetch failed"))
                continue

            pwm       = counts_to_pwm(jdata["counts"])
            gene_name = jdata["gene_name"] or ident
            source    = "JASPAR"
            source_id = jdata["matrix_id"]
            uniprot_ids = jdata["uniprot_ids"]
            uniprot_id  = uniprot_ids[0] if uniprot_ids else ""

            # Check if this JASPAR gene is in the blind test
            if gene_name.upper() in test_genes:
                print(f"  [SKIP] {gene_name} is in blind test (JASPAR)")
                skipped.append((kind, ident, "in blind test"))
                continue

            # Determine organism from JASPAR species list
            species_list = jdata.get("species", [])
            organism = "Homo sapiens"
            for sp in species_list:
                name = sp.get("name", "")
                if "sapiens" in name.lower():
                    organism = "Homo sapiens"
                    break
                elif "musculus" in name.lower():
                    organism = "Mus musculus"

        motif_length = pwm.shape[1]
        if motif_length < 4:
            print(f"  [FAIL] PWM too short ({motif_length} positions)")
            failed.append((kind, ident, "PWM too short"))
            continue

        # ── 3. Get family from InterPro ───────────────────────────────────────
        family_id, family_name = 9, "Other"
        if not uniprot_id and kind == "hocomoco":
            # Try to get UniProt ID from gene name (just for family lookup)
            uni = resolve_uniprot(session, gene_name, "Homo sapiens")
            if uni:
                uniprot_id = uni["accession"]
            time.sleep(0.3)

        if uniprot_id:
            try:
                domains = fetch_interpro_domains(session, uniprot_id)
                family_id, family_name = assign_family_label(domains)
                time.sleep(0.3)
            except Exception as e:
                print(f"  [WARN] InterPro failed for {uniprot_id}: {e}")

        print(f"  family={family_name}, motif_len={motif_length}, uniprot={uniprot_id}")

        # ── 4. Build row ──────────────────────────────────────────────────────
        filename = f"{gene_name}.{source_id}.DEEPPBS.txt"
        # Ensure filename uniqueness
        existing = set(df["filename"])
        if filename in existing or any(r["filename"] == filename for r in new_rows):
            filename = f"{gene_name}.{source_id}.{pdb_id}{chain}.DEEPPBS.txt"

        row = {
            "tf_name":       gene_name,
            "uniprot_id":    uniprot_id,
            "gene_symbol":   gene_name,
            "organism":      organism,
            "sequence":      seq,
            "seq_length":    len(seq),
            "dbd_start":     0,
            "dbd_end":       len(seq),
            "dbd_count":     1,
            "family_id":     family_id,
            "family_name":   family_name,
            "motif_length":  motif_length,
            "pwm":           pwm.tobytes(),
            "source":        source,
            "source_id":     source_id,
            "assay_type":    "",
            "quality_grade": "",
            "filename":      filename,
        }
        new_rows.append(row)
        print(f"  ✓ added as {filename}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  New rows:  {len(new_rows)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        for kind, ident, reason in failed:
            print(f"    [{kind}] {ident}: {reason}")

    if not new_rows:
        print("Nothing to add.")
        return

    # ── Write augmented parquet ────────────────────────────────────────────────
    df_new  = pd.DataFrame(new_rows)
    df_aug  = pd.concat([df, df_new], ignore_index=True)
    df_aug.to_parquet(PARQUET_OUT, index=False)
    print(f"\nWrote {len(df_aug)} rows ({len(new_rows)} new) to {PARQUET_OUT}")

    # ── Write augmented split ──────────────────────────────────────────────────
    # test: same 129 blind benchmark filenames (unchanged)
    # train+val: everything NOT in test
    all_filenames = set(df_aug["filename"])
    non_test = sorted(all_filenames - test_files)

    # stratify val by family: 10% per family
    df_non_test = df_aug[df_aug["filename"].isin(non_test)].copy()
    val_files, train_files = [], []
    for fam in df_non_test["family_name"].unique():
        group = df_non_test[df_non_test["family_name"] == fam]["filename"].tolist()
        n_val = max(1, int(len(group) * 0.10))
        import random; random.seed(42); random.shuffle(group)
        val_files.extend(group[:n_val])
        train_files.extend(group[n_val:])

    split_aug = {
        "test":  sorted(test_files),
        "val":   sorted(val_files),
        "train": sorted(train_files),
        "metadata": {
            "description": "DeepPBS blind benchmark augmented with DeepPBS CV training TFs",
            "n_test":  len(test_files),
            "n_val":   len(val_files),
            "n_train": len(train_files),
            "n_new_rows": len(new_rows),
        },
    }
    os.makedirs(os.path.dirname(SPLIT_OUT), exist_ok=True)
    with open(SPLIT_OUT, "w") as f:
        json.dump(split_aug, f, indent=2)
    print(f"Wrote split to {SPLIT_OUT}")
    print(f"  train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")


if __name__ == "__main__":
    main()
