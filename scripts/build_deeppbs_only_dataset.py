#!/usr/bin/env python
"""Build a TFScope-format dataset from the DeepPBS training + blind benchmark data only.

For each NPZ entry in train{0-4}.txt union + id.txt:
  - Protein sequence : extracted from the crystal structure PDB chain (RCSB API)
  - PWM              : from NPZ Y_pwm[0][pwm_mask[0]].T  → (4, L) float32
  - Gene / source    : from HOCOMOCO filename alias, or JASPAR REST API
  - DBD              : entire crystal chain (dbd_start=0, dbd_end=len(seq))
  - Family           : InterPro via UniProt ID returned by JASPAR API / gene lookup

Outputs:
  data/processed/tf_pwm_deeppbs_only.parquet
  data/processed/splits/deeppbs_only/benchmark.json
    train = entries from train{0-4}.txt union (all folds)
    val   = 10 % of train, stratified by family
    test  = entries from id.txt (blind benchmark)

Usage:
    python scripts/build_deeppbs_only_dataset.py
    python scripts/build_deeppbs_only_dataset.py --dry-run   # show counts only
"""

import argparse
import json
import os
import random
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from map_tf_annotations import (
    make_session,
    resolve_uniprot,
    fetch_interpro_domains,
    assign_family_label,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
FOLD_DIR    = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/deeppbsmar24/run/folds"
NPZ_DIR     = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/deeppbsmar24/data/assembly2024"
PARQUET_OUT = "data/processed/tf_pwm_deeppbs_only.parquet"
SPLIT_OUT   = "data/processed/splits/deeppbs_only/benchmark.json"
PDB_CACHE   = "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/pdb"

ALIASES = {
    "GCR":"NR3C1","TF65":"RELA","BMAL1":"ARNTL","PO2F1":"POU2F1","PO5F1":"POU5F1",
    "NFAC1":"NFATC1","NFAC2":"NFATC2","KAISO":"ZBTB33","SUH":"RBPJ","HXA13":"HOXA13",
    "HXA9":"HOXA9","HXB13":"HOXB13","NDF1":"NEUROD1","PRGR":"PGR","ZBT7A":"ZBTB7A",
    "TFE2":"TCF4","ITF2":"TCF4","BRAC":"T","STF1":"NR5A1","NKX25":"NKX2-5",
    "PO3F1":"POU3F1","COE1":"EBF1",
}

# JASPAR cache: ma_versioned → API response (avoid redundant calls)
_jaspar_cache: dict[str, dict | None] = {}


# ── PDB helpers ────────────────────────────────────────────────────────────────

def fetch_cif(pdb_id: str, session: requests.Session) -> str:
    os.makedirs(PDB_CACHE, exist_ok=True)
    path = os.path.join(PDB_CACHE, f"{pdb_id.lower()}.cif")
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

def fetch_jaspar(ma_versioned: str, session: requests.Session) -> dict | None:
    if ma_versioned in _jaspar_cache:
        return _jaspar_cache[ma_versioned]

    url = f"https://jaspar.elixir.no/api/v1/matrix/{ma_versioned}/?format=json"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            ma_base = ma_versioned.split(".")[0]
            url2 = f"https://jaspar.elixir.no/api/v1/matrix/?base_id={ma_base}&format=json"
            resp = session.get(url2, timeout=30)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                _jaspar_cache[ma_versioned] = None
                return None
            results.sort(key=lambda x: int(x.get("version", 0)), reverse=True)
            entry = results[0]
        else:
            resp.raise_for_status()
            entry = resp.json()

        pfm = entry.get("pfm", entry.get("pfms", None))
        if pfm is None:
            _jaspar_cache[ma_versioned] = None
            return None

        if isinstance(pfm, dict):
            counts = np.array([pfm["A"], pfm["C"], pfm["G"], pfm["T"]], dtype=np.float32)
        else:
            counts = np.array(pfm, dtype=np.float32)

        col_sums = counts.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        pwm = (counts / col_sums).astype(np.float32)
        if pwm.shape[1] > 20:
            pwm = pwm[:, :20]

        result = {
            "gene_name":   entry.get("name", ""),
            "uniprot_ids": entry.get("uniprot_ids", []),
            "species":     entry.get("species", []),
            "pwm":         pwm,
            "matrix_id":  entry.get("matrix_id", ma_versioned),
        }
        _jaspar_cache[ma_versioned] = result
        time.sleep(0.2)
        return result
    except Exception as e:
        _jaspar_cache[ma_versioned] = None
        return None


# ── NPZ PWM extraction ─────────────────────────────────────────────────────────

def extract_npz_pwm(npz_path: str) -> np.ndarray | None:
    try:
        d = np.load(npz_path, allow_pickle=True)
        y    = d["Y_pwm"]       # (2, L, 4)  ACGT
        mask = d["pwm_mask"]    # (2, L)
        fwd  = y[0][mask[0]]    # (n_valid, 4)
        if len(fwd) < 4:
            return None
        pwm = fwd.T.astype(np.float32)  # (4, L)
        if pwm.shape[1] > 20:
            pwm = pwm[:, :20]
        return pwm
    except Exception:
        return None


# ── Entry parsing ──────────────────────────────────────────────────────────────

def parse_entry(entry: str):
    """Returns (kind, gene_or_ma_base, ma_versioned, pdb_id, chain, species_str).
    kind = 'jaspar' | 'hocomoco' | 'unknown'
    """
    base   = os.path.basename(entry).replace(".npz", "")
    parts  = base.split("_")
    pdb_id = parts[0] if parts else ""
    chain  = parts[1] if len(parts) > 1 else ""

    m = re.search(r"(MA\d+)\.(\d+)\.jaspar$", base)
    if m:
        ma_base = m.group(1)
        return ("jaspar", ma_base, f"{ma_base}.{m.group(2)}", pdb_id, chain, "")

    m = re.search(r"_([A-Z0-9][A-Z0-9\-]*)_(HUMAN|MOUSE)\.H11MO", base)
    if m:
        gene = ALIASES.get(m.group(1), m.group(1))
        return ("hocomoco", gene, "", pdb_id, chain, m.group(2))

    return ("unknown", base, "", pdb_id, chain, "")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    session = make_session()

    # Load fold lists
    def load_set(fp):
        with open(fp) as f:
            return {l.strip() for l in f if l.strip()}

    blind_entries  = load_set(os.path.join(FOLD_DIR, "id.txt"))
    train_entries  = set()
    for i in range(5):
        train_entries.update(load_set(os.path.join(FOLD_DIR, f"train{i}.txt")))
    train_entries -= blind_entries          # ensure disjoint (already 0 overlap)

    all_entries = sorted(train_entries | blind_entries)
    print(f"Train entries : {len(train_entries)}")
    print(f"Blind entries : {len(blind_entries)}")
    print(f"Total entries : {len(all_entries)}")

    unique_pdbs = {parse_entry(e)[3] for e in all_entries}
    print(f"Unique PDB IDs: {len(unique_pdbs)}")

    if args.dry_run:
        return

    # ── Process every entry ───────────────────────────────────────────────────
    rows    = []
    failed  = []
    seen_fn = set()   # guard against filename collision

    for idx, entry in enumerate(all_entries):
        kind, ident, ma_ver, pdb_id, chain, species_str = parse_entry(entry)
        npz_path = os.path.join(NPZ_DIR, entry)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(all_entries)}] processed so far, {len(rows)} ok, {len(failed)} failed")

        # ── 1. Extract protein sequence from PDB chain ────────────────────────
        try:
            cif_path = fetch_cif(pdb_id, session)
            seq = extract_chain_sequence(cif_path, chain)
        except Exception as e:
            failed.append((entry, f"PDB: {e}"))
            continue

        if not seq or len(seq) < 5:
            failed.append((entry, f"seq too short ({len(seq)})"))
            continue

        # ── 2. Extract PWM from NPZ ───────────────────────────────────────────
        pwm = extract_npz_pwm(npz_path)
        if pwm is None:
            failed.append((entry, "PWM extraction failed"))
            continue

        motif_length = pwm.shape[1]

        # ── 3. Determine gene name, source, organism, uniprot ────────────────
        if kind == "hocomoco":
            gene_name  = ident
            source     = "HOCOMOCO"
            source_id  = os.path.basename(entry).replace(".npz", "").split(f"{pdb_id}_{chain}_")[1]
            organism   = "Homo sapiens" if species_str == "HUMAN" else "Mus musculus"
            uniprot_id = ""

        elif kind == "jaspar":
            jdata = fetch_jaspar(ma_ver, session)
            gene_name  = (jdata["gene_name"] if jdata else "") or ma_ver
            source     = "JASPAR"
            source_id  = (jdata["matrix_id"] if jdata else ma_ver)
            uniprot_id = (jdata["uniprot_ids"][0] if jdata and jdata["uniprot_ids"] else "")
            sp_list    = (jdata["species"] if jdata else [])
            organism   = "Homo sapiens"
            for sp in sp_list:
                if "musculus" in sp.get("name", "").lower():
                    organism = "Mus musculus"
                    break
                if "sapiens" in sp.get("name", "").lower():
                    organism = "Homo sapiens"
                    break

        else:
            failed.append((entry, "unknown format"))
            continue

        # ── 4. Family from InterPro ───────────────────────────────────────────
        family_id, family_name = 9, "Other"
        if not uniprot_id and kind == "hocomoco":
            try:
                uni = resolve_uniprot(session, gene_name, "Homo sapiens")
                if uni:
                    uniprot_id = uni["accession"]
                time.sleep(0.2)
            except Exception:
                pass

        if uniprot_id:
            try:
                domains = fetch_interpro_domains(session, uniprot_id)
                family_id, family_name = assign_family_label(domains)
                time.sleep(0.2)
            except Exception:
                pass

        # ── 5. Build unique filename ──────────────────────────────────────────
        base_fn = f"{pdb_id}_{chain}_{gene_name}.{source_id}.txt"
        fn = base_fn
        suffix = 0
        while fn in seen_fn:
            suffix += 1
            fn = f"{pdb_id}_{chain}_{gene_name}.{source_id}.v{suffix}.txt"
        seen_fn.add(fn)

        rows.append({
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
            "filename":      fn,
            "_entry":        entry,   # temp: to build split
        })

    print(f"\nTotal processed: {len(rows)} ok, {len(failed)} failed")
    if failed:
        print("  Failed entries:")
        for e, reason in failed[:20]:
            print(f"    {e}: {reason}")

    if not rows:
        print("No rows built — aborting.")
        return

    # ── Save parquet (drop temp _entry column) ────────────────────────────────
    df = pd.DataFrame(rows)
    entry_col = df.pop("_entry")   # keep aside for split building
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"\nWrote {len(df)} rows to {PARQUET_OUT}")

    # ── Build split ────────────────────────────────────────────────────────────
    train_fn  = [rows[i]["filename"] for i, e in enumerate(entry_col) if e in train_entries]
    test_fn   = [rows[i]["filename"] for i, e in enumerate(entry_col) if e in blind_entries]

    # stratified 10% val from train
    family_groups: dict[str, list] = {}
    fn_to_fam = dict(zip(df["filename"], df["family_name"]))
    for fn in train_fn:
        fam = fn_to_fam.get(fn, "Other")
        family_groups.setdefault(fam, []).append(fn)

    val_fn, tr_fn = [], []
    random.seed(42)
    for fam, fns in family_groups.items():
        random.shuffle(fns)
        n_val = max(1, int(len(fns) * 0.10))
        val_fn.extend(fns[:n_val])
        tr_fn.extend(fns[n_val:])

    split = {
        "train": sorted(tr_fn),
        "val":   sorted(val_fn),
        "test":  sorted(test_fn),
        "metadata": {
            "description": "DeepPBS-only dataset: sequences from PDB chains, PWMs from NPZ Y_pwm",
            "n_train":  len(tr_fn),
            "n_val":    len(val_fn),
            "n_test":   len(test_fn),
            "n_failed": len(failed),
        },
    }
    os.makedirs(os.path.dirname(SPLIT_OUT), exist_ok=True)
    with open(SPLIT_OUT, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Wrote split: train={len(tr_fn)}, val={len(val_fn)}, test={len(test_fn)}")
    print(f"Split saved to {SPLIT_OUT}")


if __name__ == "__main__":
    main()
