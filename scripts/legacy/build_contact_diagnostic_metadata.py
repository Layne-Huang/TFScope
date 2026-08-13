#!/usr/bin/env python
"""Build the metadata CSV for the frozen-ESM-2 DNA-contact residue diagnostic.

Source: TFlow protein-DNA co-crystal set (/data1/leihuang/TFlow/data/TF_split_index)
  - passed.txt : QC-passed complexes
  - dbd.json   : {pdb_file: [[DBD_name(s)], [[start,end], ...]]}  (PDB author numbering)
  - filenames  : {PDBID}_{model}_{proteinChain}_WITH_{dnaChains}.pdb

Emits columns: complex_id, pdb_file, protein_chain, dna_chains, sequence,
dbd_start, dbd_end, dbd_ranges, family. The protein sequence is read directly
from the modeled protein-chain ATOM records (no external annotation), so this
stays a purely structure+ESM diagnostic.
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser, is_aa
from Bio.Data.PDBData import protein_letters_3to1_extended as THREE2ONE

SRC = Path("/data1/leihuang/TFlow/data/TF_split_index")
OUT = Path("results/esm_contact_diagnostic")
NAME_RE = re.compile(r"^([0-9A-Za-z]{4})_(\d+)_([A-Za-z0-9])_WITH_([A-Za-z0-9]+)$")


def chain_sequence(structure, chain_id):
    """Ordered (seq, resid_list) for standard AAs in the protein chain."""
    model = next(structure.get_models())
    if chain_id not in {c.id for c in model.get_chains()}:
        return None, None
    chain = model[chain_id]
    seq, resids = [], []
    for res in chain.get_residues():
        if not is_aa(res, standard=False):
            continue
        het, resseq, icode = res.id
        if het.strip():  # skip HETATM / water
            continue
        one = THREE2ONE.get(res.get_resname().upper(), "X")
        if one in ("", None):
            one = "X"
        seq.append(one)
        resids.append(int(resseq))
    if not seq:
        return None, None
    return "".join(seq), resids


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dbd = json.load(open(SRC / "dbd.json"))
    passed = [l.strip() for l in open(SRC / "passed.txt") if l.strip()]
    parser = PDBParser(QUIET=True)

    rows, skipped = [], {"no_dbd_range": 0, "parse_name": 0, "no_seq": 0,
                         "no_dbd_residues": 0}
    for i, fp in enumerate(passed):
        if fp not in dbd:
            continue
        names, ranges = dbd[fp]
        ranges = [(int(a), int(b)) for a, b in ranges] if ranges else []
        if not ranges or not names:
            skipped["no_dbd_range"] += 1
            continue
        m = NAME_RE.match(os.path.basename(fp)[:-4])
        if not m:
            skipped["parse_name"] += 1
            continue
        pdbid, model_i, pchain, dna = m.groups()
        try:
            st = parser.get_structure(pdbid, fp)
        except Exception:
            skipped["no_seq"] += 1
            continue
        seq, resids = chain_sequence(st, pchain)
        if not seq:
            skipped["no_seq"] += 1
            continue
        # dbd.json ranges are 1-based positions into the modeled-chain sequence
        # (verified: 100% of ranges fit within [1, len(seq)]). Keep in-bounds.
        L = len(seq)
        keep = [(a, b) for a, b in ranges if 1 <= a <= b <= L]
        if not keep:
            skipped["no_dbd_residues"] += 1
            continue
        rows.append({
            "complex_id": os.path.basename(fp)[:-4],
            "pdb_file": fp,
            "protein_chain": pchain,
            "dna_chains": "".join(sorted(set(dna))),
            "sequence": seq,
            "dbd_start": min(a for a, _ in keep),
            "dbd_end": max(b for _, b in keep),
            "dbd_ranges": ";".join(f"{a}-{b}" for a, b in keep),
            "family": names[0],
            "pdb_id": pdbid,
        })
        if (i + 1) % 200 == 0:
            print(f"[meta] {i+1}/{len(passed)} processed, {len(rows)} kept",
                  flush=True)

    df = pd.DataFrame(rows)
    out_csv = OUT / "contact_diagnostic_metadata.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[meta] wrote {len(df)} complexes -> {out_csv}")
    print("[meta] skipped:", skipped)
    print("[meta] unique PDB ids:", df["pdb_id"].nunique(),
          "families:", df["family"].nunique())
    print(df["family"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
