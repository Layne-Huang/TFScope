#!/usr/bin/env python
"""Download RCSB crystal structures for the 130 test set entries.

For each entry in benchmark_no_val.json test split:
  - Downloads the full PDB structure (or uses cached copy)
  - Saves the specified protein chain + all DNA chains as {out_dir}/{test_entry_name}.pdb
  - DNA chains are identified by residues DA/DT/DG/DC

Output: /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/crystal_pdbs/
"""

import argparse
import json
import os
import sys

from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.PDBList import PDBList

DNA_RESIDUES = {"DA", "DT", "DG", "DC", "A", "T", "G", "C"}


class ProteinPlusDNASelect(Select):
    """Keep the specified protein chain and all DNA chains."""

    def __init__(self, protein_chain_id, dna_chain_ids):
        self.protein_chain_id = protein_chain_id
        self.dna_chain_ids = dna_chain_ids

    def accept_chain(self, chain):
        return chain.id == self.protein_chain_id or chain.id in self.dna_chain_ids


def get_dna_chains(model):
    """Return chain IDs that contain exclusively DNA residues."""
    dna_chains = set()
    for chain in model:
        residues = [r for r in chain if r.id[0] == " "]  # ATOM records only
        if residues and all(r.resname.strip() in DNA_RESIDUES for r in residues):
            dna_chains.add(chain.id)
    return dna_chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split",
        default="data/processed/splits/deeppbs_only/benchmark_no_val.json",
    )
    ap.add_argument(
        "--out-dir",
        default="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/crystal_pdbs",
    )
    ap.add_argument(
        "--cache-dir",
        default="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/pdb",
        help="Directory to cache raw downloaded PDB files",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    with open(args.split) as f:
        d = json.load(f)
    test_ids = d["test"]

    pdbl = PDBList(server="https://files.rcsb.org", verbose=False)
    parser = PDBParser(QUIET=True)
    io_out = PDBIO()

    # Cache: pdb_id → (raw_path, structure) to avoid re-parsing the same PDB
    downloaded = {}

    ok = 0
    failed = []
    for tid in test_ids:
        parts = tid.split("_")
        pdb_id = parts[0].lower()
        chain_id = parts[1]
        out_name = tid.replace(".txt", "") + ".pdb"
        out_path = os.path.join(args.out_dir, out_name)

        if os.path.exists(out_path):
            print(f"[skip]  {out_name}")
            ok += 1
            continue

        # Download (or reuse cached)
        if pdb_id not in downloaded:
            try:
                raw_path = pdbl.retrieve_pdb_file(
                    pdb_id, file_format="pdb", pdir=args.cache_dir
                )
                structure = parser.get_structure(pdb_id, raw_path)
                downloaded[pdb_id] = (raw_path, structure)
            except Exception as e:
                print(f"[ERROR] download {pdb_id}: {e}", file=sys.stderr)
                failed.append(tid)
                continue

        _, structure = downloaded[pdb_id]
        try:
            model = structure[0]
            chain_ids = [c.id for c in model]
            if chain_id not in chain_ids:
                print(f"[ERROR] chain {chain_id} not in {pdb_id} (has: {chain_ids})", file=sys.stderr)
                failed.append(tid)
                continue

            dna_chains = get_dna_chains(model)
            if not dna_chains:
                print(f"[WARN]  {pdb_id}: no DNA chains found — saving protein chain only")

            io_out.set_structure(structure)
            io_out.save(out_path, ProteinPlusDNASelect(chain_id, dna_chains))
            print(f"[ok]    {out_name}  (protein={chain_id}, dna={sorted(dna_chains)})")
            ok += 1
        except Exception as e:
            print(f"[ERROR] extract {tid}: {e}", file=sys.stderr)
            failed.append(tid)

    print(f"\nDone: {ok}/{len(test_ids)} written to {args.out_dir}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
