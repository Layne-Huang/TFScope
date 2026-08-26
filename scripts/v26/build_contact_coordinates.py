#!/usr/bin/env python
"""v26 Phase-2 step 1: canonical protein-DNA contact coordinates.

Replaces the pre-shifted crop-local JSON indices that caused audit Findings C/D (v23 silently
clipped 1,034/10,232 contact links and emptied 122 PWM columns; v25flank relocated 680 onto flank
residues, so v24 and v25flank optimised different objectives).

Every contact is stored ONCE, in biological coordinates, with the full traceable chain:

    PDB author residue  ->  chain-local index  ->  UniProt index

Projection into any crop (core / flank20 / flank32) happens later in
project_contacts_to_crop.py, which MASKS out-of-crop contacts rather than moving or dropping them.

Output (resumable, one shard line per PDB):
  data/contacts_v26/contacts_raw.jsonl.gz     per-PDB residue-DNA contacts + chain metadata
  data/contacts_v26/contacts_canonical.parquet  flattened, UniProt-mapped
  results/v26/contact_mapping_report.csv        per-chain mapping coverage, never silently dropped

Coordinates are 1-based UniProt. Distances in Angstrom. Cutoff matches the legacy pipeline (4.5 A)
so the v24-compatible rebuild is comparable.

  python scripts/v26/build_contact_coordinates.py
  python scripts/v26/build_contact_coordinates.py --limit 20      # smoke test
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CIF = "data/raw/pdb_cif_cache"
OUTD = "data/contacts_v26"
RESD = "results/v26"
V26D = "data/processed/v26"

CUTOFF = 4.5
DUPLEX_CUTOFF = 10.0
PROGRESS_EVERY = 25

AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
          "MSE": "M", "SEC": "U", "PYL": "O"}
DNA1 = {"DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U"}


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


def chain_to_uniprot_map(chain_seq: str, unp_seq: str, al) -> dict[int, int]:
    """chain-local index (0-based) -> UniProt index (1-based). Unmapped positions are ABSENT."""
    if not chain_seq or not unp_seq:
        return {}
    try:
        aln = al.align(unp_seq, chain_seq)[0]
    except Exception:
        return {}
    out = {}
    for (u0, u1), (c0, c1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(u1 - u0):
            out[c0 + k] = u0 + k + 1          # UniProt is 1-based
    return out


def parse_structure(pdb_id: str):
    """Return (protein_chains, dna_chains) with author residue numbering preserved."""
    from Bio.PDB import MMCIFParser
    path = os.path.join(CIF, f"{pdb_id.lower()}.cif")
    if not os.path.exists(path):
        return None, None
    st = MMCIFParser(QUIET=True).get_structure(pdb_id, path)
    model = next(iter(st))
    prot, dna = {}, {}
    for ch in model:
        pres, dres = [], []
        for res in ch:
            name = res.get_resname().strip().upper()
            het, seqid, icode = res.id
            atoms = np.array([a.coord for a in res if a.element != "H"], dtype=np.float32)
            if not len(atoms):
                continue
            if name in AA3to1:
                pres.append({"auth": int(seqid), "icode": icode.strip(),
                             "aa": AA3to1[name], "coords": atoms})
            elif name in DNA1:
                dres.append({"auth": int(seqid), "icode": icode.strip(),
                             "base": DNA1[name], "coords": atoms})
        if pres:
            prot[ch.id] = pres
        if dres:
            dna[ch.id] = dres
    return prot, dna


def assign_duplexes(dna: dict) -> dict[str, str]:
    """Group DNA chains whose residues come within DUPLEX_CUTOFF into one duplex id."""
    ids = list(dna)
    if not ids:
        return {}
    cen = {c: np.concatenate([r["coords"] for r in dna[c]]) for c in ids}
    parent = {c: c for c in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = np.linalg.norm(cen[a][:, None, :] - cen[b][None, :, :], axis=-1)
            if d.min() <= DUPLEX_CUTOFF:
                parent[find(a)] = find(b)
    return {c: f"duplex_{sorted(ids).index(find(c))}" for c in ids}


def contacts_for_pdb(pdb_id: str):
    prot, dna = parse_structure(pdb_id)
    if prot is None:
        return {"pdb_id": pdb_id, "status": "cif_missing"}
    if not dna:
        return {"pdb_id": pdb_id, "status": "no_dna_chain",
                "protein_chains": sorted(prot or {})}
    duplex = assign_duplexes(dna)

    # index DNA residues along each chain for a stable base-pair coordinate
    dna_idx = {c: {r["auth"]: k for k, r in enumerate(sorted(dna[c], key=lambda x: x["auth"]))}
               for c in dna}

    out = {"pdb_id": pdb_id, "status": "ok", "duplex": duplex,
           "chains": {}, "contacts": []}
    for pc, pres in prot.items():
        pres = sorted(pres, key=lambda x: (x["auth"], x["icode"]))
        out["chains"][pc] = {
            "sequence": "".join(r["aa"] for r in pres),
            "auth_ids": [r["auth"] for r in pres],
            "icodes": [r["icode"] for r in pres],
            "n_residues": len(pres),
        }
        pxyz = [r["coords"] for r in pres]
        for dc, dres in dna.items():
            dxyz = np.concatenate([r["coords"] for r in dres])
            owner = np.concatenate([[k] * len(r["coords"]) for k, r in enumerate(dres)])
            for li, coords in enumerate(pxyz):
                d = np.linalg.norm(coords[:, None, :] - dxyz[None, :, :], axis=-1)
                if d.min() > CUTOFF:
                    continue
                # nearest DNA residue for this protein residue
                per_res = {}
                hit = np.where(d.min(axis=0) <= CUTOFF)[0]
                for j in hit:
                    k = int(owner[j])
                    v = float(d[:, j].min())
                    if k not in per_res or v < per_res[k]:
                        per_res[k] = v
                for k, mind in per_res.items():
                    r = dres[k]
                    out["contacts"].append({
                        "protein_chain": pc,
                        "chain_local_idx": li,
                        "pdb_auth_resid": pres[li]["auth"],
                        "pdb_icode": pres[li]["icode"],
                        "aa": pres[li]["aa"],
                        "dna_chain": dc,
                        "duplex_id": duplex.get(dc),
                        "dna_auth_resid": r["auth"],
                        "dna_chain_index": dna_idx[dc].get(r["auth"]),
                        "base": r["base"],
                        "min_distance": round(mind, 3),
                    })
    return out


def done_pdbs(path):
    if not os.path.exists(path):
        return set()
    got = set()
    with gzip.open(path, "rt") as fh:
        for line in fh:
            try:
                got.add(json.loads(line)["pdb_id"])
            except Exception:
                continue
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    os.makedirs(RESD, exist_ok=True)
    raw = f"{OUTD}/contacts_raw.jsonl.gz"

    core = pd.read_parquet(f"{V26D}/v26_core.parquet")
    pdbs = sorted({str(p).upper() for p in core.structure_id.dropna()})
    got = done_pdbs(raw)
    todo = [p for p in pdbs if p not in got]
    if a.limit:
        todo = todo[:a.limit]
    print(f"structures referenced by v26_core: {len(pdbs)}; cached {len(got)}; "
          f"to parse {len(todo)}", flush=True)

    t0 = time.time()
    counts = {"ok": 0, "no_dna_chain": 0, "cif_missing": 0, "error": 0}
    for i, p in enumerate(todo, 1):
        try:
            rec = contacts_for_pdb(p)
        except Exception as e:                                   # noqa: BLE001
            rec = {"pdb_id": p, "status": "error", "error": str(e)[:300]}
        counts[rec.get("status", "error")] = counts.get(rec.get("status", "error"), 0) + 1
        with gzip.open(raw, "at") as fh:
            fh.write(json.dumps(rec) + "\n")
        if i % PROGRESS_EVERY == 0 or i == len(todo):
            el = time.time() - t0
            rate = i / max(el, 1e-6)
            print(f"  {i}/{len(todo)}  {counts}  {rate:.2f}/s  "
                  f"elapsed={el/60:.1f}m eta={(len(todo)-i)/max(rate,1e-9)/60:.1f}m", flush=True)

    print(f"\nparse summary: {counts}", flush=True)
    print("run scripts/v26/map_contacts_to_uniprot.py next to add UniProt coordinates")


if __name__ == "__main__":
    main()
