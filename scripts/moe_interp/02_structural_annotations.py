#!/usr/bin/env python
"""Per-residue structural-role annotations for the co-crystal subset of the v23 table.

Panel B of the MoE biochemical-interpretability figure asks whether an expert's residue
*chemistry* signature corresponds to a residue *role*.  That needs real structural
labels, so we go back to the deposited co-crystals.

Mapping.  v23 structure rows (`str_*`) carry only a cropped protein sequence, not a PDB
pointer.  We recover it by (i) generating candidate (PDB, chain) pairs whose deposited
sequence contains the crop verbatim, (ii) parsing that CIF and locating the crop inside
the chain's *observed* residue sequence, and (iii) keeping the candidate whose
recomputed DNA-contact set best matches the v23 supervision set for that row.  Step
(iii) makes the mapping self-validating: a wrong structure does not reproduce the
stored contacts.

Two things about step (iii).  First, v23's stored indices mark residues that touch DNA
at all, base or backbone, so we score against that union rather than base atoms alone.
Second, for a substantial fraction of rows v23's indices sit in a *shifted* frame
relative to the crop (a rigid offset of up to a few residues), so the score maximises
over rigid shifts and records the winning shift in `v23_index_shift`.  The annotation
itself is always emitted in crop coordinates -- position i of the annotation is the
structural residue whose one-letter code is crop[i], which is exactly the residue the
router saw at token i -- so the shift affects only which candidate structure is picked,
never the alignment between routing and structure.

Annotations (all distances between heavy atoms, hydrogens ignored):

  base_contact       >=1 heavy atom within BASE_CUT of a DNA *base* atom
  backbone_contact   >=1 heavy atom within BASE_CUT of a DNA sugar/phosphate atom
  major_groove       >=1 heavy atom within BASE_CUT of a major-groove base edge atom
  minor_groove       >=1 heavy atom within BASE_CUT of a minor-groove base edge atom
  zn_coord           >=1 candidate ligand atom (S/N/O) within ZN_CUT of a Zn ion
  buried_core        relative SASA of the isolated chain < RSA_BURIED
  helix / strand / loop   pydssp three-state assignment (H / E / -)
  ppi                >=1 heavy atom within BASE_CUT of a different protein chain
  exposed_nondna     rel SASA >= RSA_BURIED and no DNA contact and no protein interface

Relative SASA uses the Tien et al. 2013 theoretical maximum ASA.

Usage:
  python scripts/moe_interp/02_structural_annotations.py [--outdir results/moe_biochem]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import warnings

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, ShrakeRupley
from Bio.PDB.Polypeptide import protein_letters_3to1
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

RAW_CONTACTS = "data/contacts_v26/contacts_raw.jsonl.gz"
CIF_DIR = "data/raw/pdb_cif_cache"
V23_CONTACTS = "data/contact_maps/contact_targets_v23.json"

# ── annotation criteria (documented here, mirrored in the figure Methods) ────────
BASE_CUT = 4.0      # Å, heavy-atom to heavy-atom
ZN_CUT = 2.8        # Å, ligand heavy atom to Zn
RSA_BURIED = 0.20   # relative SASA below which a residue counts as buried core
SHIFT_TOL = 8       # rigid index shift allowed when scoring a candidate mapping against
                    # the v23 supervision set (see the module docstring)

# DNA sugar + phosphate atoms; everything else in a nucleotide is a base atom
SUGAR_PHOS = {"P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "O4'",
              "C3'", "O3'", "C2'", "C1'", "O2'", "O1P", "O2P", "O3P"}
MAJOR_EDGE = {"N7", "O6", "N6", "C5", "C7", "C5M", "O4", "N4"}
MINOR_EDGE = {"N3", "N2", "O2", "C2"}
DNA_RES = {"DA", "DC", "DG", "DT", "DU", "DI"}
ZN_LIGAND_ELEMENTS = {"S", "N", "O"}

# Tien et al. 2013, PLoS ONE 8:e80635 - theoretical maximum accessible surface area
MAX_ASA = {"A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "E": 223, "Q": 225,
           "G": 104, "H": 224, "I": 197, "L": 201, "K": 236, "M": 224, "F": 240,
           "P": 159, "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
def load_candidate_index():
    """chain sequence -> list of (pdb_id, chain) from the v26 contact scan."""
    idx = {}
    with gzip.open(RAW_CONTACTS, "rt") as f:
        for line in f:
            d = json.loads(line)
            if d.get("status") != "ok":
                continue
            for ch, info in d["chains"].items():
                idx.setdefault(info["sequence"], []).append((d["pdb_id"], ch))
    return idx


def cif_path(pdb_id):
    for name in (f"{pdb_id.lower()}.cif", f"{pdb_id.upper()}.cif"):
        p = os.path.join(CIF_DIR, name)
        if os.path.exists(p):
            return p
    return None


class StructureCache:
    def __init__(self):
        self.parser = MMCIFParser(QUIET=True)
        self.sr = ShrakeRupley()
        self._cache = {}

    def get(self, pdb_id):
        """Parse once; return per-chain observed residues + DNA/Zn atom arrays."""
        if pdb_id in self._cache:
            return self._cache[pdb_id]
        p = cif_path(pdb_id)
        if p is None:
            self._cache[pdb_id] = None
            return None
        try:
            model = self.parser.get_structure(pdb_id, p)[0]
        except Exception as exc:
            log(f"  !! parse failed {pdb_id}: {exc}")
            self._cache[pdb_id] = None
            return None

        prot_chains, dna_atoms, zn_pos = {}, [], []
        for ch in model:
            reslist, seq = [], []
            for r in ch:
                het = r.id[0]
                nm = r.get_resname().strip()
                if het == " " and nm in protein_letters_3to1:
                    reslist.append(r)
                    seq.append(protein_letters_3to1[nm])
                elif nm in DNA_RES:
                    for a in r:
                        if a.element == "H":
                            continue
                        an = a.get_name().strip()
                        is_base = an not in SUGAR_PHOS
                        dna_atoms.append((a.coord, is_base,
                                          is_base and an in MAJOR_EDGE,
                                          is_base and an in MINOR_EDGE))
                elif a_is_zn(r):
                    zn_pos.append(next(iter(r)).coord)
            if len(reslist) >= 8:
                prot_chains[ch.id] = {"residues": reslist, "seq": "".join(seq)}

        dna_xyz = np.array([d[0] for d in dna_atoms], float).reshape(-1, 3)
        ent = {
            "model": model,
            "prot_chains": prot_chains,
            "dna_xyz": dna_xyz,
            "dna_tree": cKDTree(dna_xyz) if len(dna_xyz) else None,
            "dna_is_base": np.array([d[1] for d in dna_atoms], bool),
            "dna_is_major": np.array([d[2] for d in dna_atoms], bool),
            "dna_is_minor": np.array([d[3] for d in dna_atoms], bool),
            "zn_xyz": np.array(zn_pos, float).reshape(-1, 3),
            "_ppi_trees": {},
        }
        self._cache[pdb_id] = ent
        return ent


def a_is_zn(res):
    if res.id[0] == " ":
        return False
    nm = res.get_resname().strip().upper()
    return nm == "ZN" and len(res) == 1


def heavy(res):
    return [a for a in res if a.element != "H"]


def res_contacts(ent, residues):
    """Per-residue DNA contact flags for one chain's residue list (KD-tree)."""
    n = len(residues)
    out = {k: np.zeros(n, bool) for k in ("base", "backbone", "major", "minor", "zn")}
    tree, ZN = ent["dna_tree"], ent["zn_xyz"]
    for i, r in enumerate(residues):
        atoms = heavy(r)
        if not atoms:
            continue
        if tree is not None:
            xyz = np.array([a.coord for a in atoms], float)
            hit = set()
            for lst in tree.query_ball_point(xyz, BASE_CUT):
                hit.update(lst)
            if hit:
                h = np.fromiter(hit, int, len(hit))
                out["base"][i] = bool(ent["dna_is_base"][h].any())
                out["backbone"][i] = bool((~ent["dna_is_base"][h]).any())
                out["major"][i] = bool(ent["dna_is_major"][h].any())
                out["minor"][i] = bool(ent["dna_is_minor"][h].any())
        if len(ZN):
            lig = np.array([a.coord for a in atoms if a.element in ZN_LIGAND_ELEMENTS],
                           float).reshape(-1, 3)
            if len(lig):
                out["zn"][i] = bool((np.linalg.norm(lig[:, None, :] - ZN[None, :, :],
                                                    axis=-1) <= ZN_CUT).any())
    return out


def chain_rsa(sr, residues):
    """Relative SASA of the isolated chain (burial inside its own fold)."""
    import copy
    from Bio.PDB.Chain import Chain
    ch = Chain("X")
    for i, r in enumerate(residues):
        rc = copy.deepcopy(r)
        rc.id = (" ", i + 1, " ")
        rc.detach_parent()
        ch.add(rc)
    try:
        sr.compute(ch, level="R")
    except Exception as exc:
        log(f"  !! SASA failed: {exc}")
        return np.full(len(residues), np.nan)
    out = []
    for r, rc in zip(residues, ch):
        aa = protein_letters_3to1.get(r.get_resname().strip(), "A")
        out.append(rc.sasa / MAX_ASA.get(aa, 200.0))
    return np.array(out, float)


def chain_sse(residues):
    """pydssp three-state secondary structure from N/CA/C/O backbone coords."""
    import pydssp
    bb, ok = [], []
    for r in residues:
        try:
            bb.append([r["N"].coord, r["CA"].coord, r["C"].coord, r["O"].coord])
            ok.append(True)
        except KeyError:
            bb.append([[0.0, 0.0, 0.0]] * 4)
            ok.append(False)
    arr = np.array(bb, float)
    try:
        ss = np.array(list(pydssp.assign(arr, out_type="c3")))
    except Exception as exc:
        log(f"  !! pydssp failed: {exc}")
        return np.array(["-"] * len(residues))
    ss[~np.array(ok)] = "-"
    return ss


def ppi_flags(ent, chain_id, residues):
    """Heavy atom within BASE_CUT of any *other* protein chain."""
    if chain_id not in ent["_ppi_trees"]:
        other = []
        for cid, c in ent["prot_chains"].items():
            if cid == chain_id:
                continue
            for r in c["residues"]:
                other.extend(a.coord for a in heavy(r))
        ent["_ppi_trees"][chain_id] = cKDTree(np.array(other, float)) if other else None
    tree = ent["_ppi_trees"][chain_id]
    out = np.zeros(len(residues), bool)
    if tree is None:
        return out
    for i, r in enumerate(residues):
        xyz = np.array([a.coord for a in heavy(r)], float).reshape(-1, 3)
        if len(xyz):
            out[i] = any(len(l) for l in tree.query_ball_point(xyz, BASE_CUT))
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/moe_biochem")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N rows")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    ptab = pd.read_parquet(os.path.join(a.outdir, "protein_table.parquet"))
    v23 = json.load(open(V23_CONTACTS))
    rows = ptab[ptab.filename.astype(str).isin(v23.keys())].reset_index(drop=True)
    if a.limit:
        rows = rows.head(a.limit)
    log(f"{len(rows)} v23 rows carry co-crystal contact supervision")

    log("indexing candidate PDB chains by deposited sequence ...")
    seq_index = load_candidate_index()
    seq_keys = list(seq_index)
    log(f"  {len(seq_keys)} unique deposited chain sequences")

    cache = StructureCache()
    ann, mapping = [], []
    t0 = time.time()
    for n_done, r in enumerate(rows.itertuples(), 1):
        crop = str(r.seq)
        # v23 supervision set for this row (base-contacting residue indices)
        want = set()
        for _c, lst in v23[str(r.filename)]["cols"].items():
            for ridx, w in lst:
                if w > 0 and int(ridx) < len(crop):
                    want.add(int(ridx))

        cand_pdbs = []
        for q in seq_keys:
            if crop in q:
                cand_pdbs.extend(p for p, _c in seq_index[q])
        cand_pdbs = list(dict.fromkeys(cand_pdbs))[:12]

        best = None
        for pdb_id in cand_pdbs:
            ent = cache.get(pdb_id)
            if ent is None:
                continue
            # try every protein chain of this entry, at every position where the crop
            # occurs verbatim in that chain's *observed* residue sequence
            for chain_id, obs in ent["prot_chains"].items():
                off = obs["seq"].find(crop)
                while off >= 0:
                    residues = obs["residues"][off:off + len(crop)]
                    flags = res_contacts(ent, residues)
                    # v23 supervision marks residues contacting DNA at all (base or
                    # backbone), so score against that union, not base atoms alone.
                    got = set(np.where(flags["base"] | flags["backbone"])[0].tolist())
                    # v23's stored indices are offset from crop coordinates for a
                    # substantial fraction of rows, so allow a rigid shift when scoring
                    # the *mapping*; the annotation itself stays in crop coordinates.
                    sc = [(len({x + d for x in want} & got), d)
                          for d in range(-SHIFT_TOL, SHIFT_TOL + 1)]
                    nov, shift = max(sc)
                    recall = nov / max(len(want), 1)
                    prec = nov / max(len(got), 1)
                    score = (round(recall, 3), round(prec, 3))
                    if best is None or score > best[0]:
                        best = (score, pdb_id, chain_id, off, residues, flags, ent, shift)
                    off = obs["seq"].find(crop, off + 1)
        cands = cand_pdbs
        if best is None:
            mapping.append(dict(filename=r.filename, gene=r.gene_symbol, pdb=None,
                                chain=None, offset=-1, contact_recall=np.nan,
                                v23_index_shift=0, n_candidates=len(cand_pdbs)))
            continue

        (recall, _), pdb_id, chain_id, off, residues, flags, ent, shift = best
        rsa = chain_rsa(cache.sr, residues)
        ss = chain_sse(ent["prot_chains"][chain_id]["residues"])[off:off + len(crop)]
        ppi = ppi_flags(ent, chain_id, residues)

        dna_any = flags["base"] | flags["backbone"]
        ann.append(pd.DataFrame({
            "prot": int(r.prot), "pos": np.arange(len(crop)),
            "aa": list(crop),
            "pdb_id": pdb_id, "chain": chain_id,
            "base_contact": flags["base"], "backbone_contact": flags["backbone"],
            "major_groove": flags["major"], "minor_groove": flags["minor"],
            "zn_coord": flags["zn"],
            "rsa": rsa, "buried_core": rsa < RSA_BURIED,
            "sse": ss, "helix": ss == "H", "strand": ss == "E", "loop": ss == "-",
            "ppi": ppi,
            "exposed_nondna": (rsa >= RSA_BURIED) & ~dna_any & ~ppi,
        }))
        mapping.append(dict(filename=r.filename, gene=r.gene_symbol, pdb=pdb_id,
                            chain=chain_id, offset=off, contact_recall=round(recall, 3),
                            v23_index_shift=shift, n_candidates=len(cands)))
        if n_done % 25 == 0 or n_done == len(rows):
            log(f"  annotated {n_done}/{len(rows)} rows ({time.time() - t0:.0f}s)")

    ann = pd.concat(ann, ignore_index=True)
    mp = pd.DataFrame(mapping)
    ann.to_parquet(os.path.join(a.outdir, "structural_annotations.parquet"), index=False)
    mp.to_csv(os.path.join(a.outdir, "structure_mapping.csv"), index=False)

    ok = mp.pdb.notna()
    log(f"mapped {int(ok.sum())}/{len(mp)} rows to a co-crystal chain; "
        f"{len(ann)} annotated residues")
    log(f"  median contact recall vs v23 supervision = "
        f"{mp.contact_recall.median():.3f}  "
        f"(>=0.5 for {int((mp.contact_recall >= 0.5).sum())} rows)")
    nz = mp[mp.pdb.notna()].v23_index_shift
    log(f"  v23 index shift needed: 0 for {int((nz == 0).sum())}/{len(nz)} rows, "
        f"non-zero for {int((nz != 0).sum())} "
        f"(shift distribution: {dict(nz.value_counts().sort_index())})")
    log("role prevalence:")
    for c in ("base_contact", "backbone_contact", "major_groove", "minor_groove",
              "zn_coord", "buried_core", "helix", "strand", "loop", "ppi",
              "exposed_nondna"):
        log(f"  {c:18s} {ann[c].mean():.3f}  (n={int(ann[c].sum())})")


if __name__ == "__main__":
    main()
