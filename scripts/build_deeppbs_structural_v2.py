#!/usr/bin/env python
"""Rebuild the DeepPBS-derived structural TF-DNA dataset (v2).

Supersedes build_deeppbs_only_dataset.py. Key differences, worked out over an
extended data-audit session (see docs/ or ask Lei for the conversation log):

  1. Sequence comes from the real PDB/mmCIF chain (Bio.PDB), NOT from DeepPBS's
     own npz atom-graph. The npz graph is a DNA-proximity-cropped, ATOM-level
     representation of the WHOLE biological assembly (every chain in the
     structure, e.g. FOS+JUN+NFAT bundled into one graph) with no residue
     identity string and no chain boundary marker -- unusable as a clean
     per-TF sequence input.

  2. DBD cropping uses a CONTIGUOUS span: residues within CONTACT_CUTOFF (A)
     of any DNA atom, then the crop is [min_contact_resnum, max_contact_resnum]
     -- NOT just the literal contacting residues (which can have scattered
     internal gaps from loops/turns swinging away from the DNA; confirmed on
     CTCF/5und_A). A contiguous crop keeps ESM's local sequence context intact.

  3. Residue extraction tracks true author residue numbers so crystallographic
     disorder gaps are detected (recorded, chain split) rather than silently
     glued together (the original script's Bio.PDB PPBuilder-then-join bug,
     confirmed on PDB 5t00's CTCF chain).

  4. Dimers are detected EMPIRICALLY, not via a hardcoded family whitelist: if
     a second protein chain also has residues within CONTACT_CUTOFF of the
     SAME DNA, both chains are kept together as one paired training example.
     Single-chain structures stay single-chain.

  5. PWM is assigned by gene, preferring our OWN pwm.parquet (already newer --
     HOCOMOCO v13 Core / CIS-BP v1.94 / current JASPAR -- than DeepPBS's own
     embedded H11MO-tagged motifs), falling back to a live JASPAR REST fetch
     (latest version by base ID) only for genes missing from our archive.

Candidate PDB pool = union of:
  (a) the 566 structures in DeepPBS's own preprocessed npz cache
      (/data1/leihuang/DeepPBS/deeppbsmar24/data/assembly2024)
  (b) 513 additional structures found via RCSB search matched to our existing
      ~1322 known TF UniProt IDs (genes we already have PWM data for)
  (c) structures for ~53 clean new TF genes found via a DBD-Pfam-family RCSB
      search that fall outside our current gene list entirely (STAT3, FOXP3,
      RORC, SOX9, NF-kB family, bHLH-PAS clock genes, etc.)

Output: data/processed/tf_pwm_deeppbs_v2.parquet
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import requests
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

CIF_CACHE = "data/raw/pdb_cif_cache"
CONTACT_CUTOFF = 5.0          # Angstrom, residue-DNA direct contact
OUT_PARQUET = "data/processed/tf_pwm_deeppbs_v2.parquet"
FAIL_LOG = "data/processed/tf_pwm_deeppbs_v2_failures.json"

DNA_RESNAMES = {"DA", "DC", "DG", "DT", "DI", "DU"}

HOCOMOCO_ALIASES = {
    "BC11A": "BCL11A", "BMAL1": "ARNTL", "BRAC": "TBXT", "COE1": "EBF1",
    "ERR2": "ESRRB", "GCR": "NR3C1", "HXA13": "HOXA13", "HXA9": "HOXA9",
    "HXB13": "HOXB13", "ITF2": "TCF4", "KAISO": "ZBTB33", "NDF1": "NEUROD1",
    "NFAC1": "NFATC1", "NFAC2": "NFATC2", "NKX25": "NKX2-5", "P63": "TP63",
    "P73": "TP73", "PEBB": "CBFB", "PIT1": "POU1F1",
}

_session = requests.Session()


def fetch_cif(pdb_id):
    os.makedirs(CIF_CACHE, exist_ok=True)
    path = os.path.join(CIF_CACHE, f"{pdb_id.lower()}.cif")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.cif"
    resp = _session.get(url, timeout=60)
    resp.raise_for_status()
    with open(path, "w") as f:
        f.write(resp.text)
    return path


def load_chains(cif_path):
    """Return (protein_chains, dna_atom_coords) for the first model.

    protein_chains: {chain_id: [(auth_resnum, icode, aa_letter, [atom_coords])]}
    dna_atom_coords: (N,3) array of every DNA heavy-atom coordinate, any chain.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", cif_path)
    model = next(iter(structure))

    protein_chains = {}
    dna_coords = []
    dna_chain_atoms = {}  # chain_id -> list of atom coords (for dimer/DNA-grouping)

    for chain in model:
        residues = list(chain)
        is_dna_chain = any(r.get_resname().strip() in DNA_RESNAMES for r in residues)
        is_protein_chain = any(is_aa(r, standard=True) for r in residues)

        if is_dna_chain and not is_protein_chain:
            coords = []
            for r in residues:
                if r.get_resname().strip() not in DNA_RESNAMES:
                    continue
                for atom in r:
                    coords.append(atom.coord)
            if coords:
                dna_chain_atoms[chain.id] = np.array(coords)
                dna_coords.extend(coords)
            continue

        if is_protein_chain:
            entries = []
            for r in residues:
                if not is_aa(r, standard=True):
                    continue
                hetflag, resnum, icode = r.id
                aa = seq1(r.get_resname())
                if aa == "X":
                    continue
                coords = np.array([atom.coord for atom in r])
                entries.append((resnum, icode, aa, coords))
            if entries:
                protein_chains[chain.id] = entries

    dna_coords = np.array(dna_coords) if dna_coords else np.zeros((0, 3))
    return protein_chains, dna_coords, dna_chain_atoms


def contiguous_dbd_crop(entries, dna_coords, cutoff=CONTACT_CUTOFF):
    """entries: list of (resnum, icode, aa, atom_coords) in chain order (already
    sorted as encountered in the structure, which is monotonic in resnum for
    well-formed mmCIF). Returns (crop_seq, gap_flag, n_contact_residues) or
    None if no residue contacts DNA.
    """
    if len(dna_coords) == 0:
        return None

    contact_mask = []
    for resnum, icode, aa, coords in entries:
        d = np.sqrt(((coords[:, None, :] - dna_coords[None, :, :]) ** 2).sum(-1)).min()
        contact_mask.append(d <= cutoff)

    if not any(contact_mask):
        return None

    idxs = [i for i, c in enumerate(contact_mask) if c]
    lo, hi = min(idxs), max(idxs)
    window = entries[lo:hi + 1]

    # detect true disorder gaps: author resnum should increase by exactly 1
    # (ignoring insertion codes) across the contiguous window we're taking
    gap_flag = False
    for a, b in zip(window[:-1], window[1:]):
        if b[0] - a[0] > 1:
            gap_flag = True
            break

    crop_seq = "".join(e[2] for e in window)
    return crop_seq, gap_flag, sum(contact_mask)


def find_dimer_partners(protein_chains, dna_coords, dna_chain_atoms, cutoff=CONTACT_CUTOFF):
    """Return ({chain_id: crop_result}, {chain_id: primary_DNA_chain_id_set}, {chain_id: [real_partner_chain_ids]}).

    The third dict is the authoritative, final partner list -- callers should
    use it directly rather than re-deriving partnership from the DNA-contact
    sets themselves (that split once caused a real bug: a later patch reused
    only the DNA-sharing criterion and silently dropped the protein-protein
    contact requirement, re-introducing false-positive dimers like 1DU0 A/B,
    15.9A apart with zero real contact).

    Partnership is defined by DIRECT contact with the SAME specific DNA chain
    letter(s), not by clustering DNA chains via inter-chain distance -- that
    approach failed on 5yef, where crystal packing puts 4 INDEPENDENT
    CTCF-DNA copies' duplexes within the naive clustering cutoff of each
    other even though each protein chain only ever touches its own duplex
    (confirmed: chain A -> DNA C/D at 2.3-2.8A, all other DNA >=7.5A). Tying
    partnership to the literal contacted chain ID sidesteps that ambiguity.
    """
    if not dna_chain_atoms:
        out = {}
        for cid, entries in protein_chains.items():
            res = contiguous_dbd_crop(entries, dna_coords, cutoff)
            if res is not None:
                out[cid] = res
        return out, {}, {cid: [] for cid in out}

    out = {}
    chain_dna_contacts = {}
    for cid, entries in protein_chains.items():
        # count contact RESIDUES per DNA chain (not just min-distance), so we
        # can tell a real binding interface from an incidental reach-across
        # contact at a crystal-packing seam (confirmed necessary on 5E8I:
        # chain D has a genuine 29-residue interface with its own duplex E/F,
        # but only a secondary 12-residue reach-across onto the NEIGHBORING,
        # sequence-identical duplex B/C where chain A is independently bound
        # -- same protein, same DNA sequence, crystallographic redundancy,
        # not a real composite/tandem dimer)
        n_contact_per_dna = {}
        for dna_cid, dna_atoms in dna_chain_atoms.items():
            n = 0
            for resnum, icode, aa, coords in entries:
                d = np.sqrt(((coords[:, None, :] - dna_atoms[None, :, :]) ** 2).sum(-1)).min()
                if d <= cutoff:
                    n += 1
            if n > 0:
                n_contact_per_dna[dna_cid] = n
        if not n_contact_per_dna:
            continue
        max_n = max(n_contact_per_dna.values())
        # keep only this chain's PRIMARY duplex: DNA chains carrying at least
        # half its strongest contact count (keeps both complementary strands
        # of one real duplex, drops a secondary reach-across duplex)
        primary = {c for c, n in n_contact_per_dna.items() if n >= 0.5 * max_n}

        own_dna_coords = np.vstack([dna_chain_atoms[c] for c in primary])
        res = contiguous_dbd_crop(entries, own_dna_coords, cutoff)
        if res is not None:
            out[cid] = res
            chain_dna_contacts[cid] = primary

    # Final partnership = BOTH (1) shares a primary DNA duplex AND (2) real
    # direct protein-protein contact. Either check alone is insufficient:
    # DNA-duplex-sharing alone falsely pairs unrelated chains that just reach
    # across a crystal-packing seam (e.g. 1DU0 A/B, 15.9A apart, no contact,
    # yet shared a DNA chain under the naive rule); protein-protein contact
    # alone was never used in isolation but is required here so any future
    # caller can't accidentally skip it the way a separate patch script did.
    protein_partners = {}
    chain_ids = list(out.keys())
    for i, c1 in enumerate(chain_ids):
        coords1 = np.vstack([e[3] for e in protein_chains[c1]])
        real_partners = []
        for c2 in chain_ids:
            if c2 == c1:
                continue
            if not (chain_dna_contacts.get(c2, set()) & chain_dna_contacts.get(c1, set())):
                continue
            coords2 = np.vstack([e[3] for e in protein_chains[c2]])
            d = np.sqrt(((coords1[:, None, :] - coords2[None, :, :]) ** 2).sum(-1)).min()
            if d <= 8.0:
                real_partners.append(c2)
        protein_partners[c1] = real_partners

    return out, chain_dna_contacts, protein_partners


# ── gene / PWM resolution ────────────────────────────────────────────────────

_main_pwm_df = None
_uid_to_gene = None
_gene_to_pwm_rows = None


def _load_main_pwm():
    global _main_pwm_df, _uid_to_gene, _gene_to_pwm_rows
    if _main_pwm_df is not None:
        return
    _main_pwm_df = pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim.parquet")
    _uid_to_gene = dict(zip(_main_pwm_df["uniprot_id"].astype(str), _main_pwm_df["gene_symbol"].astype(str)))
    _gene_to_pwm_rows = {}
    for _, r in _main_pwm_df.iterrows():
        g = str(r["gene_symbol"]).upper()
        _gene_to_pwm_rows.setdefault(g, []).append(r)


_uid_genename_cache = {}


def _uniprot_gene_name(uid):
    """Look up the gene symbol UniProt itself assigns to an accession, so we
    can match e.g. mouse Stat3 (P42227) to human STAT3 (P40763, already in
    our gene list) by SYMBOL when the accessions themselves don't overlap --
    confirmed necessary empirically (STAT3 case)."""
    if uid in _uid_genename_cache:
        return _uid_genename_cache[uid]
    try:
        resp = _session.get(f"https://rest.uniprot.org/uniprotkb/{uid}.json",
                             params={"fields": "gene_names"}, timeout=15)
        resp.raise_for_status()
        genes = resp.json().get("genes", [])
        name = genes[0]["geneName"]["value"] if genes else None
    except Exception:
        name = None
    _uid_genename_cache[uid] = name
    return name


def resolve_gene(uniprot_ids):
    """Given a list of UniProt accessions for one chain, return a gene symbol
    if any resolve against our known set. Two passes: (1) direct accession
    match, applying the HOCOMOCO alias table; (2) cross-species fallback by
    gene SYMBOL via a live UniProt lookup, since a structure's protein may
    carry a different species' ortholog accession than what's in our own
    archive (e.g. mouse Stat3 vs human STAT3 already in our gene list)."""
    _load_main_pwm()
    known_gene_upper = {g.upper() for g in _gene_to_pwm_rows}

    for u in uniprot_ids:
        if u in _uid_to_gene:
            return _uid_to_gene[u]

    canonical_upper_to_stored = {g.upper(): g for g in _gene_to_pwm_rows}
    for u in uniprot_ids:
        gname = _uniprot_gene_name(u)
        if not gname:
            continue
        gname_resolved = HOCOMOCO_ALIASES.get(gname.upper(), gname)
        if gname_resolved.upper() in known_gene_upper:
            # return OUR table's canonical casing, not UniProt's, so the same
            # gene never splits into two differently-cased set entries
            return canonical_upper_to_stored[gname_resolved.upper()]
    return None


_jaspar_cache = {}


def fetch_jaspar_latest(gene_name):
    if gene_name in _jaspar_cache:
        return _jaspar_cache[gene_name]
    url = f"https://jaspar.elixir.no/api/v1/matrix/?search={gene_name}&format=json"
    try:
        resp = _session.get(url, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        results = [r for r in results if r.get("name", "").upper() == gene_name.upper()]
        if not results:
            _jaspar_cache[gene_name] = None
            return None
        results.sort(key=lambda x: int(x.get("version", 0)), reverse=True)
        best = results[0]
        detail = _session.get(f"https://jaspar.elixir.no/api/v1/matrix/{best['matrix_id']}/?format=json", timeout=30)
        detail.raise_for_status()
        _jaspar_cache[gene_name] = detail.json()
        return _jaspar_cache[gene_name]
    except Exception:
        _jaspar_cache[gene_name] = None
        return None


_GRAPHQL_CHAIN_QUERY = """
query($id: String!) {
  entry(entry_id: $id) {
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        auth_asym_ids
        reference_sequence_identifiers { database_name database_accession }
      }
    }
  }
}"""


def fetch_chain_uniprot_map(pdb_id):
    """Return {auth_chain_id: [uniprot_accessions]} for one PDB entry, so a
    heteromeric complex's chains (e.g. FOS vs JUN) get the RIGHT gene each,
    rather than one flat uniprot set for the whole entry."""
    try:
        resp = requests.post("https://data.rcsb.org/graphql",
                              json={"query": _GRAPHQL_CHAIN_QUERY, "variables": {"id": pdb_id}},
                              timeout=30)
        resp.raise_for_status()
        entry = resp.json()["data"]["entry"]
        if entry is None:
            return {}
    except Exception:
        return {}

    chain_map = {}
    for pe in (entry.get("polymer_entities") or []):
        ids = pe["rcsb_polymer_entity_container_identifiers"]
        chains = ids.get("auth_asym_ids") or []
        uids = [ref["database_accession"] for ref in (ids.get("reference_sequence_identifiers") or [])
                if ref["database_name"] == "UniProt"]
        for c in chains:
            chain_map[c] = uids
    return chain_map


def resolve_pwm(gene_name):
    """Return (pwm_array (4,L), source, source_id) or None."""
    _load_main_pwm()
    rows = _gene_to_pwm_rows.get(gene_name.upper())
    if rows:
        # prefer HOCOMOCO > JASPAR > CISBP > others, already-latest-version archive
        pref = {"HOCOMOCO": 0, "JASPAR": 1, "CISBP": 2}
        rows_sorted = sorted(rows, key=lambda r: pref.get(r["source"], 9))
        r = rows_sorted[0]
        pwm = np.frombuffer(r["pwm"], dtype=np.float32).reshape(4, -1)
        return pwm, r["source"], r["source_id"]

    jdata = fetch_jaspar_latest(gene_name)
    if jdata:
        pfm = jdata.get("pfm")
        if pfm:
            pwm = np.array([pfm["A"], pfm["C"], pfm["G"], pfm["T"]], dtype=np.float32)
            return pwm, "JASPAR_live", jdata["matrix_id"]
    return None


if __name__ == "__main__":
    print("Module loaded OK. Run via build_deeppbs_structural_v2_driver.py for the batch job.")
