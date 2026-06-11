"""PDB/CIF file I/O and structure-preparation utilities."""

import gemmi
from Bio.PDB import PDBParser, PDBIO, Chain
from Bio.PDB.Polypeptide import is_aa

from pwm_hybrid.constants import DNA_RESIDUES, METAL_RES, BASE_MAPPING


def convert_cif_to_pdb(cif_path, pdb_path=None):
    """Convert CIF file to PDB format."""
    cif_block = gemmi.cif.read_file(cif_path)[0]
    structure = gemmi.make_structure_from_block(cif_block)

    for model in structure:
        for chain in model:
            if len(chain.name) > 1:
                chain.name = chain.name[0]

    if pdb_path is None:
        pdb_path = cif_path.replace('.cif', '.pdb')
    structure.write_pdb(pdb_path)
    return pdb_path


def rename_chains(input_pdb, output_pdb):
    """
    Rebuild a 2-chain complex:
      - Protein -> chain A
      - DNA     -> chain B
      - Metals  -> appended to chain A

    Returns
    -------
    metal_free : int
        1 if no metals present, 0 otherwise.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", input_pdb)
    model = next(structure.get_models())

    dna_residues = []
    protein_residues = []
    metal_residues = []

    for chain in list(model):
        for res in chain:
            resname = res.resname.strip()
            if resname in METAL_RES:
                metal_residues.append(res.copy())
                continue
            if resname in DNA_RESIDUES:
                dna_residues.append(res.copy())
                continue
            if is_aa(res, standard=True):
                protein_residues.append(res.copy())
                continue
        model.detach_child(chain.id)

    chainB = Chain.Chain("B")
    for idx, res in enumerate(dna_residues, start=1):
        het, _oldid, icode = res.id
        res.id = (het, idx, icode)
        chainB.add(res)

    chainA = Chain.Chain("A")
    next_idx = 1
    for res in protein_residues:
        het, _oldid, icode = res.id
        res.id = (het, next_idx, icode)
        chainA.add(res)
        next_idx += 1
    for res in metal_residues:
        het, _oldid, icode = res.id
        if het == " ":
            het = f"H_{res.resname.strip()}"
        res.id = (het, next_idx, icode)
        chainA.add(res)
        next_idx += 1

    model.add(chainB)
    model.add(chainA)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)

    return 1 if len(metal_residues) == 0 else 0


def extract_dna_sequence(pose):
    """
    Extract DNA sequence from a PyRosetta pose (first strand only).

    Returns
    -------
    str
        Single-letter DNA sequence of the first strand.
    """
    dna_positions = []
    for i in range(1, pose.total_residue() + 1):
        resname = pose.residue(i).name()
        base_name = resname.split(':')[0]
        if base_name in BASE_MAPPING:
            dna_positions.append((i, BASE_MAPPING[base_name]))

    if not dna_positions:
        return ""

    strand_length = len(dna_positions) // 2
    return ''.join(base for _, base in dna_positions[:strand_length])


def get_dna_strand_positions(pose):
    """
    Return positions for both DNA strands.

    Returns
    -------
    tuple[list, list]
        (strand1, strand2) where each element is a list of
        ``(pose_index, base)`` tuples.
    """
    dna_positions = []
    for i in range(1, pose.total_residue() + 1):
        resname = pose.residue(i).name()
        base_name = resname.split(':')[0]
        if base_name in BASE_MAPPING:
            dna_positions.append((i, BASE_MAPPING[base_name]))

    strand_length = len(dna_positions) // 2
    return dna_positions[:strand_length], dna_positions[strand_length:strand_length * 2]
