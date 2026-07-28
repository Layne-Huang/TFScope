"""Build a TRUE per-residue DNA-contact prior from a protein-DNA complex.

Use with `TFScopeModel.forward(..., contact_override=prior)` to inject real
contacts at inference when a structure exists; when it doesn't, the model's
frozen ESM->contact probe head predicts the prior instead.
"""
import numpy as np

AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
       'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
       'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
DNA_RES = {'DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'T'}


def contacts_from_pdb(pdb_path: str, protein_seq: str, cutoff: float = 4.5):
    """Return (prior, n_contacts).

    prior: float array of len(protein_seq), 1.0 at residues with any heavy atom
           within `cutoff` A of any DNA atom, else 0.0 — aligned to protein_seq
           by locating the PDB chain's sequence inside it.
    Returns (None, 0) if the PDB has no DNA or the chain can't be aligned.
    """
    prot, dna = {}, []
    for line in open(pdb_path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resn = line[17:20].strip()
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if resn in DNA_RES:
            dna.append(xyz)
        elif resn in AA3:
            ri = int(line[22:26])
            prot.setdefault(ri, [AA3[resn], []])[1].append(xyz)
    if not dna or not prot:
        return None, 0

    D = np.asarray(dna)
    resnums = sorted(prot)
    chain_seq = "".join(prot[r][0] for r in resnums)
    contact = {
        r for r in resnums
        if ((np.asarray(prot[r][1])[:, None, :] - D[None, :, :]) ** 2).sum(-1).min() < cutoff ** 2
    }

    prior = np.zeros(len(protein_seq), dtype=np.float32)
    off = protein_seq.find(chain_seq)
    if off < 0:                                   # fall back to positional alignment
        placed = 0
        for j, r in enumerate(resnums):
            if j < len(protein_seq) and protein_seq[j] == prot[r][0] and r in contact:
                prior[j] = 1.0; placed += 1
        return (prior, placed) if placed else (None, 0)
    for j, r in enumerate(resnums):
        k = off + j
        if r in contact and 0 <= k < len(protein_seq):
            prior[k] = 1.0
    return prior, int(prior.sum())
