"""Interface ddG scoring using PyRosetta."""

import os

from pwm_hybrid.rosetta import init as _init


def _protein_dna_chain_groups(pose):
    """Return an InterfaceAnalyzer interface string 'PROT_DNA' grouping ALL
    protein chains vs ALL nucleic-acid chains (e.g. 'AB_CD' for a dimer)."""
    info = pose.pdb_info()
    prot, na = [], []
    for ch_idx in range(1, pose.num_chains() + 1):
        begin = pose.chain_begin(ch_idx)
        res = pose.residue(begin)
        letter = info.chain(begin) if info is not None else chr(64 + ch_idx)
        if res.is_protein():
            prot.append(letter)
        elif res.is_NA() or res.is_DNA() or res.is_RNA():
            na.append(letter)
    return "".join(sorted(set(prot))) + "_" + "".join(sorted(set(na)))


def _interface_dg_protein_dna(pose):
    """Binding dG of (all protein chains) vs (all DNA chains), independent of
    chain count — correct for multi-chain (dimer) complexes where the default
    jump=1 ddg_filter would only separate chain 1 from the rest."""
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    interface = _protein_dna_chain_groups(pose)
    ia = InterfaceAnalyzerMover(interface)
    ia.set_compute_packstat(False)
    ia.set_pack_separated(True)
    if _init.sfxn is not None:
        ia.set_scorefunction(_init.sfxn)
    ia.apply(pose)
    return ia.get_interface_dG()


def calculate_interface_ddg(pose):
    """
    Calculate interface ddG.

    Default: the xml_loader ddg_filter (jump=1, chain-1-vs-rest) — correct for a
    2-chain (1 protein + DNA) complex, the standard pwm_rosetta case.

    Set env ``PWM_INTERFACE_MODE=prot_dna`` to instead score ALL protein chains
    vs ALL DNA chains via InterfaceAnalyzer — required for multi-chain (dimer)
    complexes, where jump=1 would conflate the protein-protein interface with
    only one protomer's DNA contacts.

    Removes constraints before computing (mirrors eval_pdb.py workflow).
    """
    pose.remove_constraints()
    if os.environ.get("PWM_INTERFACE_MODE", "").lower() == "prot_dna":
        return _interface_dg_protein_dna(pose)
    return _init.ddg_filter.compute(pose)


def calculate_ddg_with_relax(wt_pose, mut_pose, scorefxn=None):
    """
    Return ``E(mutant) - E(wildtype)`` using interface ddG.

    Both poses should be minimised/relaxed before calling this.

    Parameters
    ----------
    scorefxn : optional
        Unused; kept for API compatibility.

    Returns
    -------
    float
        Delta ddG (positive = weaker binding).
    """
    wt_ddg = calculate_interface_ddg(wt_pose)
    mut_ddg = calculate_interface_ddg(mut_pose)
    return mut_ddg - wt_ddg
