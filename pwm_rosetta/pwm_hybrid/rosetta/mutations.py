"""DNA mutation and local minimization using PyRosetta."""

from pwm_hybrid.constants import COMPLEMENTARY, BASE_TO_FULL
from pwm_hybrid.rosetta.structure import get_dna_strand_positions


def mutate_dna_base(pose, position, new_base, dna_chain=None):
    """
    Mutate a DNA base at the given position in double-stranded DNA.

    Both the target base (strand 1) and its Watson-Crick complement
    (strand 2) are mutated.

    Parameters
    ----------
    pose : pyrosetta.Pose
    position : int
        1-based position in the DNA sequence.
    new_base : str
        New base letter ('A', 'C', 'G', or 'T').
    dna_chain : str, optional
        Ignored; kept for API compatibility.

    Returns
    -------
    pyrosetta.Pose
        Cloned pose with the mutation applied.
    """
    from pyrosetta.rosetta import protocols

    new_pose = pose.clone()
    strand1, strand2 = get_dna_strand_positions(new_pose)

    if position > len(strand1):
        raise ValueError(
            f"Position {position} out of range (strand length: {len(strand1)})"
        )

    pos_idx = position - 1
    strand1_pose_idx, strand1_base = strand1[pos_idx]

    new_base_upper = new_base.upper()
    new_resname = BASE_TO_FULL[new_base_upper]

    # Mutate strand 1
    protocols.simple_moves.MutateResidue(strand1_pose_idx, new_resname).apply(new_pose)

    # Mutate complementary position on strand 2
    if pos_idx < len(strand2):
        strand2_pose_idx, strand2_base = strand2[len(strand2) - 1 - pos_idx]
        expected_strand2 = COMPLEMENTARY[strand1_base]
        if strand2_base != expected_strand2:
            print(
                f"Warning: bases at position {position} are not complementary: "
                f"{strand1_base} vs {strand2_base}"
            )
        complementary_resname = BASE_TO_FULL[COMPLEMENTARY[new_base_upper]]
        protocols.simple_moves.MutateResidue(
            strand2_pose_idx, complementary_resname
        ).apply(new_pose)

    return new_pose


def minimize_around_mutation(pose, mutation_position, dna_chain=None, distance=6.0):
    """
    Local minimization around a mutation site on both DNA strands.

    Parameters
    ----------
    pose : pyrosetta.Pose
    mutation_position : int
        1-based position in the DNA sequence.
    dna_chain : str, optional
        Ignored; kept for API compatibility.
    distance : float
        Neighbourhood radius in Ångstroms.

    Returns
    -------
    pyrosetta.Pose
        The same pose after minimization (modified in-place and returned).
    """
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pwm_hybrid.rosetta import init as _init

    strand1, strand2 = get_dna_strand_positions(pose)
    if mutation_position > len(strand1):
        return pose

    pos_idx = mutation_position - 1
    strand1_pose_idx = strand1[pos_idx][0]

    strand2_pose_idx = None
    if pos_idx < len(strand2):
        strand2_pose_idx = strand2[pos_idx][0]

    mm = MoveMap()
    for i in range(1, pose.total_residue() + 1):
        res_xyz = pose.residue(i).atom(1).xyz()
        s1_xyz = pose.residue(strand1_pose_idx).atom(1).xyz()

        dist1 = (
            (res_xyz.x - s1_xyz.x) ** 2
            + (res_xyz.y - s1_xyz.y) ** 2
            + (res_xyz.z - s1_xyz.z) ** 2
        ) ** 0.5

        near = dist1 < distance
        if strand2_pose_idx is not None:
            s2_xyz = pose.residue(strand2_pose_idx).atom(1).xyz()
            dist2 = (
                (res_xyz.x - s2_xyz.x) ** 2
                + (res_xyz.y - s2_xyz.y) ** 2
                + (res_xyz.z - s2_xyz.z) ** 2
            ) ** 0.5
            near = near or dist2 < distance

        if near:
            mm.set_bb(i, True)
            mm.set_chi(i, True)

    min_mover = MinMover()
    min_mover.score_function(_init.sfxn)
    min_mover.movemap(mm)
    min_mover.apply(pose)

    return pose
