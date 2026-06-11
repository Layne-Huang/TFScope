"""Interface ddG scoring using PyRosetta."""

from pwm_hybrid.rosetta import init as _init


def calculate_interface_ddg(pose):
    """
    Calculate interface ddG using the xml_loader ddg_filter.

    Removes constraints before computing (mirrors eval_pdb.py workflow).

    Returns
    -------
    float
        ddG value (binding free energy).
    """
    pose.remove_constraints()
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
