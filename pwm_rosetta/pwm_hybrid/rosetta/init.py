"""
Deferred PyRosetta initialisation.

Call ``init_pyrosetta()`` once before using any rosetta submodule functions.
All Rosetta weight/flag files are bundled inside this package — no external
``dbp_design`` installation is needed.
"""

import os

# Path to the bundled data directory (flags_and_weights/ lives here)
_DATA_DIR = os.path.join(os.path.dirname(__file__), '_data')
_FLAGS_DIR = os.path.join(_DATA_DIR, 'flags_and_weights')

# Bundled file paths (absolute, resolved at import time)
_WEIGHTS_FILE = os.path.join(_FLAGS_DIR, 'RM8B_torsional.wts')
_RELAX_SCRIPT = os.path.join(_FLAGS_DIR, 'no_ref.rosettacon2018.beta_nov16_constrained.txt')
_INIT_FLAGS   = os.path.join(_FLAGS_DIR, 'RM8B_flags')

# Module-level globals set by init_pyrosetta()
ddg_filter = None
sfxn = None
fast_relax = None

# Default psipred path (overridable via env var)
_DEFAULT_PSIPRED = os.environ.get('PSIPRED_EXE', '/software/psipred4/runpsipred_single')


def get_pyrosetta_init_flags():
    """Return the standard ``pyrosetta.init()`` flags string using bundled files."""
    return (
        "-mute all "
        "-beta_nov16 "
        "-in:file:silent_struct_type binary "
        "-holes:dalphaball /software/rosetta/DAlphaBall.gcc "
        "-use_terminal_residues true "
        "-mute basic.io.database core.scoring "
        f"@{_INIT_FLAGS}"
    )


def init_pyrosetta(psipred_exe=None):
    """
    Initialise PyRosetta and load the bundled xml_loader components.

    Parameters
    ----------
    psipred_exe : str, optional
        Path to the ``runpsipred_single`` executable for secondary-structure
        prediction filters.  Defaults to the ``PSIPRED_EXE`` environment
        variable, then ``/software/psipred4/runpsipred_single``.
        Pass an empty string ``''`` to disable SSPrediction filters.
        The ΔΔG calculation is unaffected either way.
    """
    global ddg_filter, sfxn, fast_relax

    if psipred_exe is None:
        psipred_exe = _DEFAULT_PSIPRED
        if not os.path.isfile(psipred_exe):
            psipred_exe = ''  # disable SSPrediction silently

    from pwm_hybrid.rosetta import _xml_loader as xml_loader
    from pyrosetta.rosetta import protocols, core

    (_, _, _, fast_relax,
     ddg_filter, _, _, _,
     _, _) = xml_loader.fast_relax(
        protocols,
        _WEIGHTS_FILE,
        psipred_exe,
        _RELAX_SCRIPT,
    )

    sfxn = core.scoring.ScoreFunctionFactory.create_score_function(_WEIGHTS_FILE)

    print("PyRosetta initialised with bundled ddg_filter and fast_relax.")
    print(f"  Weights:      {_WEIGHTS_FILE}")
    print(f"  Relax script: {_RELAX_SCRIPT}")
    print(f"  psipred:      {psipred_exe or '(disabled)'}")
