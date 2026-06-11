"""
pwm_hybrid — Hybrid AF3 + PyRosetta PWM generation package.

Public API (no PyRosetta or AF3 required at import time):

    from pwm_hybrid import pwm_from_hybrid_csv, plot, HAS_AF3
    from pwm_hybrid import generate_pwm_hybrid   # needs PyRosetta initialised first
"""

from pwm_hybrid._version import __version__
from pwm_hybrid.pwm.energies import pwm_from_energies_with_temperature, pwm_from_hybrid_csv
from pwm_hybrid.pwm.viz import seqToOneHot, makeLogo, plotPWM, plot
from pwm_hybrid.af3 import HAS_AF3
from pwm_hybrid.pipeline import generate_pwm_hybrid

__all__ = [
    "__version__",
    "pwm_from_energies_with_temperature",
    "pwm_from_hybrid_csv",
    "seqToOneHot",
    "makeLogo",
    "plotPWM",
    "plot",
    "HAS_AF3",
    "generate_pwm_hybrid",
]
