#!/usr/bin/env python
"""Master runner for the ADNP homeodomain case study."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
STEPS = ["run_adnp.py", "run_homeodomain_masked_control.py", "scan_neurodev_promoters_adnp.py",
         "make_figure7.py", "write_manuscript_adnp.py"]
for s in STEPS:
    print(f"\n===== {s} =====")
    subprocess.run([PY, os.path.join(HERE, s)], check=True)
print("\nDone. See results/adnp_case/ and figures/figure7/")
