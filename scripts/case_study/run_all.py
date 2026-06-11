#!/usr/bin/env python
"""Master runner for the revised confidence-calibrated SOHLH1 case study."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
STEPS = ["build_calibration.py", "run_sohlh1.py", "build_orphan_distribution.py",
         "run_sohlh2_masked_control.py", "scan_germ_cell_promoters.py", "make_figure5.py",
         "write_manuscript.py"]
for s in STEPS:
    print(f"\n===== {s} =====")
    subprocess.run([PY, os.path.join(HERE, s)], check=True)
print("\nDone. See results/sohlh1_case/ and figures/figure5/")
