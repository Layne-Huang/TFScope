#!/usr/bin/env python
"""Master runner for the ZGLP1 GATA-class case study."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
STEPS = ["run_zglp1.py", "run_gata_masked_control.py", "scan_germ_cell_promoters_zglp1.py",
         "make_figure6.py", "write_manuscript_zglp1.py"]
for s in STEPS:
    print(f"\n===== {s} =====")
    subprocess.run([PY, os.path.join(HERE, s)], check=True)
print("\nDone. See results/zglp1_case/ and figures/figure6/")
