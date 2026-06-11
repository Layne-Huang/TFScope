#!/usr/bin/env python
"""Master runner for the remaining orphan-TF nominations (ADNP2, ZHX2, ZHX3) on deeppbs v18a RAG."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
STEPS = ["build_calibration_deeppbs.py", "run_orphans_deeppbs.py",
         "scan_promoters_orphans_deeppbs.py", "make_figure_orphans.py", "write_manuscript_orphans.py"]
for s in STEPS:
    print(f"\n===== {s} =====")
    subprocess.run([PY, os.path.join(HERE, s)], check=True)
print("\nDone. See results/orphan_homeodomain_deeppbs/ and figures/figure8/")
