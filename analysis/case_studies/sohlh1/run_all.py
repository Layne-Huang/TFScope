#!/usr/bin/env python
"""Master runner for the SOHLH1 orphan-TF case study."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
for step in ["run_case_study.py", "plot_figure5.py", "write_manuscript.py"]:
    print(f"\n===== {step} =====")
    subprocess.run([PY, os.path.join(HERE, step)], check=True)
print("\nDone. See results/case_study_sohlh1/")
