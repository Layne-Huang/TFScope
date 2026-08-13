#!/bin/bash
cd /afs/csail.mit.edu/u/l/leihuang/project/TFScope
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
mamba activate tfscope 2>/dev/null || conda activate tfscope 2>/dev/null || true
python scripts/run_sohlh1_adnp_cre.py
echo "=== ALL_DONE exit=$? $(date) ==="
