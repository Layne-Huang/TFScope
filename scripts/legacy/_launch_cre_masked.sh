#!/bin/bash
cd /afs/csail.mit.edu/u/l/leihuang/project/TFScope
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
mamba activate tfscope 2>/dev/null || conda activate tfscope 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0 TORCH_HOME=/data1/leihuang/.cache/torch HF_HOME=/data1/leihuang/.cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/run_fig3bc_cre_masked.py
echo "=== ALL_DONE exit=$? $(date) ==="
