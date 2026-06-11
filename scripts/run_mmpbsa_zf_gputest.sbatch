#!/bin/bash
#SBATCH -J mmpbsa_zf
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/zf_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/zf_%j.err

# MM-GBSA (OpenMM AMBER14/DNA.OL15 + GBn2) base scan on the 4 ZF design structures.
mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/run_mmpbsa_zf.py --all --out results/zf_struct/mmpbsa --tau 1.5 --min-iter 300
echo "Done at $(date)"
