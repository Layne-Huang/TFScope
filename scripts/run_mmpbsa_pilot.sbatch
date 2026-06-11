#!/bin/bash
#SBATCH -J mmpbsa_pilot
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 16:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/pilot_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/pilot_%j.err

# MM-GBSA DNA base scanning pilot on 31 cases where Rosetta currently improves.
# Uses PyRosetta for DNA mutation + OpenMM AMBER14/DNA.OL15 + GBn2 for energies.
# Replaces Rosetta RM8B ΔΔG with proper implicit-solvent energetics.

mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1

set -eo pipefail

echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"
nvidia-smi || true

cd /n/home13/leihuang/project/TFScope

python scripts/run_mmpbsa_scan.py \
    --pilot \
    --out results/mmpbsa_v10_pilot \
    --tau 1.5 \
    --min-iter 300

echo "Pilot complete at $(date)"
echo "Results: $(ls results/mmpbsa_v10_pilot/*.npz 2>/dev/null | wc -l) / 31 samples"
