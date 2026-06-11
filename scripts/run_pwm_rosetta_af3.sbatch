#!/bin/bash
#SBATCH -J pwmros_af3_zf
#SBATCH -p sapphire
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/af3_zf_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/af3_zf_%j.err

# Canonical pwm_rosetta README workflow: AF3 wild-type fold -> PyRosetta single-mutation
# ΔΔG scan -> Boltzmann PWM, for TFScope's predicted zinc-finger PWMs (zf21/zf93/zf92/zf129).
mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"
cd /n/home13/leihuang/project/TFScope

python scripts/run_pwm_rosetta_af3.py --out results/zf_struct/pwm_rosetta_af3
echo "Done at $(date)"
