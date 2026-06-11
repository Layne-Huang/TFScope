#!/bin/bash
#SBATCH --job-name=crystal_pwm_rosetta
#SBATCH --output=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/crystal_test/slurm_%j.out
#SBATCH --error=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/crystal_test/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/crystal_test

cd /n/home13/leihuang/project/TFScope

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"

python scripts/run_pwm_rosetta_crystal.py
