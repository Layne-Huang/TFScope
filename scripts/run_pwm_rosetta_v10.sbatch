#!/bin/bash
#SBATCH -J pwm_rosetta_all
#SBATCH -p sapphire,shared
#SBATCH -c 4
#SBATCH --mem=16g
#SBATCH -t 03:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/aug_v10/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/aug_v10/slurm_%j.err

# Single-process pwm_rosetta calibration of all 130 Boltz CIFs.
# Per-TF cost ~ 67s (PyRosetta init amortised once, 36 mutations × ~1s each).
# Total ~ 80–120 min for the full benchmark.

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
set -eo pipefail

cd /n/home13/leihuang/project/TFScope

python scripts/run_pwm_rosetta_calibration.py \
    --af3-out-dir /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/out_v10/boltz_results_inputs_all/predictions \
    --manifest    /n/home13/leihuang/project/TFScope/results/boltz_pipeline_v10/manifest_all.json \
    --out-dir     /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/aug_v10

echo "All 130 calibrations complete at $(date)"
