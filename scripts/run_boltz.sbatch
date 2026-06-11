#!/bin/bash
#SBATCH -J boltz_tf
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/boltz_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/boltz_%j.err

# Boltz-2 structure prediction for protein-DNA complexes (faster AF3 alternative).
# Cache (weights + CCD) lives on holylabs per lab convention.

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export BOLTZ_CACHE=/n/holylabs/lpinello_lab/Lab/leihuang/.boltz
export HF_HOME=/n/holylabs/lpinello_lab/Lab/leihuang/.cache/huggingface

set -eo pipefail

INPUT_DIR=${1:-/n/home13/leihuang/project/TFScope/results/boltz_pipeline/inputs}
OUT_DIR=${2:-/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/out}
mkdir -p $OUT_DIR

echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"
nvidia-smi
echo "Input dir: $INPUT_DIR  ($(ls $INPUT_DIR/*.fasta 2>/dev/null | wc -l) fasta files)"

# Boltz processes a directory of FASTA files. --use_msa_server fetches protein MSAs
# from the remote MMseqs2 server (fast). Output as mmCIF for pwm_rosetta.
boltz predict $INPUT_DIR \
    --out_dir $OUT_DIR \
    --cache $BOLTZ_CACHE \
    --model boltz2 \
    --output_format mmcif \
    --no_kernels \
    --accelerator gpu \
    --devices 1 \
    --override

echo "Boltz complete at $(date)"
find $OUT_DIR -name "*.cif" | head -20
