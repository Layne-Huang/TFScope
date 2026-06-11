#!/bin/bash
#SBATCH -J boltz_rela_msa
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 02:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/rela_af3_msa/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/rela_af3_msa/slurm_%j.err

# Run Boltz-2 on RELA with full ColabFold MSA (vs. the earlier |empty single-sequence run).
# MSA is fetched from https://api.colabfold.com — needs outbound internet on the compute node.
# Only the protein chain gets an MSA; DNA chains are never MSA-searched.

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
set -eo pipefail

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/rela_af3_msa
mkdir -p "$OUT"

CACHE=/n/holylabs/lpinello_lab/Lab/leihuang/.cache

boltz predict \
    /n/home13/leihuang/project/TFScope/results/boltz_rela_msa/rela_msa.fasta \
    --out_dir        "$OUT" \
    --cache          "$CACHE" \
    --model          boltz2 \
    --output_format  mmcif \
    --use_msa_server \
    --msa_server_url https://api.colabfold.com \
    --msa_pairing_strategy greedy \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --no_kernels

echo "Done at $(date)"
echo "CIF output:"
find "$OUT" -name "*.cif" | sort
