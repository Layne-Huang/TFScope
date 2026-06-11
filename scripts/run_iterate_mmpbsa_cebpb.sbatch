#!/bin/bash
#SBATCH -J cebpb_iterate
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/cebpb_iterate_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/cebpb_iterate_%j.err

# Iterative Boltz2+MM-GBSA for CEBPB — two starting points in parallel:
#   wrong_start: TATTTAAAAATA (v10 seed consensus, wrong)
#   correct_start: TATTGCGCAATA (known CEBPB motif, correct)
# Tests whether iteration can recover from a bad seed AND whether it's stable
# from a good seed.

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export BOLTZ_CACHE=/n/holylabs/lpinello_lab/Lab/leihuang/.boltz
export HF_HOME=/n/holylabs/lpinello_lab/Lab/leihuang/.cache/huggingface
export PYTHONUNBUFFERED=1

set -eo pipefail

echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"
nvidia-smi || true

cd /n/home13/leihuang/project/TFScope

AF3=2e42_A_CEBPB.CEBPB_HUMAN.H11MO.0.A

# ── Run wrong-start and correct-start sequentially (one GPU) ──────────────────
echo ""
echo "=== EXPERIMENT A: wrong start (v10 seed consensus) ==="
python scripts/iterate_mmpbsa.py \
    --af3-name  $AF3 \
    --start-dna TATTTAAAAATA \
    --max-rounds 5 \
    --out   results/mmpbsa_iterate_cebpb \
    --label wrong_start \
    --tau 1.5 --min-iter 300

echo ""
echo "=== EXPERIMENT B: correct start (true CEBPB motif) ==="
python scripts/iterate_mmpbsa.py \
    --af3-name  $AF3 \
    --start-dna TATTGCGCAATA \
    --max-rounds 5 \
    --out   results/mmpbsa_iterate_cebpb \
    --label correct_start \
    --tau 1.5 --min-iter 300

echo ""
echo "Both experiments complete at $(date)"
