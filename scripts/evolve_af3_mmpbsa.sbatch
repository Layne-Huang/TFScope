#!/bin/bash
#SBATCH -J evo_af3_mmpbsa
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/evo_af3_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/evo_af3_%j.err

# Evolve TFScope's RAG-PREDICTED PWM starting from the AF3-folded (protein + RAG-consensus DNA)
# structure, scored by MM-GBSA. Tests whether structure-based ΔΔG can refine TFScope's prediction.
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

for s in zf21 zf93; do
  echo "=== MM-GBSA AF3-fold evolve $s ==="
  python scripts/evolve_pwm_struct.py --method mmgbsa \
     --pdb results/zf_struct/${s}_rag_fold.pdb --name $s \
     --out results/zf_struct/mmpbsa_evolve_af3 --rounds 4 --tau 1.5 --min-iter 300
done
echo "Done at $(date)"
