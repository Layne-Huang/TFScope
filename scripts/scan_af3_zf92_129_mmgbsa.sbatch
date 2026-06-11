#!/bin/bash
#SBATCH -J mg_af3_92129
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/mg_af3_92129_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/mg_af3_92129_%j.err

# Single-shot MM-GBSA scan of the AF3-folded (protein + RAG-consensus DNA) structures
# for zf92/zf129 -> round0 PWM (deployable mode). --rounds 1 produces round0_pwm.npy.
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

for s in zf92 zf129; do
  echo "=== MM-GBSA AF3-fold scan $s ==="
  python scripts/evolve_pwm_struct.py --method mmgbsa \
     --pdb results/zf_struct/${s}_rag_fold.pdb --name $s \
     --out results/zf_struct/mmpbsa_evolve_af3 --rounds 1 --tau 1.5 --min-iter 300
done
echo "Done at $(date)"
