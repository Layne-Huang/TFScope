#!/bin/bash
#SBATCH -J fx_af3_92129
#SBATCH -p sapphire
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/foldx_runs/fx_af3_92129_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/foldx_runs/fx_af3_92129_%j.err

# Single-shot FoldX AnalyseComplex scan of the AF3-folded (protein + RAG-consensus DNA)
# structures for zf92/zf129 -> round0 PWM (deployable mode). --rounds 1 -> round0_pwm.npy.
mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/foldx_runs
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"
cd /n/home13/leihuang/project/TFScope

for s in zf92 zf129; do
  echo "=== FoldX AF3-fold scan $s ==="
  python scripts/evolve_pwm_struct.py --method foldx \
     --pdb results/zf_struct/${s}_rag_fold.pdb --name $s \
     --out results/zf_struct/foldx_evolve_af3 --rounds 1 --tau 1.5
done
echo "Done at $(date)"
