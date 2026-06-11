#!/bin/bash
#SBATCH -J myod_struct
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-08:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/myod_struct_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs/myod_struct_%j.err

# Fold WT and L12R MyoD + E-box via Boltz, then MM-GBSA scan both, compare PWMs.
# Tests whether the structure/physics path detects the single-residue mutation
# that the seq-only model is blind to (r=0.99 WT vs mut).

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export BOLTZ_CACHE=/n/holylabs/lpinello_lab/Lab/leihuang/.boltz
export HF_HOME=/n/holylabs/lpinello_lab/Lab/leihuang/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/mmpbsa_runs
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/myod_struct
echo "=== Boltz fold WT + L12R (MSA server) ==="
boltz predict results/myod1_struct/inputs \
    --out_dir $OUT --cache $BOLTZ_CACHE --model boltz2 \
    --output_format mmcif --use_msa_server --no_kernels \
    --accelerator gpu --devices 1 --override

echo "=== MM-GBSA scan + compare ==="
python scripts/mmpbsa_compare_driver.py $OUT 10

echo "Done at $(date)"
