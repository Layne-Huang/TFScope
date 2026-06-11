#!/bin/bash
#SBATCH -J boltz_v10_msa
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 24:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/boltz_v10_msa_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/boltz_v10_msa_%j.err

# Boltz-2 with MSA server for all 130 v10 test TFs.
# FASTAs have |empty removed from protein chain so MSA is fetched
# from ColabFold MMseqs2 server.
# Tested locally: iPTM improves 0.932 → 0.964 for 1a1g_A_Egr1 sample.

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export BOLTZ_CACHE=/n/holylabs/lpinello_lab/Lab/leihuang/.boltz
export HF_HOME=/n/holylabs/lpinello_lab/Lab/leihuang/.cache/huggingface

set -eo pipefail

INPUT_DIR=/n/home13/leihuang/project/TFScope/results/boltz_pipeline_v10/inputs_all_msa
OUT_DIR=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/out_v10_msa
mkdir -p $OUT_DIR

echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"
nvidia-smi
echo "Input: $INPUT_DIR  ($(ls $INPUT_DIR/*.fasta | wc -l) FASTA files)"
echo "Output: $OUT_DIR"

boltz predict $INPUT_DIR \
    --out_dir $OUT_DIR \
    --cache $BOLTZ_CACHE \
    --model boltz2 \
    --output_format mmcif \
    --use_msa_server \
    --no_kernels \
    --accelerator gpu \
    --devices 1 \
    --override

echo "Boltz MSA run complete at $(date)"
echo "Structures produced: $(find $OUT_DIR -name '*.cif' | wc -l)"
