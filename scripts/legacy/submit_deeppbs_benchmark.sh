#!/bin/bash
#SBATCH --job-name=tfscope_deeppbs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark/slurm_%j.out
#SBATCH --error=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark/slurm_%j.err

mkdir -p /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark

cd /n/home13/leihuang/project/TFScope

source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
conda activate tfscope

python scripts/train.py \
    --split        data/processed/splits/deeppbs/benchmark.json \
    --data         data/processed/tf_pwm.parquet \
    --out          /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark \
    --epochs       200 \
    --batch-size   128 \
    --wandb-project TFScope \
    --wandb-name   deeppbs_benchmark

