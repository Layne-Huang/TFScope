#!/bin/bash
#SBATCH --job-name=tfscope_eval115
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark/eval_epoch115_%j.out
#SBATCH --error=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark/eval_epoch115_%j.err

cd /n/home13/leihuang/project/TFScope

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
conda activate tfscope

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODENAME"
echo "GPU:     $CUDA_VISIBLE_DEVICES"

python scripts/evaluate.py \
    --ckpt  /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark/ckpt_epoch115.pt \
    --split data/processed/splits/deeppbs/benchmark.json \
    --data  data/processed/tf_pwm.parquet \
    --out   results/deeppbs_benchmark_epoch115 \
    --deeppbs-dir results/deeppbs_eval

echo "Evaluation complete at $(date)"
