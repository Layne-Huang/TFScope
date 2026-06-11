#!/bin/bash
#SBATCH -J deeppbs_struct
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 0-04:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark/struct_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_benchmark/struct_%j.err

# Re-run DeepPBS 5-model ensemble on the 130-TF blind benchmark (id.txt),
# now saving PER-STRUCTURE predicted + target PWMs (struct_preds.npz) so the
# full metric set (CE/KL/AUC/F1/MCC) and per-TF logos can be computed without
# the gene-symbol aggregation loss that dropped 14 TFs.

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/eval_deeppbs.py \
    --npz-list /n/home13/leihuang/project/DeepPBS/run/folds/id.txt \
    --parquet  data/processed/tf_pwm.parquet \
    --out      results/deeppbs_blind_benchmark

echo "Done at $(date)"
ls -la results/deeppbs_blind_benchmark/struct_preds.npz
