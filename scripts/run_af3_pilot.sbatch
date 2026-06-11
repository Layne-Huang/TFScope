#!/bin/bash
#SBATCH -J af3_pilot
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 08:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/pilot_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/pilot_%j.err

# AF3 pilot: data pipeline (MSA) + inference for the 3 pilot protein-DNA complexes.
# Singularity image + databases + model params per FASRC docs.

set -eo pipefail

SIF=/n/singularity_images/FAS/alphafold/alphafold_3.0.1.sif
DATA_DIR=/n/holylabs/rc_admin/Everyone/alphafold_databases/v3
MODEL_DIR=/n/holylabs/lpinello_lab/Lab/leihuang/AF3_parameter
INPUT_DIR=/n/home13/leihuang/project/TFScope/results/af3_pipeline/af3_inputs
OUTPUT_DIR=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/pilot_out

mkdir -p $OUTPUT_DIR

echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"
nvidia-smi
echo "Input JSONs:"; ls $INPUT_DIR/*.json

# Full AF3 (data pipeline + inference). --input_dir processes every JSON in the dir.
singularity exec \
    --nv \
    --bind $DATA_DIR:/data \
    --bind $MODEL_DIR:/models \
    --bind $INPUT_DIR:/input \
    --bind $OUTPUT_DIR:/output \
    $SIF \
    python /app/alphafold/run_alphafold.py \
        --input_dir=/input \
        --model_dir=/models \
        --db_dir=/data \
        --pdb_database_path=/data/mmcif_files \
        --output_dir=/output

echo "AF3 pilot complete at $(date)"
ls -R $OUTPUT_DIR | head -40
