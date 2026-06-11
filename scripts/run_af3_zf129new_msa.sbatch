#!/bin/bash
#SBATCH -J af3_zf129new
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 00:30:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/zf129new_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/zf129new_%j.err

# Generate the AF3 MSA (data.json) for the CORRECTED zf129 protein via AF3_single
# (--mmseqs2). We only need the resulting *_data.json (protein MSA) to clone for all
# zinc folds; the single fold itself is a free byproduct.
set -eo pipefail
source /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3tools_venv/bin/activate
cd /n/home13/leihuang/project/TFScope
OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/zf129_new
mkdir -p "$OUT"
Z129="ERPFQCRICMRNFSTAAVLTRHIRTHTGEKPFQCRICMRNFSDRANLTRHIRTHTGEKPFQCRICMRNFSRIDKLGDHIRTHTGLR"
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
python AF3_single.py --protein-seq "$Z129" --dna-seq "GGGAGGGGGCGC" \
    --job-name "zf129new" --output-dir "$OUT"
echo "data.json:"; ls -la $OUT/zf129new/zf129new_data.json
echo "Done at $(date)"
