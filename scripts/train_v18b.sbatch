#!/bin/bash
#SBATCH -J tfscope_v18b
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-08:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18b_contact/logs/train_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18b_contact/logs/train_%j.err

# =============================================================
# TFScope v18b — CONTACT-SUPERVISED head (build on v17_200ep prior)
#
# v18a repairs PLUS family-canonical recognition-residue supervision:
#   --v18-contact-supervision   pull attention marginal onto recognition residues
#   --v18-contact-weight 0.3    (cross-entropy of attn marginal vs recog prior)
# Recognition residues from scripts/build_recognition_prior.py (rule-based,
# leak-free: derived from family structural biology, NOT test PDBs).
#
# Success: attention lands on canonical recognition residues; KLF4 K409 / MyoD
# L122 receive mass>0; PWM r >= v18a; shuffled-prior negative control does not help.
# =============================================================
OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18b_contact
mkdir -p $OUT/logs
export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
export TORCH_HOME="/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch"
export PYTHONUNBUFFERED=1
set -eo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME"; nvidia-smi || true
cd /n/home13/leihuang/project/TFScope

python scripts/train.py \
    --data   data/processed/tf_pwm_deeppbs_only.parquet \
    --split  data/processed/splits/deeppbs_only/benchmark_no_val.json \
    --out    $OUT \
    --epochs 80 --batch-size 16 \
    --lr 6e-4 --lora-rank 0 \
    --warmup-steps 150 --workers 4 --save-every 20 \
    --use-retrieval --retrieval-k 3 --retrieval-dropout 0.0 \
    --retrieval-index-path data/processed/tf_nn_index_lgo_deeppbs.json \
    --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
    --pwm-head-v18 --v18-freeze-prior \
    --v18-row-div-weight 0.05 --v18-hub-weight 0.05 --v18-delta-scale-init 0.1 \
    --v18-contact-supervision --v18-contact-weight 0.3 \
    --recognition-prior-path data/contact_maps/recognition_residues.json \
    --init-from /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v17_200ep/ckpt_best.pt \
    --no-wandb

echo "v18b complete at $(date)"
