#!/usr/bin/env bash
# END-TO-END head-swap: v24 recipe (LoRA-ESM + residue-MoE + span gate + N-chain),
# but PWM head = RecognitionEnergyHead instead of PWMHeadV18 (v18-contact losses
# dropped). Only the head differs from v24 -> the fair "is the head the bottleneck" test.
set -uo pipefail
cd /afs/csail.mit.edu/u/l/leihuang/project/TFScope
source /data1/leihuang/miniconda3/etc/profile.d/conda.sh; conda activate tfscope
export TORCH_HOME=/data1/leihuang/.cache/torch PYTHONPATH=src PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=7
OUT="checkpoints/iclr_phase1/recog_e2e/seed42"; mkdir -p "$OUT"
python scripts/train.py \
  --data data/processed/tf_pwm_training_v23.parquet \
  --split data/processed/splits/train_v22/split.json --out "$OUT" \
  --seed 42 --epochs 225 --batch-size 12 --grad-accum-steps 3 \
  --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6 \
  --warmup-steps 150 --workers 2 --save-every 25 \
  --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0 \
  --pwm-head-recog --group-balanced-sampling \
  --family-embedding-path none \
  --moe-granularity residue --num-experts 8 --n-shared-experts 2 --top-k 2 \
  --expert-hidden-dim 512 --balance-loss-weight 0.01 --diversity-loss-weight 0.0 \
  --gate-length-weight 0.05 --latent-registration \
  --gate-mode span --max-motif-length 42 --motif-overflow-policy error \
  --pwm-cov-r-weight 0.25 --pwm-core-ic-thresh 0.25 \
  --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 0 \
  --oracle-aggregation gene --early-stop-patience 30 \
  --two-chain-input --chain-id-embedding --max-chains 4 \
  --precision bf16 --tf32 --no-wandb
echo "[$(date +%T)] recog_e2e DONE" | tee "$OUT/DONE"
