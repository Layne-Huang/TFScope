python scripts/evaluate.py \
    --ckpt  /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_homeodomain_v2/ckpt_best.pt \
    --split data/processed/splits/lofo/train_sanity.json \
    --data  data/processed/tf_pwm.parquet \
    --out   results/train_sanity