#!/usr/bin/env bash
# Ordered v22 ablation: repaired data -> span gate -> covR loss -> controlled multichain.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
DATA="${DATA:-data/processed/tf_pwm_training_v22.parquet}"
SPLIT="${SPLIT:-data/processed/splits/train_v22/split.json}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v22_ablation}"
SEEDS="${SEEDS:-42 43 44}"
STAGES="${STAGES:-data span loss multichain}"
EPOCHS="${EPOCHS:-225}"
BATCH_SIZE="${BATCH_SIZE:-12}"
ACCUM="${ACCUM:-3}"

for stage in $STAGES; do
  for seed in $SEEDS; do
    gate_mode=independent
    max_motif=20
    overflow=warn
    covr_weight=0.0
    chain_args=()
    case "$stage" in
      data) ;;
      span)
        gate_mode=span; max_motif=42; overflow=error ;;
      loss)
        gate_mode=span; max_motif=42; overflow=error; covr_weight=0.25 ;;
      multichain)
        gate_mode=span; max_motif=42; overflow=error; covr_weight=0.25
        chain_args=(--two-chain-input --chain-id-embedding) ;;
      *) echo "unknown stage: $stage" >&2; exit 2 ;;
    esac

    out="$OUT_ROOT/${stage}_seed${seed}"
    command=(
      "$PYTHON_BIN" scripts/train.py
      --data "$DATA" --split "$SPLIT" --out "$out"
      --seed "$seed" --epochs "$EPOCHS"
      --batch-size "$BATCH_SIZE" --grad-accum-steps "$ACCUM"
      --lr 4.5e-4 --lora-lr 7.5e-6 --lora-rank 16 --lora-alpha 32 --lora-n-layers 6
      --warmup-steps 150 --workers 2 --save-every 25
      --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0
      --pwm-head-v18 --group-balanced-sampling
      --v18-contact-supervision --v18-contact-weight 0.3 --v18-contact-bias-scale 0.0
      --recognition-prior-path data/contact_maps/recognition_residues_cluster40trainonly.json
      --family-embedding-path none
      --moe-granularity residue --num-experts 8 --n-shared-experts 2 --top-k 2
      --expert-hidden-dim 512 --balance-loss-weight 0.01 --diversity-loss-weight 0.0
      --gate-length-weight 0.05 --latent-registration
      --gate-mode "$gate_mode" --max-motif-length "$max_motif"
      --motif-overflow-policy "$overflow" --pwm-cov-r-weight "$covr_weight"
      --pwm-core-ic-thresh 0.25
      --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 0
      --oracle-aggregation gene --early-stop-patience 30
      --precision bf16 --tf32 --no-wandb
      "${chain_args[@]}"
    )
    resume_path="${RESUME_PATH:-}"
    if [[ -z "$resume_path" && "${RESUME_AUTO:-0}" == "1" ]]; then
      shopt -s nullglob
      candidates=("$out"/ckpt_epoch*.pt "$out"/ckpt_best.pt)
      shopt -u nullglob
      for candidate in "${candidates[@]}"; do
        if [[ -z "$resume_path" || "$candidate" -nt "$resume_path" ]]; then
          resume_path="$candidate"
        fi
      done
    fi
    if [[ -n "$resume_path" ]]; then
      command+=(--resume "$resume_path")
    fi

    if [[ "${SMOKE:-0}" == "1" ]]; then
      command=("$PYTHON_BIN" scripts/train.py --dummy --out "$out-smoke"
        --seed "$seed" --epochs 1 --batch-size 16 --workers 0 --no-wandb
        --num-experts 2 --expert-hidden-dim 64 --warmup-steps 1
        --gate-mode "$gate_mode" --max-motif-length "$max_motif"
        --motif-overflow-policy "$overflow" --pwm-cov-r-weight "$covr_weight"
        --eval-oracle-r --oracle-r-every 1 --oracle-r-n-tfs 0
        --oracle-aggregation gene
        "${chain_args[@]}")
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf '%q ' "${command[@]}"; printf '\n'
    else
      mkdir -p "$out"
      tee_args=()
      [[ -n "$resume_path" ]] && tee_args=(-a)
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH=src \
        TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}" \
        "${command[@]}" 2>&1 | tee "${tee_args[@]}" "$out/train.log"
      if [[ "${SMOKE:-0}" != "1" && -f "$out/ckpt_best.pt" ]]; then
        result="results/v22_ablation/${stage}_seed${seed}.json"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH=src:scripts \
          TORCH_HOME="${TORCH_HOME:-/data1/leihuang/.cache/torch}" \
          "$PYTHON_BIN" scripts/eval_v22_diagnostics.py \
          --checkpoint "$out/ckpt_best.pt" --data "$DATA" --split "$SPLIT" \
          --out "$result"
      fi
    fi
  done
done
