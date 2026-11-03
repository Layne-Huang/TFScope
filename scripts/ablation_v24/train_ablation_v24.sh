#!/usr/bin/env bash
# Train ONE v24 ablation variant. Everything not named by the variant is byte-for-byte
# the v24 recipe (scripts/run_v24_contact_ddp.sh), re-expressed for a single GPU:
# micro-batch 12 x accum 3 = the same global batch of 36 the 6-GPU DDP run and the
# ensemble seeds used. Identical to scripts/lofo/train_lofo_v24.sh apart from the
# variant delta, so ablation numbers are comparable to both the LOFO and the B-ladder runs.
#
#   scripts/ablation_v24/train_ablation_v24.sh <TAG> [SEED]
#
# Not meant to be run directly -- use scripts/ablation_v24/run_ablation_queue.sh, which
# pins a free GPU by UUID and detaches the job so it survives the SSH/agent session.
#
# ── variants ──────────────────────────────────────────────────────────────────────
#   A0_full          the v24 recipe verbatim. Not redundant with the shipped seed-42
#                    checkpoint: it re-runs it single-GPU under this harness so every
#                    ablation delta is measured against a same-harness reference.
#   A1_deeppbsfold   DATA ablation. Byte-identical copy of DeepPBS fold 0
#                    (train0.txt 396 + valid0.txt 81) with DeepPBS's own MASKED PWM
#                    labels (Y_pwm[0] restricted to pwm_mask[0]). Test set unchanged.
#                    "Does TFScope still win when handed exactly DeepPBS's data?"
#                    Built by scripts/ablation_v24/build_deeppbs_fold_dataset.py; see its
#                    docstring for the 1:1 NPZ->row mapping, the contact re-key, and the
#                    16 leaky test genes (score with --leak-free as well).
#   A1b_structonly   cheaper data ablation, not part of the headline study: train/val
#                    restricted to structure-backed rows (936/144) = the DeepPBS regime
#                    without the exact fold membership or the masked labels.
#   A2_nodistill     drop the 2-D contact-distillation KL (--contact-distill-weight 0).
#                    Keeps 1-D attention supervision and the learnable bias.
#   A3_noattnsup     drop the 1-D recognition-residue attention supervision.
#                    Keeps distillation and the learnable bias.
#   A4_nobias        drop the learnable contact bias at the head (scale 0, non-learnable).
#                    Keeps both supervision terms.
#   A5_shufcontact   CONTROL. All three contact terms kept, but the residue targets are
#                    permuted within each chain (same counts, same weights, wrong residues).
#                    Separates "structural knowledge" from "extra regulariser".
#   A6_nolora        freeze ESM-2 entirely (--lora-rank 0), full v24 head. Isolates encoder
#                    adaptation from the head/MoE/contact stack (B2-B4 confound the two).
#   A7_singlechain   drop the N-chain protomer input. B7 in the ICLR ladder strips MoE and
#                    the v18 head at the same time; this is the clean chain-only delta.
#
# A2+A3+A4 decompose the ICLR ladder's B6 ("v24 without contact losses/bias"), which
# removes all three at once and therefore cannot say which one carries the effect.
set -euo pipefail

TAG="${1:?usage: $0 <TAG> [SEED]}"
SEED="${2:-42}"
PYTHON_BIN="${PYTHON_BIN:-/data1/leihuang/miniconda3/envs/tfscope/bin/python}"
EPOCHS="${EPOCHS:-225}"

DATA="data/processed/tf_pwm_training_v23.parquet"
SPLIT="data/processed/splits/train_v22/split.json"
OUT="checkpoints/ablation_v24/${TAG}/seed${SEED}"

# ── composable blocks (mirror iclr/variants.py so the two ladders stay in sync) ─────
LORA=(--lora-rank 16 --lora-alpha 32 --lora-n-layers 6 --lr 4.5e-4 --lora-lr 7.5e-6)
MOE=(--moe-granularity residue --num-experts 8 --n-shared-experts 2 --top-k 2
     --expert-hidden-dim 512 --balance-loss-weight 0.01 --diversity-loss-weight 0.0
     --family-embedding-path none)
HEAD=(--pwm-head-v18)
ATTNSUP=(--v18-contact-supervision --v18-contact-weight 0.3)
BIAS=(--v18-contact-bias-scale 1.0 --v18-contact-bias-learnable 1)
DISTILL=(--contact-distill-weight 0.2)
TARGETS=(--contact-targets-path data/contact_maps/contact_targets_v23.json
         --recognition-prior-path data/contact_maps/recognition_residues_v23.json)
CHAIN=(--two-chain-input --chain-id-embedding --max-chains 4)

case "$TAG" in
  A0_full)        ;;
  A1_deeppbsfold) DATA="data/processed/tf_pwm_deeppbs_fold0.parquet"
                  SPLIT="data/processed/splits/ablation_v24/deeppbs_fold0.json"
                  # contact files re-keyed onto the dpb_* ids; without this the fold rows
                  # would get zero contact supervision and A1 would confound data with contacts
                  TARGETS=(--contact-targets-path data/contact_maps/contact_targets_deeppbsfold0.json
                           --recognition-prior-path data/contact_maps/recognition_residues_deeppbsfold0.json) ;;
  A1b_structonly) SPLIT="data/processed/splits/ablation_v24/structonly.json" ;;
  A2_nodistill)   DISTILL=(--contact-distill-weight 0.0) ;;
  A3_noattnsup)   ATTNSUP=() ;;
  A4_nobias)      BIAS=(--v18-contact-bias-scale 0.0 --v18-contact-bias-learnable 0) ;;
  A5_shufcontact) TARGETS=(--contact-targets-path data/contact_maps/contact_targets_v23_shuffled.json
                           --recognition-prior-path data/contact_maps/recognition_residues_v23_shuffled.json) ;;
  A6_nolora)      LORA=(--lora-rank 0 --lr 4.5e-4) ;;
  A7_singlechain) CHAIN=() ;;
  *) echo "unknown variant: $TAG" >&2
     echo "known: A0_full A1_deeppbsfold A1b_structonly A2_nodistill A3_noattnsup A4_nobias A5_shufcontact A6_nolora A7_singlechain" >&2
     exit 2 ;;
esac

[[ -f "$SPLIT" ]] || { echo "no such split: $SPLIT" >&2; exit 2; }
[[ -f "$DATA" ]]  || { echo "no such data: $DATA" >&2; exit 2; }
for i in "${!TARGETS[@]}"; do
  case "${TARGETS[$i]}" in --*) continue ;; esac
  [[ -f "${TARGETS[$i]}" ]] || { echo "no such contact file: ${TARGETS[$i]}" >&2; exit 2; }
done
mkdir -p "$OUT"

echo "=== v24 ABLATION | variant=${TAG} seed=${SEED} ==="
echo "split: $SPLIT"
echo "out:   $OUT"

TRAIN_ARGS=(
  --data "$DATA" --split "$SPLIT" --out "$OUT"
  --seed "$SEED" --epochs "$EPOCHS"
  --batch-size 12 --grad-accum-steps 3
  "${LORA[@]}"
  --warmup-steps 150 --workers 2 --save-every 25
  --ic-pcc-weight 0.5 --topbase-weight 0.1 --topbase-margin 2.0
  "${HEAD[@]}" --group-balanced-sampling
  ${ATTNSUP[@]+"${ATTNSUP[@]}"}
  "${BIAS[@]}"
  "${DISTILL[@]}"
  "${TARGETS[@]}"
  "${MOE[@]}"
  --gate-length-weight 0.05 --latent-registration
  --gate-mode span --max-motif-length 42 --motif-overflow-policy error
  --pwm-cov-r-weight 0.25 --pwm-core-ic-thresh 0.25
  --eval-oracle-r --oracle-r-every 5 --oracle-r-n-tfs 0
  --oracle-aggregation gene --early-stop-patience 30
  ${CHAIN[@]+"${CHAIN[@]}"}
  --precision bf16 --tf32 --no-wandb
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "$PYTHON_BIN scripts/train.py ${TRAIN_ARGS[*]}"
  exit 0
fi

"$PYTHON_BIN" - "$SPLIT" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"  train={len(s['train']):>5} rows  val={len(s['val']):>5} rows  test={len(s['test']):>5} rows")
PY

# Fail in seconds, not after model construction, if the card was mis-pinned. The dead
# GPU 0 is invisible to CUDA, so an index-based CUDA_VISIBLE_DEVICES lands one card off
# and an out-of-range one yields a silent CPU fallback that only trips much later on
# "--precision bf16 requires CUDA".
"$PYTHON_BIN" -c "
import os, sys, torch
if not torch.cuda.is_available():
    sys.exit('ABORT: no CUDA device for CUDA_VISIBLE_DEVICES=%r -- pin the GPU by UUID'
             % os.environ.get('CUDA_VISIBLE_DEVICES'))
print('  gpu: %s (%.0f GiB free)' % (torch.cuda.get_device_name(0),
      torch.cuda.mem_get_info(0)[0] / 2**30))"

set -x
exec "$PYTHON_BIN" scripts/train.py "${TRAIN_ARGS[@]}"
