#!/usr/bin/env bash
# Launch the v26 Phase-5 ablation sweep: one detached worker per GPU, jobs round-robin.
#
# Schedule matches v24 for comparability: 225 epochs, early-stop patience 30.
# Measured cost: ~0.875 s/step, 405 micro-steps per epoch (one true pass over 3,242 train
# examples at batch 8 x accum 2) => ~5.9 min/epoch => ~23 h per run at 225 epochs.
#
# STAGE 1 (default) -- the two comparisons most likely to change the architecture:
#   core vs core_nomoe            does the sequence-conditioned MoE earn its parameters?
#   context vs context_shuffled   does flank CONTENT matter, or only extra length?
#   prior_only                    does the contact-correction head help at all?
#   5 configs x 3 seeds = 15 runs, 3 per GPU  =>  ~69 h (~3 days) wall clock
#
# STAGE 2 -- assembly, full model, and the v24-style 2-D-contact diagnostic:
#   complex_primary, complex_partners, full, context32, v24style_contact2d
#
#   scripts/v26/run_sweep.sh              # stage 1
#   STAGE=2 scripts/v26/run_sweep.sh     # stage 2
#   DRY=1 scripts/v26/run_sweep.sh       # print the plan, launch nothing
set -uo pipefail
: "${PWD:?}"
# Pin by UUID, resolved at launch. Index pinning is unsafe here: CUDA's default enumeration is
# "fastest first", not nvidia-smi order, so CUDA_VISIBLE_DEVICES=4 landed on physical GPU 3.
GPU_IDX=${GPU_IDX:-"1 2 4 5 9"}    # GPU0 is bad on this node; verify the rest are free first
GPUS=()
for i in $GPU_IDX; do
  u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v k="$i" '$1==k{print $2}')
  m=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v k="$i" '$1==k{print $2}')
  if [ -z "$u" ]; then echo "  skip GPU $i: not found"; continue; fi
  if [ "${m:-0}" -gt 2000 ]; then echo "  skip GPU $i: ${m} MiB already in use"; continue; fi
  GPUS+=("$u"); echo "  GPU $i -> $u (${m} MiB used)"
done
if [ ${#GPUS[@]} -eq 0 ]; then echo "no free GPUs"; exit 1; fi
SEEDS=(42 1 7)
STAGE=${STAGE:-1}
DRY=${DRY:-0}

if [ "$STAGE" = "1" ]; then
  CFGS=(core core_nomoe context context_shuffled prior_only)
else
  CFGS=(complex_primary complex_partners full context32 v24style_contact2d)
fi

JOBS=()
for c in "${CFGS[@]}"; do
  for s in "${SEEDS[@]}"; do JOBS+=("configs/v26/$c.yaml:$s"); done
done
per=$(( (${#JOBS[@]} + ${#GPUS[@]} - 1) / ${#GPUS[@]} ))
echo "stage $STAGE: ${#JOBS[@]} runs over ${#GPUS[@]} GPUs = ${per} per GPU"
echo "estimated wall clock: ~$(( per * 23 )) h at 225 epochs/run"

for g in "${!GPUS[@]}"; do
  QUEUE=()
  for j in "${!JOBS[@]}"; do
    if [ $(( j % ${#GPUS[@]} )) -eq "$g" ]; then QUEUE+=("${JOBS[$j]}"); fi
  done
  [ ${#QUEUE[@]} -eq 0 ] && continue
  echo "  GPU ${GPUS[$g]}: ${QUEUE[*]}"
  if [ "$DRY" != "1" ]; then
    scripts/v26/run_detached.sh --mirror "v26_sweep_s${STAGE}_gpu${GPUS[$g]}" \
        scripts/v26/run_sweep_worker.sh "${GPUS[$g]}" "${QUEUE[@]}"
  fi
done
if [ "$DRY" = "1" ]; then
  echo "DRY RUN -- nothing launched."
else
  echo "launched. monitor with: scripts/v26/sweep_status.sh"
fi
