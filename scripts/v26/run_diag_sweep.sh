#!/usr/bin/env bash
# v26 regression diagnosis: 4 configs x 1 seed, one per GPU, all detached.
# Single seed is appropriate here: the effect being chased is the 0.23 cov_r gap between
# v24 (0.5828) and v26 core (0.3507) on the SAME clean split -- ~20x the measured seed
# sd of 0.0087. Return to 3 seeds once differences shrink to the 0.01 scale again.
set -uo pipefail
: "${PWD:?}"
GPU_IDX=${GPU_IDX:-"1 2 4 5 6 7 8 9"}
CFGS=(loss_reg loss_v24parity reg_lronly reg_strong)
DRY=${DRY:-0}

GPUS=()
for i in $GPU_IDX; do
  u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v k="$i" '$1==k{print $2}')
  m=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v k="$i" '$1==k{print $2}')
  [ -z "$u" ] && continue
  if [ "${m:-0}" -gt 2000 ]; then echo "  skip GPU $i (${m} MiB in use)"; continue; fi
  GPUS+=("$u"); echo "  GPU $i free -> $u"
  [ ${#GPUS[@]} -ge ${#CFGS[@]} ] && break
done
if [ ${#GPUS[@]} -lt 1 ]; then echo "no free GPU"; exit 1; fi
echo "launching ${#CFGS[@]} configs on ${#GPUS[@]} GPUs"

for k in "${!CFGS[@]}"; do
  g=$(( k % ${#GPUS[@]} ))
  cfg="configs/v26/${CFGS[$k]}.yaml"
  [ "$DRY" = "1" ] && { echo "  ${CFGS[$k]} -> gpu slot $g"; continue; }
  scripts/v26/run_detached.sh --mirror "v26_diag_${CFGS[$k]}" \
      scripts/v26/run_sweep_worker.sh "${GPUS[$g]}" "$cfg:42"
done
echo "monitor: scripts/v26/sweep_status.sh"
