#!/usr/bin/env bash
# Rosetta (-relax) DNA-base scan on the AF3 homodimer complexes (WT + L112R/TFScope-DNA).
set -uo pipefail
cd /afs/csail.mit.edu/u/l/leihuang/project/TFScope
MF=/data1/leihuang/miniconda3/envs/multiflow/bin/python
PDBDIR=case_study/pdb/mutation
OUT=results/pwm_rosetta
mkdir -p "$OUT/logs"

declare -A JOBS=(
  [myod1_wt_dimer]="$PDBDIR/fold_myod_wt_bhlh_model0.pdb"
  [myod1_l112r_tfscope_dimer]="$PDBDIR/fold_myod_l112r_tfscope_bhlh_model0.pdb"
)
for name in "${!JOBS[@]}"; do
  pdb="${JOBS[$name]}"; od="$OUT/${name}_protdna_relax"; mkdir -p "$od"
  echo "[launch] $name <- $pdb -> $od"
  PYTHONPATH=pwm_rosetta CUDA_VISIBLE_DEVICES="" PSIPRED_EXE="" PWM_INTERFACE_MODE=prot_dna \
    "$MF" -m pwm_hybrid.cli -pdb "$pdb" -relax -output_dir "$od" \
    > "$OUT/logs/${name}_relax.log" 2>&1 &
done
wait
echo "ALL MYOD1 DIMER PROTDNA ROSETTA SCANS DONE"
