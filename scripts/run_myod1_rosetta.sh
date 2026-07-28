#!/usr/bin/env bash
# Rosetta (-relax) DNA-base scan on the three MyoD1 complexes.
set -uo pipefail
cd /afs/csail.mit.edu/u/l/leihuang/project/TFScope
MF=/data1/leihuang/miniconda3/envs/multiflow/bin/python
PDBDIR=case_study/pdb/mutation
OUT=results/pwm_rosetta
mkdir -p "$OUT/logs"

declare -A JOBS=(
  [myod1_wt_cacctg]="$PDBDIR/myod_acacctgt_model.pdb"
  [myod1_wt_cagctg]="$PDBDIR/myod_acagctgt_model.pdb"
  [myod1_l112r]="$PDBDIR/myod1_l112r_model.pdb"
)

for name in "${!JOBS[@]}"; do
  pdb="${JOBS[$name]}"
  od="$OUT/${name}_relax"
  mkdir -p "$od"
  echo "[launch] $name  <- $pdb  -> $od"
  PYTHONPATH=pwm_rosetta CUDA_VISIBLE_DEVICES="" PSIPRED_EXE="" \
    "$MF" -m pwm_hybrid.cli -pdb "$pdb" -relax -output_dir "$od" \
    > "$OUT/logs/${name}_relax.log" 2>&1 &
done
wait
echo "ALL MYOD1 ROSETTA SCANS DONE"
