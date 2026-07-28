#!/bin/bash
# FoldX DNAScan on the AF3 homodimer complexes (whole-complex energy; handles multi-chain natively).
set -e
cd /afs/csail.mit.edu/u/l/leihuang/project/TFScope
FOLDX=/data1/leihuang/foldx_new/foldx_20261231
ROTA=/data1/leihuang/foldx_new/rotabase.txt
SRC=case_study/pdb/mutation
declare -A PDBS=(
  [wt]="$SRC/fold_myod_wt_bhlh_model0.pdb"
  [l112r_tfscope]="$SRC/fold_myod_l112r_tfscope_bhlh_model0.pdb"
)
OUT=results/foldx_dnascan_myod1_dimer; mkdir -p "$OUT"
for name in "${!PDBS[@]}"; do
( W="$OUT/$name"; mkdir -p "$W"; cp "$ROTA" "$W/"; cp "${PDBS[$name]}" "$W/in.pdb"
  cd "$W"
  "$FOLDX" --command=RepairPDB --pdb=in.pdb > repair.log 2>&1
  "$FOLDX" --command=DNAScan --pdb=in_Repair.pdb > dnascan.log 2>&1
  echo "$name done $(date)" >> ../progress.log
) &
done
wait
echo "ALL DONE $(date)" >> "$OUT/progress.log"
