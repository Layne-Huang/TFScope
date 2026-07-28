#!/bin/bash
set -e
cd /afs/csail.mit.edu/u/l/leihuang/project/TFScope
FOLDX=/data1/leihuang/foldx_new/foldx_20261231
ROTA=/data1/leihuang/foldx_new/rotabase.txt
for d in 005 006 009 035; do
( W=results/foldx_dnascan_semfam34/DBP${d}; mkdir -p "$W"; cp "$ROTA" "$W/"; cp results/foldx_dnascan_semfam34/DBP${d}.pdb "$W/"
  cd "$W"
  "$FOLDX" --command=RepairPDB --pdb=DBP${d}.pdb > repair.log 2>&1
  "$FOLDX" --command=DNAScan --pdb=DBP${d}_Repair.pdb > dnascan.log 2>&1
  echo "DBP${d} done" >> ../progress.log
) &
done
wait
echo "ALL DONE $(date)" >> results/foldx_dnascan_semfam34/progress.log
