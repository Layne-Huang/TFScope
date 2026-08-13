#!/bin/bash
# Sequential FoldX evolve: design structures (ground-truth DNA) then AF3 folds (RAG-predicted DNA).
cd /n/home13/leihuang/project/TFScope
PY=/n/home13/leihuang/.conda/envs/tfscope/bin/python

for s in zf21 zf93; do
  echo "########## FoldX DESIGN evolve $s ##########"
  $PY scripts/evolve_pwm_struct.py --method foldx \
     --pdb design_pdbs/zf_samples/${s}_model.pdb --name $s \
     --out results/zf_struct/foldx_evolve --rounds 4 --tau 1.5
done

for s in zf21 zf93; do
  echo "########## FoldX AF3-FOLD evolve $s ##########"
  $PY scripts/evolve_pwm_struct.py --method foldx \
     --pdb results/zf_struct/${s}_rag_fold.pdb --name $s \
     --out results/zf_struct/foldx_evolve_af3 --rounds 4 --tau 1.5
done
echo "ALL FOLDX EVOLVE DONE $(date)"
