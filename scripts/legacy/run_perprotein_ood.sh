#!/bin/bash
cd /afs/csail.mit.edu/u/l/leihuang/project/TFScope
export CUDA_VISIBLE_DEVICES=0
PY=/data1/leihuang/miniconda3/envs/tfscope/bin/python
OUT=results/design_case_study/perprotein_ood_RESULTS.txt
ckep=$($PY -c "import torch;print(torch.load('/data1/leihuang/project/TFScope/checkpoints/v19_combined_perprotein_text/rag_seed42/ckpt_best.pt',map_location='cpu',weights_only=False)['epoch'])")
echo "=== per-protein-text (ckpt_best epoch $ckep) OOD tests — started $(date) ===" > $OUT
echo "" >> $OUT
echo "############## 4 DESIGNS (homolog / generic-HD / HD-centroid; core-r + CAC) ##############" >> $OUT
$PY scripts/eval_perprotein_familytext_designs.py >> $OUT 2>&1
echo "" >> $OUT
echo "############## MUTATION: MyoD1 WT vs L112R (combined / rag_contact / per-protein-text) ##############" >> $OUT
$PY scripts/eval_two_versions_mutation.py >> $OUT 2>&1
echo "" >> $OUT
echo "=== DONE $(date) ===" >> $OUT
