#!/usr/bin/env python
"""Dump the retrained (PDB-disjoint) DeepPBS ensemble predictions for the 20
struct-having test genes to a pickle {gene: pred_pwm(4,L)}, so the full metric
suite can be computed in the tfscope env against v24. Run in the deeppbs env.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
sys.path.insert(0, "/afs/csail.mit.edu/u/l/leihuang/project/TFScope")
from iclr.run_deeppbs_retrained import run_deeppbs, _gene_of
import torch

REPO = Path("/data1/leihuang/DeepPBS/deeppbsmar24")
OUT_ROOT = Path("/data1/leihuang/DeepPBS/iclr_retrain_pdb")
FOLDS = "iclr_folds_pdbdisjoint"

models = [l.strip() for l in (OUT_ROOT / "model_list.txt").read_text().split() if l.strip()]
structs = [l.strip() for l in (REPO / "run" / FOLDS / "test20.txt").read_text().split() if l.strip()]
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
preds = run_deeppbs(REPO, OUT_ROOT, models, structs, dev)
by_gene = {_gene_of(s): preds[s] for s in structs if _gene_of(s)}
pickle.dump(by_gene, open("/data1/leihuang/TFScope_store/deeppbs_pdbclean_preds.pkl", "wb"))
print("dumped", len(by_gene), "gene preds ->", "/data1/leihuang/TFScope_store/deeppbs_pdbclean_preds.pkl")
