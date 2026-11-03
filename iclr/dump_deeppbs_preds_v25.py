#!/usr/bin/env python
"""Dump the v25-retrained DeepPBS ensemble over the 84-row benchmark. Run in the `deeppbs` env.

Keyed by npz name, NOT by gene: the 84 benchmark rows cover 41 genes, so several genes
contribute more than one structure and a gene-keyed dict would silently drop all but one.
The scorer maps npz -> parquet filename itself and does its own gene-balancing.

The `deeppbs` env is mandatory here -- its torch 2.3.0+cu121 / PyG 2.5.0 matches the
compiled C++ extensions, which segfault under the tfscope/multiflow envs.

  /data1/leihuang/miniconda3/envs/deeppbs/bin/python iclr/dump_deeppbs_preds_v25.py
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/afs/csail.mit.edu/u/l/leihuang/project/TFScope")
from iclr.run_deeppbs_retrained import run_deeppbs  # noqa: E402

REPO = Path("/data1/leihuang/DeepPBS/deeppbsmar24")
OUT_ROOT = Path("/data1/leihuang/DeepPBS/v25_retrain")
STRUCT_LIST = Path(os.environ.get("STRUCT_LIST", OUT_ROOT / "test84_benchmark.txt"))
OUT = Path(os.environ.get("DP_OUT", "/data1/leihuang/TFScope_store/deeppbs_v25_bench84_preds.pkl"))

models = [l.strip() for l in (OUT_ROOT / "model_list.txt").read_text().split() if l.strip()]
structs = [l.strip() for l in STRUCT_LIST.read_text().split() if l.strip()]
print(f"{len(models)} models x {len(structs)} structures")

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
preds = run_deeppbs(REPO, OUT_ROOT, models, structs, dev)
assert len(preds) == len(structs), f"{len(preds)} preds for {len(structs)} structures"

OUT.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(preds, open(OUT, "wb"))
print(f"saved {OUT}  ({len(preds)} predictions, "
      f"widths {min(p.shape[1] for p in preds.values())}-{max(p.shape[1] for p in preds.values())})")
