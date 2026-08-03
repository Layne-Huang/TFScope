#!/usr/bin/env python
"""Score the GENE-DISJOINT RETRAINED DeepPBS ensemble on the 20 primary-test genes
that have a co-crystal structure — the leakage-clean DeepPBS-vs-v24 comparison.

Unlike run_deeppbs_291.py (which used the PRETRAINED checkpoints in run/outputmar24
that saw all 20 test genes -> leaky 0.806), this uses the ensemble retrained on the
477 structures whose gene is NOT in the 291 test set (iclr_retrain/m0..m4). It is run
in the dedicated `deeppbs` env (torch 2.3.0+cu121, PyG 2.5.0) so the C++ extensions
match the torch ABI (fixes the earlier multiflow-env segfault).

    /data1/leihuang/miniconda3/envs/deeppbs/bin/python iclr/run_deeppbs_retrained.py \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import re
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader


def _load_unified_eval(path: Path):
    spec = importlib.util.spec_from_file_location("tfscope_unified_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode_pwm(raw) -> np.ndarray:
    arr = raw.astype(np.float32, copy=False) if isinstance(raw, np.ndarray) \
        else np.frombuffer(raw, dtype=np.float32)
    return arr.reshape(4, -1).astype(np.float32, copy=False)


def _norm_pwm(pwm: np.ndarray) -> np.ndarray:
    arr = np.asarray(pwm, dtype=np.float32)
    if arr.shape[0] != 4 and arr.shape[1] == 4:
        arr = arr.T
    arr = np.clip(arr, 0.0, None)
    sums = arr.sum(axis=0, keepdims=True)
    return np.divide(arr, sums, out=np.full_like(arr, 0.25), where=sums > 0).astype(np.float32)


def _gene_of(name: str) -> str | None:
    m = re.search(r"_([A-Z0-9]+)_(HUMAN|MOUSE|RAT)", name)
    return m.group(1) if m else None


def _gene_balanced_mean(items, key):
    by_gene: dict[str, list[float]] = {}
    for it in items:
        by_gene.setdefault(str(it["gene"]), []).append(float(it[key]))
    gm = {g: float(np.nanmean(v)) for g, v in sorted(by_gene.items())}
    return float(np.nanmean(list(gm.values()))), gm


def run_deeppbs(deep_repo: Path, out_root: Path, model_names, structure_names, device):
    sys.path.insert(0, str(deep_repo))
    sys.path.insert(0, str(deep_repo / "run"))
    from deeppbs.nn import processBatch
    from deeppbs.nn.utils import loadDataset
    from models.model_v2 import Model

    config = json.loads((out_root / model_names[0] / "config.json").read_text())
    config["data_dir"] = str(deep_repo / "data/assembly2024")

    batches_by_model, models = [], []
    for mn in model_names:
        scaler = pickle.load(open(out_root / mn / "scaler.pkl", "rb"))
        dataset, _, _, _ = loadDataset(
            structure_names, config["nc"], config["labels_key"], config["data_dir"],
            cache_dataset=False, balance=config.get("balance", "unmasked"),
            remove_mask=False, scale=True, scaler=scaler, pre_transform=None, feature_mask=None,
        )
        batches_by_model.append(list(DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)))
        model = Model(13, 14, condition=config["condition"])
        state = torch.load(out_root / mn / "Model.best.tar", map_location=device)
        model.load_state_dict(state["model_state_dict"])
        models.append(model.to(device).eval())

    preds = {}
    for i, structure in enumerate(structure_names):
        outs = []
        for dl_idx, model in enumerate(models):
            batch = processBatch(device, batches_by_model[dl_idx][i])
            with torch.no_grad():
                logits = model(batch["batch"])
                outs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        avg = reduce(lambda a, b: a + b, outs) / len(outs)
        half = avg.shape[0] // 2
        pred_l4 = (avg[:half, :] + np.flip(avg[half:, :])) / 2.0   # symmetrize strands
        preds[structure] = _norm_pwm(pred_l4)
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfscope", type=Path, default=Path("/afs/csail.mit.edu/u/l/leihuang/project/TFScope"))
    ap.add_argument("--deep-repo", type=Path, default=Path("/data1/leihuang/DeepPBS/deeppbsmar24"))
    ap.add_argument("--out-root", type=Path, default=Path("/data1/leihuang/DeepPBS/iclr_retrain"))
    ap.add_argument("--out", type=Path, default=Path("results/iclr_phase1_apples_to_apples/deeppbs_291_retrained.json"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tfscope = args.tfscope.resolve()
    unified = _load_unified_eval(tfscope / "iclr/unified_eval.py")
    device = torch.device(args.device)

    model_names = [l.strip() for l in (args.out_root / "model_list.txt").read_text().splitlines() if l.strip()]
    test_structs = [l.strip() for l in (args.out_root.parent / "deeppbsmar24/run/iclr_folds/test20.txt").read_text().splitlines() if l.strip()]
    struct_gene = {s: _gene_of(s) for s in test_structs}

    # TFScope ground-truth PWMs for the 291 test set, keyed by gene
    split = json.loads((tfscope / "data/processed/splits/train_v22/split.json").read_text())
    df = pd.read_parquet(tfscope / "data/processed/tf_pwm_training_v23.parquet")
    test_df = df[df["filename"].isin(set(split["test"]))].copy()
    test_df["G"] = test_df["gene_symbol"].astype(str).str.upper()

    preds = run_deeppbs(args.deep_repo.resolve(), args.out_root, model_names, test_structs, device)
    pred_by_gene = {struct_gene[s]: preds[s] for s in test_structs if struct_gene[s]}

    per_sample = []
    for _, row in test_df[test_df["G"].isin(pred_by_gene)].iterrows():
        gt = _decode_pwm(row["pwm"])
        core = unified.trimmed_core(gt)
        if core is None:
            continue
        pred = pred_by_gene[row["G"]]
        pa = unified.panel_A(pred, core)
        pb = unified.panel_B(pred, core, pred.shape[1])
        per_sample.append({"filename": str(row["filename"]), "gene": row["G"],
                           "A_content_r": float(pa["content_r"]), "B_covR": float(pb["covR"])})

    content_mean, content_by_gene = _gene_balanced_mean(per_sample, "A_content_r")
    covr_mean, covr_by_gene = _gene_balanced_mean(per_sample, "B_covR")

    # v24 on the SAME genes (from B8_v24 per_sample)
    um = json.loads((tfscope / "results/iclr_phase1_apples_to_apples/unified_models.json").read_text())
    v24_s = [{"gene": str(s["gene"]).upper(), "A_content_r": s["A_content_r"], "B_covR": s["B_covR"]}
             for s in um["B8_v24"]["per_sample"] if str(s["gene"]).upper() in content_by_gene]
    v24_content, _ = _gene_balanced_mean(v24_s, "A_content_r")
    v24_covr, _ = _gene_balanced_mean(v24_s, "B_covR")

    payload = {
        "status": "ok",
        "protocol": "gene-disjoint retrain (test20 genes removed from training); 5-model ensemble; "
                    "dedicated deeppbs env torch2.3.0+cu121/PyG2.5.0",
        "n_genes_scored": len(content_by_gene),
        "n_rows_scored": len(per_sample),
        "DeepPBS_retrained": {"gene_content_r": content_mean, "gene_covR": covr_mean},
        "v24_same_genes": {"gene_content_r": v24_content, "gene_covR": v24_covr},
        "leaky_pretrained_reference": {"gene_covR_or_r": 0.806,
            "note": "run_deeppbs_291.py pretrained ckpts saw all 20 test genes -> not valid"},
        "per_gene": [{"gene": g, "DeepPBS_content_r": content_by_gene[g], "DeepPBS_covR": covr_by_gene[g]}
                     for g in sorted(content_by_gene)],
    }
    outp = args.out if args.out.is_absolute() else tfscope / args.out
    outp.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ["n_genes_scored", "DeepPBS_retrained", "v24_same_genes"]}, indent=2))
    print("wrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
