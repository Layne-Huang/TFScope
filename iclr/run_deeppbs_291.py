#!/usr/bin/env python
"""Run DeepPBS on the v24 291-row primary benchmark structure subset."""

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
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import unified_eval from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode_pwm(raw) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        arr = raw.astype(np.float32, copy=False)
    else:
        arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f"PWM byte length is not divisible by 4: {arr.size}")
    return arr.reshape(4, -1).astype(np.float32, copy=False)


def _norm_pwm(pwm: np.ndarray) -> np.ndarray:
    arr = np.asarray(pwm, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D PWM, got shape {arr.shape}")
    if arr.shape[0] != 4 and arr.shape[1] == 4:
        arr = arr.T
    if arr.shape[0] != 4:
        raise ValueError(f"Cannot orient PWM to (4,L), got shape {arr.shape}")
    arr = np.clip(arr, 0.0, None)
    sums = arr.sum(axis=0, keepdims=True)
    arr = np.divide(arr, sums, out=np.full_like(arr, 0.25), where=sums > 0)
    return arr.astype(np.float32, copy=False)


def _gene_from_npz_name(name: str) -> str | None:
    for token in Path(name).stem.split("_"):
        if token in {"HUMAN", "MOUSE", "RAT"}:
            break
        if re.match(r"^[A-Z][A-Z0-9]+$", token):
            return token
    return None


def _gene_balanced_mean(items: list[dict], key: str) -> tuple[float, dict[str, float]]:
    by_gene: dict[str, list[float]] = {}
    for item in items:
        by_gene.setdefault(str(item["gene"]), []).append(float(item[key]))
    gene_means = {g: float(np.nanmean(v)) for g, v in sorted(by_gene.items())}
    return float(np.nanmean(list(gene_means.values()))), gene_means


def run_deeppbs(
    deep_repo: Path,
    structure_names: list[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(deep_repo))
    sys.path.insert(0, str(deep_repo / "run"))
    from deeppbs.nn import processBatch
    from deeppbs.nn.utils import loadDataset
    from models.model_v2 import Model

    model_list = [
        line.strip()
        for line in (deep_repo / "run/plot_scripts/txts/DeepPBSmar24.txt").read_text().splitlines()
        if line.strip()
    ]
    config_path = deep_repo / "run/outputmar24" / model_list[0] / "config.json"
    config = json.loads(config_path.read_text())
    config["data_dir"] = str(deep_repo / "data/assembly2024")
    config.setdefault("balance", "unmasked")
    config.setdefault("cache_dataset", False)
    config.setdefault("labels_key", "Y_pwm")
    config.setdefault("nc", 4)

    batches_by_model = []
    for model_name in model_list:
        scaler_path = deep_repo / "run/outputmar24" / model_name / "scaler.pkl"
        scaler = pickle.load(open(scaler_path, "rb"))
        dataset, _, _, loaded_files = loadDataset(
            structure_names,
            config["nc"],
            config["labels_key"],
            config["data_dir"],
            cache_dataset=False,
            balance=config["balance"],
            remove_mask=False,
            scale=True,
            scaler=scaler,
            pre_transform=None,
            feature_mask=None,
        )
        batches_by_model.append(list(DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)))

    models = []
    for model_name in model_list:
        checkpoint = deep_repo / "run/outputmar24" / model_name / "Model.best.tar"
        model = Model(13, 14, condition=config["condition"])
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    preds = {}
    for data_idx, structure in enumerate(structure_names):
        outputs = []
        for dl_idx, model in enumerate(models):
            batch = batches_by_model[dl_idx][data_idx]
            batch_data = processBatch(device, batch)
            with torch.no_grad():
                logits = model(batch_data["batch"])
                outputs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        avg = reduce(lambda x, y: x + y, outputs) / len(outputs)
        half = avg.shape[0] // 2
        pred_l4 = (avg[:half, :] + np.flip(avg[half:, :])) / 2.0
        preds[structure] = _norm_pwm(pred_l4)
    return preds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfscope", type=Path, default=Path("/afs/csail.mit.edu/u/l/leihuang/project/TFScope"))
    parser.add_argument("--deep-repo", type=Path, default=Path("/data1/leihuang/DeepPBS/deeppbsmar24"))
    parser.add_argument("--out", type=Path, default=Path("results/iclr_phase1_apples_to_apples/deeppbs_291.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tfscope = args.tfscope.resolve()
    deep_repo = args.deep_repo.resolve()
    out_path = args.out if args.out.is_absolute() else tfscope / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    unified = _load_unified_eval(tfscope / "iclr/unified_eval.py")

    split = json.loads((tfscope / "data/processed/splits/train_v22/split.json").read_text())
    df = pd.read_parquet(tfscope / "data/processed/tf_pwm_training_v23.parquet")
    test_df = df[df["filename"].isin(set(split["test"]))].copy()

    structures_by_gene: dict[str, list[str]] = {}
    for path in sorted((deep_repo / "data/assembly2024").glob("*.npz")):
        gene = _gene_from_npz_name(path.name)
        if gene:
            structures_by_gene.setdefault(gene, []).append(path.name)

    test_genes = sorted(str(g) for g in test_df["gene_symbol"].unique())
    covered = {
        gene: sorted(structures_by_gene[gene])[0]
        for gene in test_genes
        if gene in structures_by_gene
    }
    skipped = [
        {"gene": gene, "reason": "no assembly2024 npz structure matched by gene token"}
        for gene in test_genes
        if gene not in covered
    ]

    if not covered:
        payload = {
            "status": "blocked",
            "reason": "No 291-row primary benchmark test genes matched assembly2024 structures.",
            "per_gene": [],
            "summary": {"n_genes_scored": 0, "gene_content_r": None, "gene_covR": None},
            "skipped_genes": skipped,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        return 2

    device = torch.device(args.device)
    structure_preds = run_deeppbs(deep_repo, list(covered.values()), device)
    pred_by_gene = {gene: structure_preds[structure] for gene, structure in covered.items()}

    per_sample = []
    for _, row in test_df[test_df["gene_symbol"].isin(covered)].iterrows():
        gene = str(row["gene_symbol"])
        gt = _decode_pwm(row["pwm"])
        core = unified.trimmed_core(gt)
        if core is None:
            continue
        pred = pred_by_gene[gene]
        panel_a = unified.panel_A(pred, core)
        panel_b = unified.panel_B(pred, core, pred.shape[1])
        per_sample.append(
            {
                "filename": str(row["filename"]),
                "gene": gene,
                "structure_used": covered[gene],
                "A_content_r": float(panel_a["content_r"]),
                "B_covR": float(panel_b["covR"]),
            }
        )

    content_mean, content_by_gene = _gene_balanced_mean(per_sample, "A_content_r")
    covr_mean, covr_by_gene = _gene_balanced_mean(per_sample, "B_covR")
    per_gene = [
        {
            "gene": gene,
            "structure_used": covered[gene],
            "panelA_content_r": content_by_gene[gene],
            "panelB_covR": covr_by_gene[gene],
        }
        for gene in sorted(content_by_gene)
    ]

    unified_models_path = tfscope / "results/iclr_phase1_apples_to_apples/unified_models.json"
    unified_models = json.loads(unified_models_path.read_text())
    v24_samples = [
        {
            "gene": str(s["gene"]),
            "A_content_r": float(s["A_content_r"]),
            "B_covR": float(s["B_covR"]),
        }
        for s in unified_models["B8_v24"]["per_sample"]
        if str(s["gene"]) in content_by_gene
    ]
    v24_content, _ = _gene_balanced_mean(v24_samples, "A_content_r")
    v24_covr, _ = _gene_balanced_mean(v24_samples, "B_covR")

    payload = {
        "status": "ok",
        "method": {
            "deeppbs_repo": str(deep_repo),
            "model_list": "run/plot_scripts/txts/DeepPBSmar24.txt",
            "checkpoint": "Model.best.tar",
            "ensemble_n": 5,
            "device": str(device),
        },
        "per_gene": per_gene,
        "summary": {
            "n_genes_scored": len(per_gene),
            "n_rows_scored": len(per_sample),
            "gene_content_r": content_mean,
            "gene_covR": covr_mean,
        },
        "matched_subset_comparison": {
            "genes": sorted(content_by_gene),
            "n_genes": len(content_by_gene),
            "DeepPBS": {
                "gene_content_r": content_mean,
                "gene_covR": covr_mean,
            },
            "B8_v24": {
                "gene_content_r": v24_content,
                "gene_covR": v24_covr,
                "source": str(unified_models_path),
            },
        },
        "skipped_genes": skipped,
        "structure_candidates_count": {
            gene: len(structures_by_gene.get(gene, [])) for gene in test_genes
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(json.dumps(payload["matched_subset_comparison"], indent=2, sort_keys=True))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
