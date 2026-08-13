#!/usr/bin/env python
"""Evaluate one checkpoint with row/gene-balanced and length-boundary diagnostics."""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from eval_full_metrics import aligned_cols, panel_full, trimmed_core
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel


def load_model(checkpoint, device):
    directory = os.path.dirname(checkpoint)
    cfg = TFScopeConfig()
    with open(os.path.join(directory, "config.json")) as handle:
        for key, value in json.load(handle).items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    model = TFScopeModel(cfg).to(device).eval()
    # The ESM backbone is lazy; materialize it before loading LoRA parameters.
    if cfg.lora_rank > 0 and model.backbone._esm_model is None:
        model.backbone.build(torch.device(device))
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    unexpected = [key for key in unexpected if "._esm_model." not in key]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    non_esm_missing = [key for key in missing if "._esm_model." not in key]
    if non_esm_missing:
        raise RuntimeError(f"Missing non-ESM checkpoint keys: {non_esm_missing}")
    return model, cfg


def score(pred, core):
    aligned, cols, _ = aligned_cols(pred, core)
    return panel_full(core, aligned, cols, pred_ncols=pred.shape[1])


def oracle_window(pred, core):
    length = min(core.shape[1], pred.shape[1])
    candidates = [
        pred[:, start:start + length]
        for start in range(pred.shape[1] - length + 1)
    ]
    return max(candidates, key=lambda value: score(value, core)["r_cov"])


def aggregate(rows, key="predicted_gate"):
    values = [row[key]["r_cov"] for row in rows]
    by_gene = {}
    for row in rows:
        by_gene.setdefault(row["gene"], []).append(row[key]["r_cov"])
    return {
        "row_mean": float(np.mean(values)),
        "gene_balanced_mean": float(
            np.mean([np.mean(value) for value in by_gene.values()])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    dataset = TFDataset(cfg, args.data, args.split, args.split_name)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
        collate_fn=collate_variable_length,
    )
    rows = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            tensors = {
                key: value.to(
                    args.device,
                    dtype=torch.float32 if value.is_floating_point() else torch.long,
                )
                for key, value in batch.items()
            }
            gate, pwm, _ = model(
                tensors["sequence_tokens"],
                tensors["dbd_mask"],
                tensors["family_id"],
                retrieved_pwms=tensors.get("retrieved_pwms"),
                retrieved_masks=tensors.get("retrieved_masks"),
                retrieved_sims=tensors.get("retrieved_sims"),
                recog_prior=tensors.get("recog_prior"),
            )
            probs = F.softmax(pwm, dim=1).cpu().numpy()
            gates = gate.sigmoid().cpu().numpy()
            targets = tensors["target_pwm"].cpu().numpy()
            masks = tensors["pwm_mask"].cpu().numpy()
            for index in range(len(probs)):
                meta = dataset.df.iloc[offset + index]
                core = trimmed_core(targets[index], masks[index], 0.25)
                if core is None:
                    continue
                active = gates[index] > 0.5
                if not active.any():
                    active[np.argmax(gates[index])] = True
                pred_gate = probs[index][:, active]
                gt_length = min(core.shape[1], probs[index].shape[1])
                variants = {
                    "predicted_gate": pred_gate,
                    "gt_span": probs[index][:, :gt_length],
                    "fixed_10bp": probs[index][:, :min(10, probs[index].shape[1])],
                    "oracle_length_window": oracle_window(probs[index], core),
                }
                rows.append(
                    {
                        "filename": str(meta["filename"]),
                        "gene": str(meta.get("gene_symbol", "")).upper(),
                        "family": str(meta.get("family_name", "unknown")),
                        "target_length": int(core.shape[1]),
                        "multichain_eligible": bool(
                            meta.get("multichain_eligible", False)
                        ),
                        **{key: score(value, core) for key, value in variants.items()},
                    }
                )
            offset += len(probs)

    payload = {
        "n": len(rows),
        "metrics": {
            key: aggregate(rows, key)
            for key in (
                "predicted_gate", "gt_span", "fixed_10bp", "oracle_length_window"
            )
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else value,
        )
    print(json.dumps(payload["metrics"], indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
