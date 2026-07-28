#!/usr/bin/env python
"""Evaluate one V19 E1 checkpoint with row, gene-macro, and family metrics."""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from eval_canonical_registration import canonicalize
from eval_full_metrics import aligned_cols, canon_fixed_r, panel, trimmed_core
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.alignment import align_pwm
from tfscope.models.tfscope import TFScopeModel


METRICS = (
    "gate_r",
    "panel_r",
    "canon_r",
    "mae",
    "panel_mae",
    "fixed_mae",
    "rmse",
    "ce",
    "kl",
    "top1",
    "auc",
    "f1",
    "mcc",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="ckpt_best.pt")
    parser.add_argument(
        "--data",
        default="data/processed/tf_pwm_aug_dbd_canon_trim.parquet",
    )
    parser.add_argument(
        "--split",
        default="data/processed/splits/cluster40_clean/split.json",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--predictions-out", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ic-thresh", type=float, default=0.25)
    parser.add_argument("--min-positions", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=10)
    return parser.parse_args()


def load_config(run_dir):
    config = TFScopeConfig()
    with open(os.path.join(run_dir, "config.json")) as handle:
        saved = json.load(handle)
    for key, value in saved.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        try:
            setattr(config, key, type(current)(value))
        except (TypeError, ValueError):
            setattr(config, key, value)
    return config


def informative_core(pwm, threshold):
    pwm = np.clip(pwm, 1e-8, 1.0)
    ic = 2.0 + (pwm * np.log2(pwm)).sum(axis=0)
    positions = np.where(ic >= threshold)[0]
    if len(positions) == 0:
        return pwm
    return pwm[:, positions[0] : positions[-1] + 1]


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def aggregate(rows, group_key=None):
    if group_key is None:
        return {metric: finite_mean([row[metric] for row in rows]) for metric in METRICS}

    grouped = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)
    group_metrics = {
        group: {
            metric: finite_mean([row[metric] for row in group_rows])
            for metric in METRICS
        }
        for group, group_rows in grouped.items()
    }
    macro = {
        metric: finite_mean(
            [
                values[metric]
                for values in group_metrics.values()
                if values[metric] is not None
            ]
        )
        for metric in METRICS
    }
    return macro, group_metrics


def fixed_deeppbs_mae(prediction, target):
    """Fixed-frame mean position-wise L1, matching DeepPBS's reported MAE."""
    prediction = np.clip(prediction, 1e-8, 1.0)
    prediction = prediction / prediction.sum(axis=0, keepdims=True)
    prediction = canonicalize(prediction.astype(np.float32))
    prediction = prediction / prediction.sum(axis=0, keepdims=True).clip(1e-8)

    target = np.clip(target, 1e-8, 1.0)
    target = target / target.sum(axis=0, keepdims=True)
    fixed = np.full_like(target, 0.25)
    length = min(prediction.shape[1], target.shape[1])
    fixed[:, :length] = prediction[:, :length]
    return float(np.abs(fixed - target).sum(axis=0).mean())


def describe_lengths(values):
    values = np.asarray(values, dtype=int)
    unique, counts = np.unique(values, return_counts=True)
    return {
        "n": int(len(values)),
        "min": int(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q75": float(np.quantile(values, 0.75)),
        "max": int(values.max()),
        "histogram": {
            str(int(length)): int(count) for length, count in zip(unique, counts)
        },
    }


def main():
    args = parse_args()
    checkpoint_path = os.path.join(args.run_dir, args.checkpoint)
    output_path = args.out or os.path.join(
        args.run_dir, f"{args.eval_split}_metrics.json"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(args.run_dir)
    model = TFScopeModel(config).to(device).eval()
    if config.lora_rank > 0:
        model.backbone.build(torch.device(device))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, _ = model.load_state_dict(checkpoint["model"], strict=False)
    expected_lora = {
        key
        for key in model.state_dict()
        if key.startswith("backbone._esm_model") and ".lora_" in key
    }
    missing_lora = sorted(expected_lora.intersection(missing))
    if missing_lora:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing "
            f"{len(missing_lora)} trained LoRA tensors and cannot be "
            "evaluated as a valid trained model."
        )

    dataset = TFDataset(
        config, args.data, args.split, split=args.eval_split, max_seq_len=1024
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_variable_length,
    )

    rows = []
    exported = {
        "filename": [],
        "gene": [],
        "family": [],
        "prediction": [],
        "gate": [],
        "target": [],
        "mask": [],
    }
    row_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = batch["family_id"].shape[0]
            metadata = dataset.df.iloc[row_offset : row_offset + batch_size]
            row_offset += batch_size
            batch = {
                key: value.to(
                    device,
                    dtype=torch.float32 if value.is_floating_point() else torch.long,
                )
                for key, value in batch.items()
            }
            gate_logits, pwm_logits, _ = model(
                batch["sequence_tokens"],
                batch["dbd_mask"],
                batch["family_id"],
                retrieved_pwms=batch.get("retrieved_pwms"),
                retrieved_masks=batch.get("retrieved_masks"),
                retrieved_sims=batch.get("retrieved_sims"),
                recog_prior=batch.get("recog_prior"),
            )
            predictions = F.softmax(pwm_logits, dim=1).cpu().numpy()
            gates = torch.sigmoid(gate_logits).cpu().numpy()
            targets = batch["target_pwm"].cpu().numpy()
            masks = batch["pwm_mask"].cpu().numpy()
            for position, meta in enumerate(metadata.itertuples(index=False)):
                exported["filename"].append(str(meta.filename))
                exported["gene"].append(str(meta.gene_symbol).upper())
                exported["family"].append(str(meta.family_name))
                exported["prediction"].append(predictions[position])
                exported["gate"].append(gates[position])
                exported["target"].append(targets[position])
                exported["mask"].append(masks[position])

            for position, (prediction, gate, target, mask) in enumerate(
                zip(predictions, gates, targets, masks)
            ):
                core = trimmed_core(target, mask, args.ic_thresh)
                if core is None or core.shape[1] < args.min_positions:
                    continue

                active = gate > 0.5
                if not active.any():
                    active = gate > gate.max() * 0.5
                target_valid = target[:, mask.astype(bool)]
                target_gate_core = informative_core(target_valid, args.ic_thresh)
                _, gate_shift, gate_orientation, gate_r = align_pwm(
                    prediction[:, active],
                    target_gate_core,
                    max_shift=args.max_shift,
                    consider_revcomp=True,
                    min_overlap=args.min_positions,
                )
                gate_oriented_length = int(active.sum())
                gate_overlap = sum(
                    0 <= index + gate_shift < target_gate_core.shape[1]
                    for index in range(gate_oriented_length)
                )
                if gate_r <= -1.5:
                    gate_r = float("nan")
                    gate_overlap = 0

                prediction_panel = prediction[:, mask.astype(bool)]
                aligned, columns, _ = aligned_cols(
                    prediction_panel,
                    core,
                    args.max_shift,
                    min_overlap=args.min_positions,
                )
                panel_metrics = panel(core, aligned, columns)
                if panel_metrics is None:
                    continue

                meta = metadata.iloc[position]
                rows.append(
                    {
                        "filename": str(meta["filename"]),
                        "gene": str(meta["gene_symbol"]).upper(),
                        "family": str(meta["family_name"]),
                        "gate_r": float(gate_r),
                        "panel_r": float(panel_metrics["r"]),
                        "canon_r": float(canon_fixed_r(core, prediction_panel)),
                        "mae": float(4.0 * panel_metrics["mae"]),
                        "panel_mae": float(panel_metrics["mae"]),
                        "fixed_mae": fixed_deeppbs_mae(
                            prediction_panel, target_valid
                        ),
                        "trimmed_pwm_length": int(core.shape[1]),
                        "panel_overlap_length": int(len(columns)),
                        "gate_prediction_length": gate_oriented_length,
                        "gate_overlap_length": int(gate_overlap),
                        "gate_orientation": gate_orientation,
                        **{
                            metric: float(panel_metrics[metric])
                            for metric in METRICS
                            if metric
                            not in {
                                "gate_r",
                                "panel_r",
                                "canon_r",
                                "mae",
                                "panel_mae",
                                "fixed_mae",
                            }
                        },
                    }
                )

    row_metrics = aggregate(rows)
    gene_macro, per_gene = aggregate(rows, "gene")

    families = {}
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        family_gene_macro, _ = aggregate(family_rows, "gene")
        families[family] = {
            "n_rows": len(family_rows),
            "n_genes": len({row["gene"] for row in family_rows}),
            **family_gene_macro,
        }

    result = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "split": args.eval_split,
        "split_artifact": args.split,
        "retrieval_index": (
            config.retrieval_index_path if config.use_retrieval else None
        ),
        "use_retrieval": bool(config.use_retrieval),
        "n_rows": len(rows),
        "n_genes": len(per_gene),
        "mae_protocol": (
            "DeepPBS-scale aligned MAE: mean_positions(sum_bases(abs(pred-target))) "
            "over the shift/reverse-complement aligned informative-core overlap"
        ),
        "panel_mae_protocol": (
            "Same aligned informative-core overlap as mae, divided by four"
        ),
        "fixed_mae_protocol": (
            "DeepPBS fixed-frame diagnostic over the full valid motif window"
        ),
        "minimum_alignment_overlap": args.min_positions,
        "trimmed_pwm_length_distribution": describe_lengths(
            [row["trimmed_pwm_length"] for row in rows]
        ),
        "panel_overlap_length_distribution": describe_lengths(
            [row["panel_overlap_length"] for row in rows]
        ),
        "gate_prediction_length_distribution": describe_lengths(
            [row["gate_prediction_length"] for row in rows]
        ),
        "gate_overlap_length_distribution": describe_lengths(
            [row["gate_overlap_length"] for row in rows]
        ),
        "n_valid_gate_r": int(
            sum(np.isfinite(row["gate_r"]) for row in rows)
        ),
        "row_macro": row_metrics,
        "gene_macro": gene_macro,
        "per_family_gene_macro": families,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    if args.predictions_out:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.predictions_out)), exist_ok=True
        )
        np.savez_compressed(
            args.predictions_out,
            **{
                key: np.asarray(value)
                for key, value in exported.items()
            },
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
