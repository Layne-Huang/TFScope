#!/usr/bin/env python
"""Evaluate donor transfer-quality ranking for one V19 checkpoint."""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from evaluate_v19_e1 import load_config
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.retrieval import compute_aligned_true_trust
from tfscope.models.tfscope import TFScopeModel


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--good-threshold", type=float, default=0.75)
    return parser.parse_args()


def finite_correlation(left, right):
    if len(left) < 2 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main():
    args = parse_args()
    checkpoint_path = os.path.join(args.run_dir, args.checkpoint)
    output_path = args.out or os.path.join(
        args.run_dir, "donor_reranker_metrics.json"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.run_dir)
    model = TFScopeModel(config).to(device).eval()
    if config.lora_rank > 0:
        model.backbone.build(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    missing, _ = model.load_state_dict(checkpoint["model"], strict=False)
    missing_lora = [
        key for key in missing
        if key.startswith("backbone._esm_model") and ".lora_" in key
    ]
    if missing_lora:
        raise ValueError(
            f"Checkpoint is missing {len(missing_lora)} LoRA tensors"
        )

    dataset = TFDataset(
        config, args.data, args.split, split="test", max_seq_len=1024
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_variable_length,
    )

    predicted_values = []
    cosine_values = []
    target_values = []
    predicted_top = []
    cosine_top = []
    oracle_top = []
    pairwise_correct = 0
    pairwise_total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(
                    device,
                    dtype=torch.float32
                    if value.is_floating_point()
                    else torch.long,
                )
                for key, value in batch.items()
            }
            _, _, aux = model(
                batch["sequence_tokens"],
                batch["dbd_mask"],
                batch["family_id"],
                retrieved_pwms=batch["retrieved_pwms"],
                retrieved_masks=batch["retrieved_masks"],
                retrieved_sims=batch["retrieved_sims"],
                recog_prior=batch.get("recog_prior"),
            )
            predicted = aux["trust_logits"].sigmoid()
            target = compute_aligned_true_trust(
                batch["retrieved_pwms"],
                batch["retrieved_masks"],
                batch["target_pwm"],
                batch["pwm_mask"],
                max_shift=10,
                min_overlap=4,
            )
            valid = batch["retrieved_masks"].sum(dim=-1) > 0
            for row in range(predicted.shape[0]):
                row_valid = valid[row]
                row_predicted = predicted[row, row_valid]
                row_cosine = batch["retrieved_sims"][row, row_valid]
                row_target = target[row, row_valid]
                if not row_valid.any():
                    continue
                predicted_top.append(
                    float(row_target[row_predicted.argmax()].item())
                )
                cosine_top.append(
                    float(row_target[row_cosine.argmax()].item())
                )
                oracle_top.append(float(row_target.max().item()))
                target_difference = (
                    row_target.unsqueeze(1) - row_target.unsqueeze(0)
                )
                predicted_difference = (
                    row_predicted.unsqueeze(1) - row_predicted.unsqueeze(0)
                )
                pairs = target_difference > 0.1
                pairwise_correct += int(
                    (predicted_difference[pairs] > 0).sum().item()
                )
                pairwise_total += int(pairs.sum().item())
                predicted_values.extend(row_predicted.cpu().tolist())
                cosine_values.extend(row_cosine.cpu().tolist())
                target_values.extend(row_target.cpu().tolist())

    predicted_values = np.asarray(predicted_values)
    cosine_values = np.asarray(cosine_values)
    target_values = np.asarray(target_values)
    labels = target_values >= args.good_threshold
    result = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "n_queries": len(predicted_top),
        "n_donors": int(len(target_values)),
        "good_threshold": args.good_threshold,
        "good_fraction": float(labels.mean()),
        "predicted_trust_auc": float(
            roc_auc_score(labels, predicted_values)
        ),
        "cosine_similarity_auc": float(
            roc_auc_score(labels, cosine_values)
        ),
        "predicted_trust_target_r": finite_correlation(
            predicted_values, target_values
        ),
        "cosine_target_r": finite_correlation(
            cosine_values, target_values
        ),
        "pairwise_ranking_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else None
        ),
        "mean_top1_transfer_quality": float(np.mean(predicted_top)),
        "mean_cosine_top1_transfer_quality": float(np.mean(cosine_top)),
        "mean_oracle_top1_transfer_quality": float(np.mean(oracle_top)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
