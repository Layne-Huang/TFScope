#!/usr/bin/env python
"""Plot paired V19 E1 ground-truth, no-RAG, and RAG sequence logos."""

import argparse
import json
import os
import sys
import warnings

import logomaker
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
from tfscope.models.tfscope import TFScopeModel


BASES = list("ACGT")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="/data1/leihuang/project/TFScope/checkpoints/v19_e1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="ckpt_epoch050.pt")
    parser.add_argument(
        "--data",
        default="data/processed/tf_pwm_aug_dbd_canon_trim.parquet",
    )
    parser.add_argument(
        "--split",
        default="data/processed/splits/cluster40_clean/split.json",
    )
    parser.add_argument("--out-dir", default="results/v19_e1_epoch50_logos")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--examples-per-group", type=int, default=3)
    parser.add_argument("--ic-thresh", type=float, default=0.25)
    parser.add_argument("--min-overlap", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=10)
    return parser.parse_args()


def load_config(run_dir):
    config = TFScopeConfig()
    with open(os.path.join(run_dir, "config.json")) as handle:
        saved = json.load(handle)
    for key, value in saved.items():
        if not hasattr(config, key):
            continue
        try:
            setattr(config, key, type(getattr(config, key))(value))
        except (TypeError, ValueError):
            setattr(config, key, value)
    return config


def information_content(pwm):
    pwm = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (pwm * np.log2(pwm)).sum(axis=0)


def trim_core(pwm, threshold):
    positions = np.where(information_content(pwm) >= threshold)[0]
    if len(positions) == 0:
        return pwm
    return pwm[:, positions[0] : positions[-1] + 1]


def infer(run_dir, checkpoint, data, split, batch_size, workers, device):
    config = load_config(run_dir)
    model = TFScopeModel(config).to(device).eval()
    state = torch.load(
        os.path.join(run_dir, checkpoint),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state["model"], strict=False)

    dataset = TFDataset(config, data, split, split="test", max_seq_len=1024)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_variable_length,
    )
    predictions = {}
    offset = 0
    with torch.no_grad():
        for batch in loader:
            size = batch["family_id"].shape[0]
            filenames = dataset.filenames[offset : offset + size]
            offset += size
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
            pwm = F.softmax(pwm_logits, dim=1).cpu().numpy()
            gate = torch.sigmoid(gate_logits).cpu().numpy()
            target = batch["target_pwm"].cpu().numpy()
            mask = batch["pwm_mask"].cpu().numpy()
            for index, filename in enumerate(filenames):
                predictions[filename] = {
                    "pwm": pwm[index],
                    "gate": gate[index],
                    "target": target[index],
                    "mask": mask[index],
                }
    del model
    torch.cuda.empty_cache()
    return predictions


def align_entry(entry, threshold, max_shift, min_overlap):
    valid_target = entry["target"][:, entry["mask"].astype(bool)]
    target_core = trim_core(valid_target, threshold)
    active = entry["gate"] > 0.5
    if not active.any():
        active = entry["gate"] > entry["gate"].max() * 0.5
    prediction = entry["pwm"][:, active]
    aligned, shift, orientation, score = align_pwm(
        prediction,
        target_core,
        max_shift=max_shift,
        consider_revcomp=True,
        min_overlap=min_overlap,
    )
    oriented = revcomp_pwm_np(prediction) if orientation == "rc" else prediction
    overlap = (
        sum(
            0 <= position + shift < target_core.shape[1]
            for position in range(oriented.shape[1])
        )
        if score > -1.5
        else 0
    )
    coverage = overlap / target_core.shape[1]
    adjusted_score = float(score) * coverage
    return {
        "target": target_core,
        "prediction": oriented,
        "aligned": aligned,
        "overlap_r": float(score),
        "adjusted_r": adjusted_score,
        "coverage": coverage,
        "overlap": overlap,
        "target_length": target_core.shape[1],
        "prediction_length": oriented.shape[1],
        "shift": int(shift),
        "orientation": orientation,
    }


def choose_unique(rows, count, order):
    chosen = []
    genes = set()
    for index in order:
        row = rows.iloc[index]
        if row["gene"] in genes:
            continue
        genes.add(row["gene"])
        chosen.append(index)
        if len(chosen) == count:
            break
    return chosen


def logo_frame(pwm):
    pwm = np.clip(pwm, 1e-8, 1.0)
    return pd.DataFrame(
        (pwm * information_content(pwm)[None, :]).T,
        columns=BASES,
    )


def draw_logo(ax, pwm, title, title_color):
    logomaker.Logo(
        logo_frame(pwm),
        ax=ax,
        color_scheme={"A": "green", "C": "blue", "G": "#FBA922", "T": "red"},
        show_spines=False,
        fade_probabilities=False,
        font_name="DejaVu Sans",
    )
    ax.set_ylim(0, 2.05)
    ax.set_yticks([0, 1, 2])
    ax.tick_params(axis="both", labelsize=7, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("bits", fontsize=7)
    ax.set_title(title, fontsize=9, color=title_color, pad=3)


def checkpoint_label(checkpoint):
    stem = os.path.splitext(os.path.basename(checkpoint))[0]
    suffix = stem.removeprefix("ckpt_epoch")
    return str(int(suffix)) if suffix.isdigit() else suffix


def save_comparison(flat_selection, scores, aligned, out_dir, epoch, seed, trim_predictions):
    figure, axes = plt.subplots(
        len(flat_selection),
        3,
        figsize=(14, 2.15 * len(flat_selection)),
        squeeze=False,
    )
    for row_index, (group, score_index) in enumerate(flat_selection):
        score = scores.iloc[score_index]
        target, no_rag, rag = aligned[score["filename"]]
        prefix = f"{group}: {score['gene']} [{score['family']}]"
        no_rag_pwm = (
            trim_core(no_rag["prediction"], 0.25)
            if trim_predictions
            else no_rag["prediction"]
        )
        rag_pwm = (
            trim_core(rag["prediction"], 0.25)
            if trim_predictions
            else rag["prediction"]
        )
        draw_logo(axes[row_index, 0], target, f"{prefix}\nGround truth core", "#222222")
        draw_logo(
            axes[row_index, 1],
            no_rag_pwm,
            f"No-RAG: overlap r={score['no_rag_overlap_r']:.3f}, "
            f"adjusted={score['no_rag_adjusted_r']:.3f}\n"
            f"overlap={score['no_rag_overlap']:.0f}/{score['target_length']:.0f}, "
            f"shift={score['no_rag_shift']:+.0f}, "
            f"{score['no_rag_orientation']}",
            "#2166ac",
        )
        draw_logo(
            axes[row_index, 2],
            rag_pwm,
            f"RAG: overlap r={score['rag_overlap_r']:.3f}, "
            f"adjusted={score['rag_adjusted_r']:.3f}\n"
            f"overlap={score['rag_overlap']:.0f}/{score['target_length']:.0f}, "
            f"delta adjusted={score['delta_adjusted_r']:+.3f}, "
            f"{score['rag_orientation']}",
            "#b2182b",
        )

    view = "information-trimmed predictions" if trim_predictions else "full gate-selected predictions"
    figure.suptitle(
        f"TFScope V19 E1 epoch {epoch}, seed {seed}: "
        f"ground truth vs no-RAG vs clean train-only RAG\n{view}",
        fontsize=14,
        fontweight="bold",
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    suffix = "trimmed" if trim_predictions else "full"
    prefix = f"epoch{epoch}_logo_comparison_{suffix}"
    pdf_path = os.path.join(out_dir, f"{prefix}.pdf")
    png_path = os.path.join(out_dir, f"{prefix}.png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(figure)
    return pdf_path, png_path


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    metadata = pd.read_parquet(args.data).set_index("filename")

    run_dirs = {
        "norag": os.path.join(args.base, f"norag_seed{args.seed}"),
        "rag": os.path.join(args.base, f"rag_seed{args.seed}"),
    }
    inferred = {
        mode: infer(
            run_dir,
            args.checkpoint,
            args.data,
            args.split,
            args.batch_size,
            args.workers,
            device,
        )
        for mode, run_dir in run_dirs.items()
    }

    rows = []
    aligned = {}
    for filename in inferred["norag"]:
        no_rag = align_entry(
            inferred["norag"][filename],
            args.ic_thresh,
            args.max_shift,
            args.min_overlap,
        )
        rag = align_entry(
            inferred["rag"][filename],
            args.ic_thresh,
            args.max_shift,
            args.min_overlap,
        )
        if no_rag["target_length"] < 4:
            continue
        meta = metadata.loc[filename]
        rows.append(
            {
                "filename": filename,
                "gene": str(meta["gene_symbol"]),
                "family": str(meta["family_name"]),
                "no_rag_overlap_r": no_rag["overlap_r"],
                "rag_overlap_r": rag["overlap_r"],
                "no_rag_adjusted_r": no_rag["adjusted_r"],
                "rag_adjusted_r": rag["adjusted_r"],
                "delta_adjusted_r": rag["adjusted_r"] - no_rag["adjusted_r"],
                "no_rag_coverage": no_rag["coverage"],
                "rag_coverage": rag["coverage"],
                "no_rag_overlap": no_rag["overlap"],
                "rag_overlap": rag["overlap"],
                "target_length": no_rag["target_length"],
                "no_rag_length": no_rag["prediction_length"],
                "rag_length": rag["prediction_length"],
                "no_rag_shift": no_rag["shift"],
                "no_rag_orientation": no_rag["orientation"],
                "rag_shift": rag["shift"],
                "rag_orientation": rag["orientation"],
            }
        )
        aligned[filename] = (no_rag["target"], no_rag, rag)

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(args.out_dir, "per_record_scores.csv"), index=False)

    count = args.examples_per_group
    win_order = np.argsort(-scores["delta_adjusted_r"].to_numpy())
    loss_order = np.argsort(scores["delta_adjusted_r"].to_numpy())
    typical_order = np.argsort(np.abs(scores["delta_adjusted_r"].to_numpy()))
    selected = [
        ("RAG wins", choose_unique(scores, count, win_order)),
        ("Similar", choose_unique(scores, count, typical_order)),
        ("No-RAG wins", choose_unique(scores, count, loss_order)),
    ]

    flat_selection = [
        (group, index) for group, indices in selected for index in indices
    ]
    epoch = checkpoint_label(args.checkpoint)
    for trim_predictions in (False, True):
        pdf_path, png_path = save_comparison(
            flat_selection,
            scores,
            aligned,
            args.out_dir,
            epoch,
            args.seed,
            trim_predictions,
        )
        print(f"Saved {pdf_path}")
        print(f"Saved {png_path}")
    print(f"Saved {os.path.join(args.out_dir, 'per_record_scores.csv')}")


if __name__ == "__main__":
    main()
