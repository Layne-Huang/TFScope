"""Same-gene PWM registration consistency analysis for TFScope V19."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations

import numpy as np
import pandas as pd

from tfscope.models.alignment import revcomp_pwm_np


@dataclass(frozen=True)
class AuditThresholds:
    min_overlap: int = 4
    max_shift: int = 10
    consistent_median_r: float = 0.75
    consistent_q25_r: float = 0.60
    cluster_r: float = 0.65
    registration_gain: float = 0.15
    discordant_pair_fraction: float = 0.25
    anchor_min_r: float = 0.80
    anchor_min_coverage: float = 0.60

    def to_dict(self) -> dict:
        return asdict(self)


def decode_pwm(blob) -> np.ndarray:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("PWM must be a float32 byte blob")
    pwm = np.frombuffer(blob, dtype=np.float32).reshape(4, -1).copy()
    pwm = np.clip(pwm, 1e-8, None)
    return pwm / pwm.sum(axis=0, keepdims=True).clip(1e-8)


def pwm_fingerprint(pwm: np.ndarray) -> str:
    rounded = np.round(np.asarray(pwm, dtype=np.float32), decimals=6)
    return sha256(rounded.tobytes()).hexdigest()


def information_content(pwm: np.ndarray) -> np.ndarray:
    pwm = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (pwm * np.log2(pwm)).sum(axis=0)


def _column_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(a_centered, axis=0) * np.linalg.norm(
        b_centered, axis=0
    )
    valid = denom > 1e-8
    values = np.full(a.shape[1], np.nan, dtype=np.float64)
    values[valid] = (
        (a_centered[:, valid] * b_centered[:, valid]).sum(axis=0) / denom[valid]
    )
    return values


def score_fixed_frame(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    overlap = min(a.shape[1], b.shape[1])
    if overlap == 0:
        return float("nan"), 0
    correlations = _column_correlations(a[:, :overlap], b[:, :overlap])
    return float(np.nanmean(correlations)), overlap


def align_pair(
    reference: np.ndarray,
    query: np.ndarray,
    max_shift: int = 10,
    min_overlap: int = 4,
) -> dict:
    """Symmetrically score query offset/orientation in the reference frame."""
    best = None
    normalizer = max(reference.shape[1], query.shape[1])
    for orientation, oriented in (
        ("fwd", query),
        ("rc", revcomp_pwm_np(query)),
    ):
        for shift in range(-max_shift, max_shift + 1):
            query_start = max(0, -shift)
            query_end = min(oriented.shape[1], reference.shape[1] - shift)
            overlap = query_end - query_start
            if overlap < min_overlap:
                continue
            reference_start = query_start + shift
            a = reference[:, reference_start : reference_start + overlap]
            b = oriented[:, query_start:query_end]
            correlations = _column_correlations(a, b)
            score = float(np.nanmean(correlations))
            if not np.isfinite(score):
                continue
            selection_score = score * overlap / normalizer
            candidate = (
                selection_score,
                overlap,
                score,
                orientation == "fwd",
                -abs(shift),
                shift,
                orientation,
            )
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return {
            "aligned_r": float("nan"),
            "shift": 0,
            "orientation": "fwd",
            "overlap": 0,
            "coverage": 0.0,
        }
    _, overlap, score, _, _, shift, orientation = best
    return {
        "aligned_r": score,
        "shift": int(shift),
        "orientation": orientation,
        "overlap": int(overlap),
        "coverage": float(overlap / normalizer),
    }


def deduplicate_motifs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    records = []
    for _, row in df.iterrows():
        pwm = decode_pwm(row["pwm"])
        record = row.to_dict()
        record["_pwm_array"] = pwm
        record["_pwm_fingerprint"] = pwm_fingerprint(pwm)
        records.append(record)
    work = pd.DataFrame(records)
    key = ["gene_symbol", "source_id", "_pwm_fingerprint"]
    representatives = []
    for _, group in work.groupby(key, dropna=False, sort=True):
        representative = group.sort_values("filename").iloc[0].copy()
        representative["duplicate_row_count"] = int(len(group))
        representative["duplicate_filenames"] = sorted(
            group["filename"].astype(str).tolist()
        )
        representatives.append(representative)
    unique = pd.DataFrame(representatives).reset_index(drop=True)
    return unique, {
        "input_rows": int(len(df)),
        "unique_motif_records": int(len(unique)),
        "collapsed_duplicate_rows": int(len(df) - len(unique)),
    }


def build_pair_rows(gene_rows: pd.DataFrame, thresholds: AuditThresholds) -> list[dict]:
    rows = []
    ordered = gene_rows.sort_values("filename").reset_index(drop=True)
    for left_index, right_index in combinations(range(len(ordered)), 2):
        left = ordered.iloc[left_index]
        right = ordered.iloc[right_index]
        left_pwm = left["_pwm_array"]
        right_pwm = right["_pwm_array"]
        alignment = align_pair(
            left_pwm,
            right_pwm,
            max_shift=thresholds.max_shift,
            min_overlap=thresholds.min_overlap,
        )
        fixed_r, fixed_overlap = score_fixed_frame(left_pwm, right_pwm)
        rows.append(
            {
                "gene_symbol": str(left["gene_symbol"]),
                "family_name": str(left["family_name"]),
                "left_filename": str(left["filename"]),
                "right_filename": str(right["filename"]),
                "left_source_id": str(left["source_id"]),
                "right_source_id": str(right["source_id"]),
                "left_source": str(left["source"]),
                "right_source": str(right["source"]),
                "left_length": int(left_pwm.shape[1]),
                "right_length": int(right_pwm.shape[1]),
                "fixed_r": fixed_r,
                "fixed_overlap": fixed_overlap,
                "registration_gain": alignment["aligned_r"] - fixed_r,
                **alignment,
            }
        )
    return rows


def connected_components(filenames: list[str], pair_rows: list[dict], threshold: float):
    adjacency = {filename: set() for filename in filenames}
    for pair in pair_rows:
        if pair["aligned_r"] >= threshold:
            left = pair["left_filename"]
            right = pair["right_filename"]
            adjacency[left].add(right)
            adjacency[right].add(left)
    components = []
    remaining = set(filenames)
    while remaining:
        root = min(remaining)
        stack = [root]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        remaining -= component
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values))


def classify_gene(
    gene_rows: pd.DataFrame,
    pair_rows: list[dict],
    thresholds: AuditThresholds,
) -> tuple[str, str, list[list[str]]]:
    if len(gene_rows) == 1:
        return "single_record", "Only one unique motif record.", [
            [str(gene_rows.iloc[0]["filename"])]
        ]

    aligned = np.asarray([row["aligned_r"] for row in pair_rows], dtype=float)
    q25 = float(np.quantile(aligned, 0.25))
    median = float(np.median(aligned))
    registration_discordant = np.asarray(
        [
            row["orientation"] == "rc"
            or row["shift"] != 0
            or row["registration_gain"] >= thresholds.registration_gain
            for row in pair_rows
        ]
    )
    discordant_fraction = float(registration_discordant.mean())
    components = connected_components(
        gene_rows["filename"].astype(str).tolist(),
        pair_rows,
        thresholds.cluster_r,
    )
    if (
        median >= thresholds.consistent_median_r
        and q25 >= thresholds.consistent_q25_r
    ):
        if discordant_fraction >= thresholds.discordant_pair_fraction:
            return (
                "registration_discordant",
                "Motif content agrees after offset/RC alignment but fixed frames disagree.",
                components,
            )
        return (
            "consistent_single_motif",
            "Pairwise motif content and fixed registration are consistent.",
            components,
        )
    if len(components) >= 2 and len(gene_rows) >= 3:
        component_sources = [
            set(
                gene_rows.loc[
                    gene_rows["filename"].astype(str).isin(component), "source_id"
                ].astype(str)
            )
            for component in components
        ]
        source_separated = any(
            left.isdisjoint(right)
            for left, right in combinations(component_sources, 2)
        )
        if len(set(gene_rows["source_id"].astype(str))) >= 2 and source_separated:
            return (
                "candidate_multimodal",
                "Records form multiple aligned-PWM components across source IDs.",
                components,
            )
    return (
        "noisy_or_contradictory",
        "Records lack a single high-consistency motif or stable source-linked modes.",
        components,
    )


def choose_medoid(gene_rows: pd.DataFrame, pair_rows: list[dict]) -> str:
    filenames = sorted(gene_rows["filename"].astype(str).tolist())
    if len(filenames) == 1:
        return filenames[0]
    scores = {filename: [] for filename in filenames}
    for pair in pair_rows:
        scores[pair["left_filename"]].append(pair["aligned_r"])
        scores[pair["right_filename"]].append(pair["aligned_r"])
    return max(filenames, key=lambda filename: (np.mean(scores[filename]), filename))


def build_anchor_rows(
    gene_rows: pd.DataFrame,
    pair_rows: list[dict],
    classification: str,
    thresholds: AuditThresholds,
) -> list[dict]:
    if classification not in {"consistent_single_motif", "registration_discordant"}:
        return []
    medoid_filename = choose_medoid(gene_rows, pair_rows)
    by_filename = gene_rows.set_index("filename")
    medoid = by_filename.loc[medoid_filename]
    anchors = []
    for filename, row in by_filename.iterrows():
        if filename == medoid_filename:
            alignment = {
                "aligned_r": 1.0,
                "shift": 0,
                "orientation": "fwd",
                "overlap": int(row["_pwm_array"].shape[1]),
                "coverage": 1.0,
            }
        else:
            alignment = align_pair(
                medoid["_pwm_array"],
                row["_pwm_array"],
                max_shift=thresholds.max_shift,
                min_overlap=thresholds.min_overlap,
            )
        if (
            alignment["aligned_r"] < thresholds.anchor_min_r
            or alignment["coverage"] < thresholds.anchor_min_coverage
        ):
            continue
        anchors.append(
            {
                "filename": str(filename),
                "gene_symbol": str(row["gene_symbol"]),
                "family_name": str(row["family_name"]),
                "split": str(row["split"]),
                "source_id": str(row["source_id"]),
                "reference_filename": str(medoid_filename),
                "orientation_to_reference": alignment["orientation"],
                "offset_to_reference": alignment["shift"],
                "aligned_r": alignment["aligned_r"],
                "overlap": alignment["overlap"],
                "coverage": alignment["coverage"],
                "anchor_type": "same_gene_consensus_relative",
                "absolute_orientation_resolved": False,
            }
        )
    return anchors


def summarize_gene(
    gene_rows: pd.DataFrame,
    pair_rows: list[dict],
    thresholds: AuditThresholds,
) -> tuple[dict, list[dict]]:
    classification, reason, components = classify_gene(
        gene_rows, pair_rows, thresholds
    )
    pwms = gene_rows["_pwm_array"].tolist()
    lengths = np.asarray([pwm.shape[1] for pwm in pwms], dtype=float)
    mean_ic = np.asarray([information_content(pwm).mean() for pwm in pwms])
    aligned = np.asarray([row["aligned_r"] for row in pair_rows], dtype=float)
    fixed = np.asarray([row["fixed_r"] for row in pair_rows], dtype=float)
    registration_gain = np.asarray(
        [row["registration_gain"] for row in pair_rows], dtype=float
    )
    rc_fraction = (
        float(np.mean([row["orientation"] == "rc" for row in pair_rows]))
        if pair_rows
        else 0.0
    )
    shifted_fraction = (
        float(np.mean([row["shift"] != 0 for row in pair_rows]))
        if pair_rows
        else 0.0
    )
    anchors = build_anchor_rows(
        gene_rows, pair_rows, classification, thresholds
    )
    summary = {
        "gene_symbol": str(gene_rows.iloc[0]["gene_symbol"]),
        "family_name": str(gene_rows.iloc[0]["family_name"]),
        "split": str(gene_rows.iloc[0]["split"]),
        "classification": classification,
        "classification_reason": reason,
        "n_unique_motifs": int(len(gene_rows)),
        "n_source_ids": int(gene_rows["source_id"].astype(str).nunique()),
        "n_sources": int(gene_rows["source"].astype(str).nunique()),
        "n_pairs": int(len(pair_rows)),
        "median_aligned_r": float(np.median(aligned)) if len(aligned) else None,
        "q25_aligned_r": float(np.quantile(aligned, 0.25)) if len(aligned) else None,
        "min_aligned_r": float(np.min(aligned)) if len(aligned) else None,
        "median_fixed_r": float(np.median(fixed)) if len(fixed) else None,
        "median_registration_gain": (
            float(np.median(registration_gain)) if len(registration_gain) else None
        ),
        "rc_pair_fraction": rc_fraction,
        "shifted_pair_fraction": shifted_fraction,
        "motif_length_min": int(lengths.min()),
        "motif_length_max": int(lengths.max()),
        "motif_length_std": float(lengths.std()),
        "mean_ic_mean": float(mean_ic.mean()),
        "mean_ic_std": float(mean_ic.std()),
        "n_components": int(len(components)),
        "component_sizes": [len(component) for component in components],
        "components": components,
        "medoid_filename": choose_medoid(gene_rows, pair_rows),
        "n_relative_anchors": int(len(anchors)),
    }
    return summary, anchors
