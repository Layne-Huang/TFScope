"""Unified apples-to-apples evaluator (ICLR Phase-I).

Fixes the B0/B1-vs-v24 protocol inconsistency documented in
`results/iclr_phase1_apples_to_apples/AUDIT_FINDINGS.md` by scoring *every* model
— training-free baselines, trained variants, and the frozen v24 reference —
through two clearly separated panels:

  Panel A — oracle-content
    Every model is given the ground-truth motif length and the SAME shift +
    reverse-complement registration; the score is pure PWM content quality
    (overlap Pearson r over the GT core). No coverage / length penalty. This is
    the "is the motif right" number and is directly comparable to the docs'
    "r (overlap Pearson)" row.

  Panel B — end-to-end
    Each model uses its predicted length: trained models use their predicted
    gate span; baselines use a PRE-REGISTERED length policy that never reads the
    test target (per-family median training motif length, global-train median
    fallback for zero-shot families). covR = overlap_r x coverage, and coverage,
    gate-length MAE, and length bias are reported SEPARATELY (not silently folded
    into one number).

All registration/RC/gene-aggregation is identical across models. Nothing here
tunes on the test set. Aggregation is equal-weight over gene groups (gene_covR).

The scoring core (`panel_A`, `panel_B_from_core`, length policy) is pure NumPy and
unit-tested in `tests/test_unified_eval.py` — no GPU required.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from tfscope.models.alignment import align_pwm, revcomp_pwm_np  # noqa: E402

MAX_SHIFT = 10
MIN_OVERLAP = 2
IC_THRESH = 0.25


def ic_bits(p):
    p = np.clip(p, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)


def trimmed_core(pwm: np.ndarray, ic_thresh: float = IC_THRESH):
    t = np.clip(np.asarray(pwm, np.float32), 1e-8, 1.0)
    t = t / t.sum(0, keepdims=True)
    ic = ic_bits(t)
    inf = np.where(ic >= ic_thresh)[0]
    if len(inf) == 0:
        return None
    c = t[:, inf[0]:inf[-1] + 1]
    return c / c.sum(0, keepdims=True)


def _aligned_cols(pred: np.ndarray, core: np.ndarray):
    """Oracle shift+RC align; return (aligned-in-core-frame, covered col idxs, orient)."""
    aligned, shift, orient, score = align_pwm(
        pred, core, max_shift=MAX_SHIFT, consider_revcomp=True, min_overlap=MIN_OVERLAP)
    o = revcomp_pwm_np(pred) if orient == "rc" else pred
    cols = np.array([i + shift for i in range(o.shape[1]) if 0 <= i + shift < core.shape[1]], int)
    return aligned, cols


def _overlap_r(core: np.ndarray, aligned: np.ndarray, cols: np.ndarray) -> float:
    if len(cols) < 2:
        return 0.0
    t = core[:, cols]
    p = np.clip(aligned[:, cols], 1e-8, 1.0)
    p = p / p.sum(0, keepdims=True)
    rs = []
    for j in range(t.shape[1]):
        if t[:, j].std() == 0 or p[:, j].std() == 0:
            rs.append(0.0)
        else:
            rs.append(np.corrcoef(t[:, j], p[:, j])[0, 1])
    return float(np.nanmean(rs))


# ── Panel A: oracle-content (GT length, no coverage penalty) ───────────────────
def panel_A(pred_full: np.ndarray, core: np.ndarray) -> dict:
    aligned, cols = _aligned_cols(pred_full, core)
    return {
        "content_r": _overlap_r(core, aligned, cols),
        "overlap_frac": len(cols) / core.shape[1],
    }


# ── Panel B: end-to-end (predicted length, coverage/length reported) ───────────
def _apply_length(pred_full: np.ndarray, pred_len: int | None) -> np.ndarray:
    """Truncate/pad the (already contiguous) prediction to `pred_len`, left-anchored.

    pred_len None -> use the prediction as-is (its own native length).
    """
    if pred_len is None:
        return pred_full
    pred_len = max(1, int(pred_len))
    L = pred_full.shape[1]
    if L == pred_len:
        return pred_full
    if L > pred_len:
        return pred_full[:, :pred_len]
    out = np.full((4, pred_len), 0.25, np.float32)
    out[:, :L] = pred_full
    return out


def panel_B(pred_full: np.ndarray, core: np.ndarray, pred_len: int | None) -> dict:
    pred = _apply_length(pred_full, pred_len)
    aligned, cols = _aligned_cols(pred, core)
    r = _overlap_r(core, aligned, cols)
    Lg = core.shape[1]
    coverage = len(cols) / Lg
    used_len = pred.shape[1] if pred_len is None else int(pred_len)
    return {
        "covR": r * coverage,
        "overlap_r": r,
        "coverage": coverage,
        "len_pred": used_len,
        "len_gt": Lg,
        "len_mae": abs(used_len - Lg),
        "len_bias": used_len - Lg,
    }


# ── pre-registered baseline length policy (never reads the test target) ────────
def train_length_policy(train_df):
    by_fam = {int(k): int(round(v)) for k, v in
              train_df.groupby("family_id")["motif_length"].median().items()}
    global_med = int(round(float(train_df["motif_length"].median())))
    return by_fam, global_med


def baseline_pred_len(family_id, policy) -> int:
    by_fam, global_med = policy
    return int(by_fam.get(int(family_id), global_med))


# ── aggregation ────────────────────────────────────────────────────────────────
def _gene_balanced(per_sample, key):
    by_gene = {}
    for s in per_sample:
        by_gene.setdefault(s["gene"], []).append(s[key])
    gene_means = {g: float(np.nanmean(v)) for g, v in by_gene.items()}
    return (float(np.nanmean(list(gene_means.values()))) if gene_means else float("nan"),
            gene_means)


def score_model(model_tag, preds, targets, meta, length_fn=None):
    """
    preds:   {filename: pred_full (4, Lpred)}  full predicted PWM content.
    targets: {filename: (pwm_padded_or_raw, )}  -> use meta for gt pwm; here we pass
             {filename: gt_pwm_raw (4, Lgt)}.
    meta:    {filename: {"gene":.., "family_id":.., "family":.., "n_chains":.., "pred_len":..}}
             pred_len is the model's end-to-end length (gate span for trained models,
             or None to have the caller supply it); if absent, Panel B uses native length.
    length_fn(filename, meta_row) -> int|None : overrides pred_len for Panel B.
    """
    per_sample = []
    for fn, pred in preds.items():
        gt = targets.get(fn)
        if gt is None:
            continue
        core = trimmed_core(gt)
        if core is None:
            continue
        m = meta.get(fn, {})
        A = panel_A(pred, core)
        pred_len = length_fn(fn, m) if length_fn is not None else m.get("pred_len")
        B = panel_B(pred, core, pred_len)
        per_sample.append({
            "filename": fn, "gene": str(m.get("gene", fn)),
            "family": str(m.get("family", "NA")), "n_chains": int(m.get("n_chains", 1)),
            "A_content_r": A["content_r"], "A_overlap_frac": A["overlap_frac"],
            "B_covR": B["covR"], "B_coverage": B["coverage"],
            "B_len_mae": B["len_mae"], "B_len_bias": B["len_bias"],
        })
    a_gene, a_by_gene = _gene_balanced(per_sample, "A_content_r")
    b_gene, b_by_gene = _gene_balanced(per_sample, "B_covR")
    mono = [s for s in per_sample if s["n_chains"] <= 1]
    multi = [s for s in per_sample if s["n_chains"] > 1]
    return {
        "model": model_tag,
        "panelA_oracle_content": {
            "gene_content_r": a_gene,
            "row_content_r": float(np.nanmean([s["A_content_r"] for s in per_sample])) if per_sample else float("nan"),
            "mean_overlap_frac": float(np.nanmean([s["A_overlap_frac"] for s in per_sample])) if per_sample else float("nan"),
        },
        "panelB_end_to_end": {
            "gene_covR": b_gene,
            "row_covR": float(np.nanmean([s["B_covR"] for s in per_sample])) if per_sample else float("nan"),
            "mean_coverage": float(np.nanmean([s["B_coverage"] for s in per_sample])) if per_sample else float("nan"),
            "gate_len_mae": float(np.nanmean([s["B_len_mae"] for s in per_sample])) if per_sample else float("nan"),
            "gate_len_bias": float(np.nanmean([s["B_len_bias"] for s in per_sample])) if per_sample else float("nan"),
        },
        "monomer_gene_covR": _gene_balanced(mono, "B_covR")[0] if mono else None,
        "multimer_gene_covR": _gene_balanced(multi, "B_covR")[0] if multi else None,
        "by_gene_covR": b_by_gene,
        "n_scored": len(per_sample),
        "per_sample": per_sample,
    }
