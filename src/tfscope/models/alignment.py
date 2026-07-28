"""Motif alignment for retrieval-augmented PWM prediction.

The single biggest lever in RAG-TFScope: a retrieved neighbour PWM may be the
right motif but offset / reverse-complemented relative to the query. Aligning it
before use takes the LSO top-1 baseline from r=0.33 (raw) to r=0.84 (oracle).

Two entry points:
  - `align_pwm` (numpy): search offset × orientation to align one neighbour PWM
    to a reference PWM. Used for precomputing oracle-alignment teacher targets
    and for the deployable seed-alignment baseline.
  - `align_batch_torch`: vectorised in-model alignment of K neighbours to a
    per-sample reference (e.g. the de-novo head output), used in the forward pass.
"""
import numpy as np
import torch


# ACGT row order. Reverse-complement = swap A<->T (0<->3), C<->G (1<->2) and
# reverse the column (position) order.
_RC_ROWS = [3, 2, 1, 0]


def revcomp_pwm_np(pwm: np.ndarray) -> np.ndarray:
    """(4, L) -> reverse-complement (4, L)."""
    return pwm[_RC_ROWS][:, ::-1]


def _percol_corr(a: np.ndarray, b: np.ndarray, cols) -> float:
    """Mean per-column Pearson r between a and b over the given column indices."""
    rs = []
    for j in cols:
        av, bv = a[:, j], b[:, j]
        if av.std() < 1e-8 or bv.std() < 1e-8:
            continue
        rs.append(np.corrcoef(av, bv)[0, 1])
    return float(np.mean(rs)) if rs else -2.0


def align_pwm(neighbor: np.ndarray,
              reference: np.ndarray,
              max_shift: int = 6,
              consider_revcomp: bool = True,
              coverage_norm: bool = True,
              min_overlap: int = 2):
    """Align `neighbor` (4, Ln) to `reference` (4, Lr) by offset × orientation.

    Returns
    -------
    aligned : (4, Lr)   neighbour placed in reference frame; uniform (0.25) where
                        no overlap exists.
    offset  : int       best shift (neighbour col i -> reference col i+offset).
    orient  : str       "fwd" or "rc".
    score   : float     coverage-normalised per-column Pearson at the best alignment.
    min_overlap : int   minimum number of overlapping motif columns required.

    coverage_norm (default True): the selection score is the mean per-column
    Pearson **weighted by how much of the reference the overlap covers**
    (i.e. Σ r_col / Lr, equivalently mean_overlap_r × n_overlap / Lr). Without
    this, the search degenerately prefers a tiny high-correlation tail (e.g. a
    2-column overlap at r≈1.0) over the honest full-motif alignment — which
    inflates motif-level metrics. With it, a short overlap is penalised by its
    coverage and the full-motif alignment wins. The RETURNED score is the
    honest per-column Pearson over the chosen overlap (not coverage-scaled), so
    downstream metrics read a true correlation.
    """
    Lr = reference.shape[1]
    orientations = [("fwd", neighbor)]
    if consider_revcomp:
        orientations.append(("rc", revcomp_pwm_np(neighbor)))

    best_sel = -np.inf
    best = (None, 0, "fwd", -2.0)
    for orient, o in orientations:
        Ln = o.shape[1]
        for shift in range(-max_shift, max_shift + 1):
            aligned = np.full((4, Lr), 0.25, dtype=np.float32)
            cols = []
            for i in range(Ln):
                j = i + shift
                if 0 <= j < Lr:
                    aligned[:, j] = o[:, i]
                    cols.append(j)
            if len(cols) < min_overlap:
                continue
            r = _percol_corr(reference, aligned, cols)        # mean over overlap
            if r <= -1.5:
                continue
            sel = r * (len(cols) / Lr) if coverage_norm else r
            if sel > best_sel:
                best_sel = sel
                best = (aligned, shift, orient, r)

    aligned, shift, orient, score = best
    if aligned is None:
        aligned = np.full((4, Lr), 0.25, dtype=np.float32)
    return aligned, shift, orient, score


def align_batch_torch(neighbors: torch.Tensor,
                      neighbor_masks: torch.Tensor,
                      reference: torch.Tensor,
                      ref_mask: torch.Tensor,
                      max_shift: int = 6,
                      min_overlap: int = 2,
                      return_masks: bool = False):
    """Vectorised alignment of K neighbours to a per-sample reference.

    Args:
        neighbors:      (B, K, 4, L)  retrieved PWMs (probabilities)
        neighbor_masks: (B, K, L)     1 = valid neighbour column
        reference:      (B, 4, L)     reference PWM (e.g. de-novo head softmax)
        ref_mask:       (B, L)        1 = valid reference column
        max_shift:      int

    Returns:
        aligned:      (B, K, 4, L) each neighbour shifted/oriented to best match
                      the reference; uniform 0.25 where no overlap.
        aligned_mask: (B, K, L), returned only when ``return_masks=True``.

    Notes:
        Alignment choice (shift, orientation) is selected by argmax of per-column
        correlation with the reference — non-differentiable selection, but the
        returned PWM *values* are gathered from the (fixed) neighbour inputs, so
        this is a pure data transform feeding the retrieval prior.

        Retrieved motifs are expected to be left-aligned. Reverse-complementing
        reverses only the valid motif span, not the padded array.
    """
    B, K, _, L = neighbors.shape
    device = neighbors.device

    # Build forward + reverse-complement variants. Reverse only each motif's
    # valid span so padding remains on the right.
    lengths = neighbor_masks.sum(dim=-1).long().clamp(min=0, max=L)
    positions = torch.arange(L, device=device).view(1, 1, L)
    rc_source = (lengths.unsqueeze(-1) - 1 - positions).clamp(min=0)
    rc_source = rc_source.unsqueeze(2).expand(B, K, 4, L)
    rc = torch.gather(neighbors[:, :, _RC_ROWS], dim=-1, index=rc_source)
    rc_valid = positions < lengths.unsqueeze(-1)
    rc = torch.where(rc_valid.unsqueeze(2), rc, torch.full_like(rc, 0.25))
    variants = torch.stack([neighbors, rc], dim=2)            # (B,K,2,4,L)
    var_mask = torch.stack(
        [neighbor_masks, rc_valid.to(neighbor_masks.dtype)], dim=2
    )

    shifts = list(range(-max_shift, max_shift + 1))
    nS = len(shifts)

    # For each shift, roll the variant along position and score vs reference.
    # We compute per-column correlation between 4-vectors of ref and shifted variant.
    ref = reference.unsqueeze(1).unsqueeze(2)                 # (B,1,1,4,L)
    ref_c = ref - ref.mean(dim=3, keepdim=True)
    ref_n = ref_c / (ref_c.norm(dim=3, keepdim=True) + 1e-8)  # (B,1,1,4,L)
    refm = ref_mask.view(B, 1, 1, L)                          # (B,1,1,L)

    best_score = torch.full((B, K, 2, nS), -2.0, device=device)
    rolled_all = torch.empty((B, K, 2, nS, 4, L), device=device)
    rolled_mask = torch.zeros((B, K, 2, nS, L), device=device)

    for si, s in enumerate(shifts):
        rolled = torch.roll(variants, shifts=s, dims=-1)
        rm = torch.roll(var_mask, shifts=s, dims=-1)
        # zero out wrapped-around columns
        if s > 0:
            rolled[..., :s] = 0.25; rm[..., :s] = 0
        elif s < 0:
            rolled[..., s:] = 0.25; rm[..., s:] = 0
        rolled_all[:, :, :, si] = rolled
        rolled_mask[:, :, :, si] = rm

        v_c = rolled - rolled.mean(dim=3, keepdim=True)
        v_n = v_c / (v_c.norm(dim=3, keepdim=True) + 1e-8)    # (B,K,2,4,L)
        # per-column dot product = correlation of 4-vectors
        corr = (ref_n * v_n).sum(dim=3)                       # (B,K,2,L)
        overlap = (refm * rm) > 0.5                           # (B,K,2,L)
        cnt = overlap.sum(dim=-1)
        correlation = (corr * overlap).sum(dim=-1) / cnt.clamp(min=1)
        normalizer = torch.maximum(
            ref_mask.sum(dim=-1, keepdim=True),
            lengths.to(ref_mask.dtype),
        ).clamp(min=1).unsqueeze(-1)
        score = correlation * cnt / normalizer
        score = torch.where(cnt >= min_overlap, score,
                            torch.full_like(score, -2.0))
        best_score[:, :, :, si] = score

    # pick best (orientation, shift) per (B,K)
    flat = best_score.view(B, K, 2 * nS)
    best_idx = flat.argmax(dim=-1)                            # (B,K)
    o_idx = best_idx // nS
    s_idx = best_idx % nS

    bi = torch.arange(B, device=device).view(B, 1).expand(B, K)
    ki = torch.arange(K, device=device).view(1, K).expand(B, K)
    aligned = rolled_all[bi, ki, o_idx, s_idx]                # (B,K,4,L)
    aligned_mask = rolled_mask[bi, ki, o_idx, s_idx]           # (B,K,L)
    valid_alignment = flat.max(dim=-1).values > -1.5
    aligned = torch.where(
        valid_alignment.unsqueeze(-1).unsqueeze(-1),
        aligned,
        torch.full_like(aligned, 0.25),
    )
    aligned_mask = aligned_mask * valid_alignment.unsqueeze(-1)
    if return_masks:
        return aligned, aligned_mask
    return aligned
