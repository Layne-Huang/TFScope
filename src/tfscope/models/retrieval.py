"""Retrieval helpers for v10: learned PWM-transfer-quality predictor.

The earlier v9 confidence gate used max(cosine_similarity) of DBD-pooled ESM2
embeddings to decide how much to trust retrieved neighbours. This breaks for
novel TFs (L0): same-family proteins look near-identical in embedding space
(cos ≈ 0.95) even when their PWMs are completely different.

`TrustPredictor` replaces that with a small head that learns to predict
"will this retrieved PWM transfer to the query?" from
   (query_sequence_features, retrieved_PWM_k, cosine_sim_k).
At training time it's supervised by the actual Pearson r between
`retrieved_pwm_k` and the target PWM. At inference it operates on its
learned features alone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrustPredictor(nn.Module):
    def __init__(self, query_dim: int, max_motif_length: int, hidden_dim: int = 128):
        super().__init__()
        # Encode each retrieved PWM (flattened 4×L) → hidden
        self.pwm_encoder = nn.Sequential(
            nn.Linear(4 * max_motif_length, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Project query features → hidden
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
        )
        # Fuse [query, pwm, sim] → scalar logit
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor,
                retrieved_pwms: torch.Tensor,
                retrieved_sims: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:           (B, query_dim)       per-sample query features
            retrieved_pwms:  (B, K, 4, L)         neighbour PWMs (probabilities)
            retrieved_sims:  (B, K)               cosine sims (one of many features here)

        Returns:
            trust_logits: (B, K)                  pre-sigmoid trust scores
        """
        B, K, _, L = retrieved_pwms.shape
        pwm_flat = retrieved_pwms.reshape(B, K, -1)                # (B, K, 4*L)
        pwm_feat = self.pwm_encoder(pwm_flat)                       # (B, K, hidden)
        q_feat   = self.query_proj(query).unsqueeze(1).expand(B, K, -1)
        fused    = torch.cat([q_feat, pwm_feat, retrieved_sims.unsqueeze(-1)], dim=-1)
        return self.fuse(fused).squeeze(-1)                         # (B, K)


def compute_true_trust(retrieved_pwms: torch.Tensor,
                       retrieved_masks: torch.Tensor,
                       target_pwm: torch.Tensor,
                       pwm_mask: torch.Tensor) -> torch.Tensor:
    """Vectorised "true trust" target: per-position Pearson r between each
    retrieved PWM and the target, averaged over positions valid in BOTH.

    Args:
        retrieved_pwms:  (B, K, 4, L)
        retrieved_masks: (B, K, L)        1.0 where neighbour position is valid
        target_pwm:      (B, 4, L)
        pwm_mask:        (B, L)           1.0 where target position is valid

    Returns:
        trust_target: (B, K) in [0, 1]   normalised from Pearson r ∈ [-1, 1].
    """
    B, K, _, L = retrieved_pwms.shape
    t = target_pwm.unsqueeze(1).expand(B, K, 4, L)                  # (B, K, 4, L)
    p = retrieved_pwms

    # per-position Pearson r between p[:,:,:,l] and t[:,:,:,l] (4-element vectors)
    p_mean = p.mean(dim=2, keepdim=True)
    t_mean = t.mean(dim=2, keepdim=True)
    p_c = p - p_mean
    t_c = t - t_mean
    num = (p_c * t_c).sum(dim=2)                                    # (B, K, L)
    p_std = (p_c ** 2).sum(dim=2).clamp(min=1e-12).sqrt()
    t_std = (t_c ** 2).sum(dim=2).clamp(min=1e-12).sqrt()
    r_per_pos = num / (p_std * t_std + 1e-8)                        # (B, K, L)

    # Average over positions valid in BOTH neighbour and target
    joint_mask = retrieved_masks * pwm_mask.unsqueeze(1)             # (B, K, L)
    r_sum = (r_per_pos * joint_mask).sum(dim=-1)
    cnt = joint_mask.sum(dim=-1).clamp(min=1.0)
    r_per_sample = r_sum / cnt                                       # (B, K) in [-1, 1]
    return torch.clamp((r_per_sample + 1.0) * 0.5, 0.0, 1.0)         # → [0, 1]


def compute_aligned_true_trust(
    retrieved_pwms: torch.Tensor,
    retrieved_masks: torch.Tensor,
    target_pwm: torch.Tensor,
    pwm_mask: torch.Tensor,
    *,
    max_shift: int = 10,
    min_overlap: int = 4,
) -> torch.Tensor:
    """Best shift/RC donor-transfer correlation, normalized to ``[0, 1]``."""
    target = target_pwm.unsqueeze(1)
    target_centered = target - target.mean(dim=2, keepdim=True)
    target_norm = target_centered.norm(dim=2)
    best_selection = target_pwm.new_full(retrieved_masks.shape[:2], -2.0)
    best_r = target_pwm.new_full(retrieved_masks.shape[:2], -1.0)
    target_length = pwm_mask.sum(dim=-1, keepdim=True)
    donor_length = retrieved_masks.sum(dim=-1)
    normalizer = torch.maximum(target_length, donor_length).clamp(min=1.0)

    for orientation in (0, 1):
        donor = retrieved_pwms
        donor_mask = retrieved_masks
        if orientation == 1:
            donor = donor[:, :, [3, 2, 1, 0], :].flip(-1)
            donor_mask = donor_mask.flip(-1)
        donor_centered = donor - donor.mean(dim=2, keepdim=True)
        donor_norm = donor_centered.norm(dim=2)
        for shift in range(-max_shift, max_shift + 1):
            if shift >= 0:
                donor_slice = slice(0, retrieved_pwms.shape[-1] - shift)
                target_slice = slice(shift, retrieved_pwms.shape[-1])
            else:
                donor_slice = slice(-shift, retrieved_pwms.shape[-1])
                target_slice = slice(0, retrieved_pwms.shape[-1] + shift)
            joint = (
                donor_mask[:, :, donor_slice]
                * pwm_mask[:, None, target_slice]
            )
            per_column_r = (
                donor_centered[:, :, :, donor_slice]
                * target_centered[:, :, :, target_slice]
            ).sum(dim=2) / (
                donor_norm[:, :, donor_slice]
                * target_norm[:, :, target_slice]
                + 1e-8
            )
            overlap = joint.sum(dim=-1)
            valid = overlap >= min_overlap
            correlation = (
                per_column_r * joint
            ).sum(dim=-1) / overlap.clamp(min=1.0)
            selection = correlation * overlap / normalizer
            better = valid & (selection > best_selection)
            best_selection = torch.where(better, selection, best_selection)
            best_r = torch.where(better, correlation, best_r)

    return ((best_r + 1.0) * 0.5).clamp(0.0, 1.0)


def pairwise_trust_rank_loss(
    trust_logits: torch.Tensor,
    trust_target: torch.Tensor,
    valid_neighbour: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Rank donors whose aligned transfer quality differs meaningfully."""
    target_difference = trust_target.unsqueeze(2) - trust_target.unsqueeze(1)
    valid_pairs = (
        valid_neighbour.bool().unsqueeze(2)
        & valid_neighbour.bool().unsqueeze(1)
        & (target_difference > margin)
    )
    if not valid_pairs.any():
        return trust_logits.new_zeros(())
    logit_difference = trust_logits.unsqueeze(2) - trust_logits.unsqueeze(1)
    return F.softplus(-logit_difference[valid_pairs]).mean()
