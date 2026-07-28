"""Residue × DNA-position interface head (ICLR 2026 Candidate B).

Replaces the opaque global recognition-energy path with an explicit pair
representation between chain residues and *latent* DNA positions, following
plan §5:

    Z[k,i,j]    = W_h h[k,i] + W_q q[j] + W_c c[k]
    C[k,i,j]    = sigmoid(contact_head(Z[k,i,j]))          # occupancy in [0,1]
    E[k,i,j,b]  = base_energy_head(Z[k,i,j])[b]            # base-specific energy
    logit[j,b]  = prior[j,b] + sum_{k,i} C[k,i,j] * E[k,i,j,b]

Here the chain index ``k`` is folded into the residue index ``i``: each residue
carries a *context* vector ``c`` (e.g. the equivariant chain summary produced by
:class:`~tfscope.models.chain_set_encoder.ChainSetEncoder`), so there is no fixed
chain identity. ``q[j]`` are learned latent DNA-position queries.

Structural supervision is **training-only**: the predicted occupancy ``C`` is
distilled against a 2D residue × DNA-position contact map, with missing labels
*masked* (never treated as negatives). At inference the head is sequence-only —
``C`` and ``E`` are predicted, so no structure is required. A 1D recognition
marginal and shuffled / wrong-family controls are provided for the plan's causal
ablations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterfacePairHead(nn.Module):
    """Explicit residue↔latent-DNA-position pair mixer with contact occupancy
    and base-specific energy aggregation.

    Parameters are taken from ``config`` with safe getattr defaults so the head
    composes with the existing config without touching the frozen v24 fields.
    """

    def __init__(self, config, residue_dim: int, chain_ctx_dim: int | None = None):
        super().__init__()
        self.n_positions = config.max_motif_length          # J latent DNA positions
        d = getattr(config, "interface_pair_dim", 128)       # pair hidden dim
        self.chain_ctx_dim = chain_ctx_dim

        self.W_h = nn.Linear(residue_dim, d)
        self.W_q = nn.Linear(d, d, bias=False)               # applied to query embeddings
        self.q = nn.Parameter(torch.randn(self.n_positions, d) * 0.02)
        self.W_c = nn.Linear(chain_ctx_dim, d) if chain_ctx_dim else None

        self.pair_norm = nn.LayerNorm(d)
        self.pair_act = nn.GELU()
        self.contact_head = nn.Linear(d, 1)                  # → occupancy logit
        self.base_energy_head = nn.Linear(d, 4)              # → per-base energy
        # learned positional base prior (log-space); broadcast over batch
        self.base_prior = nn.Parameter(torch.zeros(4, self.n_positions))

    def forward(self, residue_feats: torch.Tensor, valid: torch.Tensor,
                chain_ctx: torch.Tensor | None = None,
                prior: torch.Tensor | None = None):
        """
        residue_feats: (B, L, residue_dim) context-aware residue states h[k,i].
        valid:         (B, L) bool — True for real DBD residues.
        chain_ctx:     (B, L, chain_ctx_dim) per-residue chain context c[k]
                       (optional; e.g. broadcast chain summary).
        prior:         (B, 4, J) optional external log-prior; added to base_prior.

        Returns dict with:
          pwm_logits:      (B, 4, J)
          C:               (B, L, J) predicted residue↔position occupancy in [0,1]
          occ_res:         (B, L) per-residue occupancy marginal (max over j)
          occ_pos:         (B, J) per-position occupancy marginal (sum over residues)
        """
        B, L, _ = residue_feats.shape
        J, d = self.n_positions, self.q.shape[1]
        valid_f = valid.unsqueeze(-1).to(residue_feats.dtype)  # (B, L, 1)

        zh = self.W_h(residue_feats)                           # (B, L, d)
        zq = self.W_q(self.q)                                  # (J, d)
        # Z[b, i, j, :] = zh[b,i] + zq[j] (+ zc[b,i])
        Z = zh.unsqueeze(2) + zq.view(1, 1, J, d)              # (B, L, J, d)
        if self.W_c is not None and chain_ctx is not None:
            Z = Z + self.W_c(chain_ctx).unsqueeze(2)
        Z = self.pair_act(self.pair_norm(Z))

        contact_logit = self.contact_head(Z).squeeze(-1)       # (B, L, J)
        C = torch.sigmoid(contact_logit) * valid_f             # zero occupancy on padding
        E = self.base_energy_head(Z)                           # (B, L, J, 4)

        # logit[b, base, j] = prior + sum_i C[i,j] * E[i,j,base]
        agg = torch.einsum("blj,bljc->bcj", C, E)              # (B, 4, J)
        pwm_logits = agg + self.base_prior.unsqueeze(0)
        if prior is not None:
            pwm_logits = pwm_logits + prior

        occ_res = C.max(dim=2).values                          # (B, L)
        occ_pos = C.sum(dim=1)                                 # (B, J)
        return {
            "pwm_logits": pwm_logits,
            "C": C,
            "contact_logit": contact_logit,
            "occ_res": occ_res,
            "occ_pos": occ_pos,
        }

    # ── training-only structural supervision (plan §5) ─────────────────────────
    @staticmethod
    def distill_loss_2d(contact_logit: torch.Tensor, target: torch.Tensor,
                        label_mask: torch.Tensor) -> torch.Tensor:
        """Masked BCE distilling the 2D contact map into predicted occupancy.

        contact_logit: (B, L, J) pre-sigmoid occupancy logits.
        target:        (B, L, J) in [0,1] — true residue×position contacts.
        label_mask:    (B, L, J) bool — True where a label EXISTS. Missing labels
                       are masked out (never treated as negatives, per plan §5).
        """
        m = label_mask.to(contact_logit.dtype)
        if m.sum() == 0:
            return contact_logit.new_zeros(())
        loss = F.binary_cross_entropy_with_logits(contact_logit, target, reduction="none")
        return (loss * m).sum() / m.sum().clamp(min=1.0)

    @staticmethod
    def marginal_loss_1d(occ_res: torch.Tensor, target_res: torch.Tensor,
                         label_mask_res: torch.Tensor) -> torch.Tensor:
        """1D recognition-residue marginal supervision (ablation only, plan §5).

        occ_res:        (B, L) predicted per-residue occupancy marginal in [0,1].
        target_res:     (B, L) in [0,1] — is-a-recognition-residue indicator.
        label_mask_res: (B, L) bool.
        """
        m = label_mask_res.to(occ_res.dtype)
        if m.sum() == 0:
            return occ_res.new_zeros(())
        loss = F.binary_cross_entropy(occ_res.clamp(1e-6, 1 - 1e-6), target_res, reduction="none")
        return (loss * m).sum() / m.sum().clamp(min=1.0)

    @staticmethod
    def shuffle_contacts(target: torch.Tensor, label_mask: torch.Tensor,
                         generator: torch.Generator | None = None):
        """Row-shuffle the residue axis of the 2D contact map (negative control).

        Produces a contact map with the same marginal statistics but destroyed
        residue↔position correspondence — the plan's shuffled-contact control.
        Returns (shuffled_target, shuffled_mask).
        """
        B, L, J = target.shape
        perm = torch.stack([torch.randperm(L, generator=generator) for _ in range(B)])
        idx = perm.view(B, L, 1).expand(B, L, J).to(target.device)
        return torch.gather(target, 1, idx), torch.gather(label_mask, 1, idx)
