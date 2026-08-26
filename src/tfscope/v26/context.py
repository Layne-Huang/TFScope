"""v26 gated context: local flanks and partner chains enter through gated residuals.

Design requirement from the brief: DBD-only must remain a STABLE path even when flank and partner
context are enabled. Both gates are initialised near-closed (bias -3 => sigmoid ~0.047), so a
freshly initialised v26-context model behaves like the v26-core baseline and can only learn to open
the gate if context helps. This is asserted by tests/v26/test_flank_gate_init.py.

Rationale: v25flank retrained with +-20aa flanks and LOST (0.602 vs v24 0.629, and MyoD1
Delta_switch flipped to -1.40). An ungated architecture has to learn to ignore distractor residues
from scratch on 1.7k proteins; a gated residual starts by ignoring them.

Flank residues never serve as contact keys/values in the PWM head (enforced in pwm_head.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Multi-head attention pooling with a learned query."""

    def __init__(self, d, n_heads=8, d_query=64):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, key_mask):
        """h (B,L,d); key_mask (B,L) True where the position should be IGNORED."""
        B = h.shape[0]
        # a row with every key masked would produce NaN; give it a single dummy key
        allmask = key_mask.all(dim=1)
        if allmask.any():
            key_mask = key_mask.clone()
            key_mask[allmask, 0] = False
        q = self.q.expand(B, 1, -1)
        z, _ = self.attn(q, h, h, key_padding_mask=key_mask, need_weights=False)
        return self.norm(z.squeeze(1))


class GatedFlankContext(nn.Module):
    """z = z_core + alpha * z_flank, alpha = sigmoid(W[z_core; z_flank] + b), b init negative."""

    def __init__(self, d, n_heads=8, gate_bias_init=-3.0, flank_dropout=0.4):
        super().__init__()
        self.core_pool = AttentionPool(d, n_heads)
        self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.gate = nn.Linear(2 * d, 1)
        self.norm = nn.LayerNorm(d)
        self.flank_dropout = flank_dropout
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias_init)

    def forward(self, h, pad_mask, dbd_mask, use_flank=True):
        core_ignore = pad_mask | (~dbd_mask)
        z_core = self.core_pool(h, core_ignore)
        if not use_flank:
            return z_core, {"alpha": z_core.new_zeros(z_core.shape[0])}

        flank_ignore = pad_mask | dbd_mask
        if self.training and self.flank_dropout > 0:
            drop = torch.rand(h.shape[0], device=h.device) < self.flank_dropout
            flank_ignore = flank_ignore | drop.unsqueeze(1)
        has_flank = (~flank_ignore).any(dim=1)

        z_f, _ = self.cross(z_core.unsqueeze(1), h, h,
                            key_padding_mask=self._safe(flank_ignore), need_weights=False)
        z_f = z_f.squeeze(1) * has_flank.unsqueeze(-1).float()
        alpha = torch.sigmoid(self.gate(torch.cat([z_core, z_f], dim=-1))).squeeze(-1)
        alpha = alpha * has_flank.float()
        z = self.norm(z_core + alpha.unsqueeze(-1) * z_f)
        return z, {"alpha": alpha.detach()}

    @staticmethod
    def _safe(mask):
        m = mask.clone()
        allm = m.all(dim=1)
        if allm.any():
            m[allm, 0] = False
        return m


class PartnerSetAggregator(nn.Module):
    """Permutation-INVARIANT partner aggregation: primary as query, partners as an unordered set.

    v24 used a per-token chain-ID embedding over concatenated chains, so partner ORDER changed the
    prediction. Here partners enter only through attention over a set with no positional or index
    feature, so any permutation gives an identical result (asserted in
    tests/v26/test_permutation_invariance.py).
    """

    def __init__(self, d, n_heads=8, gate_bias_init=-3.0, partner_dropout=0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.gate = nn.Linear(2 * d, 1)
        self.norm = nn.LayerNorm(d)
        self.partner_dropout = partner_dropout
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias_init)

    def forward(self, z_primary, z_partners, partner_mask, use_partners=True):
        """z_primary (B,d); z_partners (B,P,d); partner_mask (B,P) True where a partner EXISTS."""
        if not use_partners or z_partners is None or z_partners.shape[1] == 0:
            return z_primary, {"beta": z_primary.new_zeros(z_primary.shape[0])}
        ignore = ~partner_mask
        if self.training and self.partner_dropout > 0:
            drop = torch.rand(z_partners.shape[:2], device=z_primary.device) < self.partner_dropout
            ignore = ignore | drop
        has = (~ignore).any(dim=1)
        m = ignore.clone()
        allm = m.all(dim=1)
        if allm.any():
            m[allm, 0] = False
        z_p, _ = self.attn(z_primary.unsqueeze(1), z_partners, z_partners,
                           key_padding_mask=m, need_weights=False)
        z_p = z_p.squeeze(1) * has.unsqueeze(-1).float()
        beta = torch.sigmoid(self.gate(torch.cat([z_primary, z_p], dim=-1))).squeeze(-1)
        beta = beta * has.float()
        return self.norm(z_primary + beta.unsqueeze(-1) * z_p), {"beta": beta.detach()}
