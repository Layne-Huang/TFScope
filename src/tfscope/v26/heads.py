"""v26 output heads: motif-length distribution + contact-aware PWM decoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotifLengthHead(nn.Module):
    """P(L | z) over an allowed range, turned into a CONTIGUOUS prefix gate g_j = P(L > j).

    v24 used 42 independent gate logits, which can produce fragmented, non-contiguous gates and
    makes motif-length bias hard to read. Targets are canonicalised so the first informative
    column sits at position 0 (canonicalize_pwms.py), so a prefix gate is the correct shape:
    predicting a LENGTH is strictly less expressive than 42 free bits, and monotone by
    construction.
    """

    def __init__(self, d, min_len=4, max_len=42, hidden=128):
        super().__init__()
        self.min_len, self.max_len = min_len, max_len
        self.n = max_len - min_len + 1
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, self.n))

    def forward(self, z):
        logits = self.net(z)                                     # (B, n_lengths)
        p = F.softmax(logits, dim=-1)
        # g_j = P(L > j) for j = 0..max_len-1  -> monotone non-increasing, contiguous prefix
        lengths = torch.arange(self.min_len, self.max_len + 1, device=z.device)
        j = torch.arange(self.max_len, device=z.device)
        gate = (p.unsqueeze(-1) * (lengths.view(1, -1, 1) > j.view(1, 1, -1)).float()).sum(1)
        exp_len = (p * lengths.float()).sum(-1)
        return gate, {"length_logits": logits, "length_probs": p, "expected_length": exp_len}


class ContactAwarePWMHead(nn.Module):
    """Z_final = Z_prior + lambda * Z_contact, column-wise softmax.

    Z_prior   from the pooled (core/context/assembly) representation.
    Z_contact from PWM-column queries cross-attending to DBD CORE residues only.

    Flank residues are excluded from the contact keys/values by construction (the caller passes
    core_only_mask), per the brief. lambda is learned and initialised small so the model starts as
    prior-only and must earn the correction.

    NOTE: only the 1-D residue-attention pathway is supervised in v26. The 2-D
    PWM-column x residue empirical map is NOT used as a loss -- see
    docs/v26_contact_2d_decision.md (consensus-base match 0.387 vs 0.25 null, unimodal,
    unfilterable). The attention here is still the interpretability read-out.
    """

    def __init__(self, d, max_len=42, n_heads=4, lambda_init=0.1):
        super().__init__()
        self.max_len = max_len
        self.col_embed = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.prior = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, max_len * 4))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.contact_out = nn.Linear(d, 4)
        self.log_lambda = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(lambda_init)))))
        nn.init.zeros_(self.contact_out.weight)
        nn.init.zeros_(self.contact_out.bias)

    def forward(self, z, h_core, core_ignore_mask):
        """z (B,d); h_core (B,L,d) residue states; core_ignore_mask (B,L) True = NOT a core residue."""
        B = z.shape[0]
        z_prior = self.prior(z).view(B, 4, self.max_len)

        m = core_ignore_mask.clone()
        allm = m.all(dim=1)
        if allm.any():
            m[allm, 0] = False
        q = self.col_embed.unsqueeze(0).expand(B, -1, -1)
        ctx, attn_w = self.attn(q, h_core, h_core, key_padding_mask=m, need_weights=True,
                                average_attn_weights=True)
        z_contact = self.contact_out(ctx).transpose(1, 2)          # (B,4,max_len)

        lam = self.log_lambda.exp()
        logits = z_prior + lam * z_contact
        pwm = F.softmax(logits, dim=1)                            # normalise over ACGT per column
        return pwm, {"pwm_logits": logits, "lambda_contact": lam.detach(),
                     "column_residue_attention": attn_w}
