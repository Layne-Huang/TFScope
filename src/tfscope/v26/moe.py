"""v26 mixture-of-experts: routed on SEQUENCE, never on family_id.

v24 passed family_id into both the residue-MoE and the pooled MoE
(tfscope.py:150,160) -- audit Finding G. Worse, the family head turned out near-inert
(memory: family-conditioning-vestigial), so the label was supplying leakage-shaped
metadata without buying accuracy.

Here the router sees only (residue state, pooled core state):
    router_i = softmax(W [h_i ; z_core])
Conservative start per the brief: 1 shared expert + 4 routed, top-2, load-balancing loss.
A parameter-matched dense FFN is provided as the control arm.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class DenseFFN(nn.Module):
    """Control arm: same parameter budget as the MoE, no routing."""

    def __init__(self, d, hidden, n_equiv=5, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden * n_equiv), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden * n_equiv, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, z_core, pad_mask=None):
        return self.norm(h + self.net(h)), {"balance_loss": h.new_zeros(())}


class SequenceConditionedMoE(nn.Module):
    def __init__(self, d, hidden=512, n_routed=4, n_shared=1, top_k=2, dropout=0.1):
        super().__init__()
        self.top_k = min(top_k, n_routed)
        self.n_routed = n_routed
        self.shared = nn.ModuleList([Expert(d, hidden, dropout) for _ in range(n_shared)])
        self.routed = nn.ModuleList([Expert(d, hidden, dropout) for _ in range(n_routed)])
        self.router = nn.Linear(2 * d, n_routed)
        self.norm = nn.LayerNorm(d)
        nn.init.zeros_(self.router.bias)

    def forward(self, h, z_core, pad_mask=None):
        """h (C,L,d) residue states; z_core (C,d) pooled core; pad_mask (C,L) True=pad."""
        C, L, d = h.shape
        q = torch.cat([h, z_core.unsqueeze(1).expand(C, L, d)], dim=-1)
        logits = self.router(q)                                   # (C,L,n_routed)
        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True).clamp(min=1e-9)

        out = torch.zeros_like(h)
        for s in self.shared:
            out = out + s(h)
        for k in range(self.top_k):
            idx = topi[..., k]                                    # (C,L)
            w = topv[..., k].unsqueeze(-1)
            for e in range(self.n_routed):
                m = idx.eq(e)
                if m.any():
                    out = out + torch.where(m.unsqueeze(-1), w * self.routed[e](h),
                                            torch.zeros_like(h))

        # load balancing (Switch-style): encourage uniform expert usage over real residues
        if pad_mask is not None:
            valid = (~pad_mask).float().unsqueeze(-1)
            n = valid.sum().clamp(min=1)
            frac = (probs * valid).sum((0, 1)) / n
            hard = F.one_hot(topi[..., 0], self.n_routed).float()
            load = (hard * valid).sum((0, 1)) / n
        else:
            frac = probs.mean((0, 1))
            load = F.one_hot(topi[..., 0], self.n_routed).float().mean((0, 1))
        balance = (frac * load).sum() * self.n_routed

        return self.norm(h + out), {"balance_loss": balance,
                                    "router_probs": probs.detach(),
                                    "top_expert": topi[..., 0].detach()}
