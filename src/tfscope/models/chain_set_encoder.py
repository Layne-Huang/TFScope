"""Permutation-equivariant chain-set encoder (ICLR 2026 Candidate A).

Motivation
----------
The production N-chain path (``two_chain_input`` / ``max_chains``) concatenates
DBD chains and adds a per-token *chain-ID* embedding, so the representation is
**order-aware**: swapping the order of two partners in a heterodimer, or the two
copies of a homodimer, changes the prediction even though the biological complex
is identical. For an unordered set of TF protomers this is a wrong inductive
bias.

This module replaces that with a genuinely set-structured encoder:

1. every chain is projected with the **same** residue projection;
2. residues exchange information *within* a chain (shared parameters, no
   chain-index feature) and *between* chains via a Set-Transformer block over
   chain-pooled summaries that are scattered back to residues by membership;
3. residue-level states are **preserved** for downstream PWM decoding (chains
   are not collapsed to a single vector before inter-chain interaction);
4. no chain-identity embedding is used, so equivalent homomer chains receive
   identical treatment automatically.

Because (a) intra-chain attention uses no chain index, (b) inter-chain mixing
uses a permutation-equivariant set-attention block, and (c) information is
scattered back to residues purely by *membership* (not by chain position), the
per-residue outputs depend only on the multiset of chains, not on their order.
Any permutation-invariant read-out (e.g. the masked mean over all valid
residues used for the span gate) is therefore exactly invariant to chain
reordering — verified numerically in ``tests/test_chain_set_equivariance.py``.

The module is deliberately backbone-agnostic: it consumes per-residue features
(e.g. frozen ESM-2 embeddings) plus a ``chain_ids`` map, so it can wrap the
existing :class:`~tfscope.models.backbone.Backbone` without modification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SelfAttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention + FFN (a Set-Transformer SAB).

    Operates on a set ``X`` of shape ``(B, N, D)`` and is permutation-equivariant
    in ``N``: reordering the ``N`` elements reorders the output identically
    (given the mask is reordered the same way). Supports both a per-pair boolean
    ``attn_mask`` (``True`` = *disallowed*) and a key-padding mask
    (``valid`` = ``True`` for real elements).
    """

    def __init__(self, dim: int, n_heads: int = 4, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.GELU(),
            nn.Linear(ffn_mult * dim, dim),
        )
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, valid: torch.Tensor,
                pair_disallow: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, N, D); valid: (B, N) bool; pair_disallow: (B, N, N) bool or None."""
        B, N, _ = x.shape
        h = self.norm1(x)
        key_padding = ~valid  # (B, N) True = ignore

        attn_mask = None
        if pair_disallow is not None:
            # nn.MultiheadAttention wants a (B*n_heads, N, N) float/bool additive mask.
            attn_mask = pair_disallow.unsqueeze(1).expand(B, self.n_heads, N, N)
            attn_mask = attn_mask.reshape(B * self.n_heads, N, N)

        # A fully-masked query row (all keys disallowed) makes softmax NaN; guard by
        # letting such rows attend to themselves, then zero them out afterwards.
        if attn_mask is not None:
            self_ok = torch.eye(N, dtype=torch.bool, device=x.device).view(1, N, N)
            fully_masked = attn_mask.all(dim=-1, keepdim=True)  # (B*h, N, 1)
            attn_mask = attn_mask & ~(fully_masked & self_ok)

        out, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding if key_padding.any() else None,
            attn_mask=attn_mask,
            need_weights=False,
        )
        out = torch.nan_to_num(out, nan=0.0)
        x = x + out * valid.unsqueeze(-1)
        x = x + self.ffn(self.norm2(x)) * valid.unsqueeze(-1)
        return x


def _masked_chain_pool(h: torch.Tensor, chain_onehot: torch.Tensor) -> torch.Tensor:
    """Mean-pool residues within each chain.

    h: (B, L, D); chain_onehot: (B, L, K) with membership (0/1), padding rows all-0.
    Returns chain summaries (B, K, D); empty chains map to 0.
    """
    # (B, K, L) @ (B, L, D) -> (B, K, D)
    summ = torch.einsum("blk,bld->bkd", chain_onehot, h)
    counts = chain_onehot.sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # (B, K, 1)
    return summ / counts


class ChainSetEncoder(nn.Module):
    """Permutation-equivariant interaction over a set of DBD chains.

    Consumes per-residue features and a chain membership map; returns refined
    per-residue features (same shape) plus a permutation-invariant pooled
    summary suitable for the span gate / global conditioning.
    """

    def __init__(self, config, in_dim: int | None = None):
        super().__init__()
        d = getattr(config, "chain_set_dim", 256)
        n_heads = getattr(config, "chain_set_heads", 4)
        n_layers = getattr(config, "chain_set_layers", 2)
        self.max_chains = getattr(config, "max_chains", 4)
        in_dim = in_dim if in_dim is not None else config.esm_embed_dim

        self.residue_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU()
        )
        self.intra = nn.ModuleList([_SelfAttentionBlock(d, n_heads) for _ in range(n_layers)])
        self.inter = nn.ModuleList([_SelfAttentionBlock(d, n_heads) for _ in range(n_layers)])
        # gated broadcast of chain context back onto residues
        self.chain_gate = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid()) for _ in range(n_layers)
        ])
        self.out_dim = d

    def forward(self, residue_feats: torch.Tensor, chain_ids: torch.Tensor,
                valid: torch.Tensor):
        """
        residue_feats: (B, L, in_dim) per-residue features (e.g. frozen ESM).
        chain_ids:     (B, L) int in [0, max_chains-1] for valid residues; any
                       value is fine for padding (masked out by ``valid``).
        valid:         (B, L) bool — True for real DBD residues.

        Returns
        -------
        h:        (B, L, d) refined per-residue features (0 on padding).
        pooled:   (B, d) permutation-invariant masked mean over valid residues.
        """
        B, L, _ = residue_feats.shape
        K = self.max_chains
        valid = valid.bool()
        cid = chain_ids.clamp(0, K - 1).long()
        # membership one-hot, zeroed on padding → chains carry only valid residues
        chain_onehot = F.one_hot(cid, num_classes=K).to(residue_feats.dtype)
        chain_onehot = chain_onehot * valid.unsqueeze(-1).to(residue_feats.dtype)
        chain_valid = chain_onehot.sum(dim=1) > 0  # (B, K) which chains are populated

        # same-chain pair mask (True = disallowed cross-chain attention within intra step)
        same_chain = cid.unsqueeze(2) == cid.unsqueeze(1)          # (B, L, L)
        pair_disallow = ~same_chain

        h = self.residue_proj(residue_feats) * valid.unsqueeze(-1)

        for intra, inter, gate in zip(self.intra, self.inter, self.chain_gate):
            # (1) intra-chain residue mixing (shared params, no chain index)
            h = intra(h, valid, pair_disallow=pair_disallow)
            # (2) inter-chain mixing over chain summaries (permutation-equivariant)
            summ = _masked_chain_pool(h, chain_onehot)             # (B, K, D)
            summ = inter(summ, chain_valid)                        # set attention among chains
            # (3) scatter chain context back to residues *by membership*
            ctx = torch.einsum("blk,bkd->bld", chain_onehot, summ)  # (B, L, D)
            g = gate(torch.cat([h, ctx], dim=-1))
            h = h + g * ctx
            h = h * valid.unsqueeze(-1)

        denom = valid.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        pooled = (h * valid.unsqueeze(-1)).sum(dim=1) / denom       # (B, d) order-invariant
        return h, pooled

    @staticmethod
    def chain_ids_from_separators(sequence_tokens: torch.Tensor, eos_id: int = 2,
                                  max_chains: int = 4) -> torch.Tensor:
        """Derive per-token chain index from ESM <eos> separators.

        Matches the production convention (chain1=0, partner1=1, …) so this
        encoder can be dropped into the existing N-chain tokenisation.
        """
        separator = sequence_tokens.eq(eos_id)
        return separator.long().cumsum(dim=1).clamp(max=max_chains - 1)


def permutation_consistency_loss(pooled_a: torch.Tensor, pooled_b: torch.Tensor) -> torch.Tensor:
    """MSE between pooled summaries of two chain orderings of the same complex.

    Zero at exact equivariance; used as a soft regulariser during training in
    addition to the architectural guarantee (belt-and-braces per plan §4.7).
    """
    return F.mse_loss(pooled_a, pooled_b)
