import torch
import torch.nn as nn
import torch.nn.functional as F

from tfscope.config import TFScopeConfig


class AttentionPooling(nn.Module):
    """Multi-head attention pooling with a learned query vector."""

    def __init__(self, config: TFScopeConfig):
        super().__init__()
        d_model = config.esm_embed_dim
        self.n_heads = config.pool_n_heads
        self.d_head = d_model // self.n_heads

        self.query = nn.Parameter(torch.randn(self.n_heads, config.pool_d_query))
        self.query_proj = nn.Linear(config.pool_d_query, self.d_head, bias=False)
        self.key_proj = nn.Linear(d_model, self.d_head, bias=False)
        self.value_proj = nn.Linear(d_model, self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.d_head ** -0.5

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            embeddings: (B, L, d_model)
            mask: (B, L) boolean, True for valid positions. None = all valid.

        Returns:
            pooled: (B, d_model)
        """
        q = self.query_proj(self.query)                        # (n_heads, d_head)
        k = self.key_proj(embeddings)                          # (B, L, d_head)
        v = self.value_proj(embeddings)                        # (B, L, d_head)

        attn = torch.einsum('hd,bld->bhl', q, k) * self.scale  # (B, n_heads, L)

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).bool(), -1e9)

        attn_weights = F.softmax(attn, dim=-1)                 # (B, n_heads, L)
        pooled = torch.einsum('bhl,bld->bhd', attn_weights, v) # (B, n_heads, d_head)
        pooled = pooled.reshape(pooled.shape[0], -1)           # (B, d_model)

        return self.out_proj(pooled)


class GatedAttentionPooling(nn.Module):
    """Multi-head attention pooling with per-position sigmoid gate on values.

    Inspired by 'Gated Attention for Large Language Models: Non-linearity,
    Sparsity, and Attention-Sink-Free' (NeurIPS 2024 Best Paper).
    The sigmoid gate suppresses uninformative residues in value space,
    preventing any single position from dominating (attention sink).
    """

    def __init__(self, config: TFScopeConfig):
        super().__init__()
        d_model = config.esm_embed_dim
        self.n_heads = config.pool_n_heads
        self.d_head = d_model // self.n_heads

        self.query    = nn.Parameter(torch.randn(self.n_heads, config.pool_d_query))
        self.query_proj = nn.Linear(config.pool_d_query, self.d_head, bias=False)
        self.key_proj   = nn.Linear(d_model, self.d_head, bias=False)
        self.value_proj = nn.Linear(d_model, self.d_head, bias=False)
        self.gate_proj  = nn.Linear(d_model, self.d_head, bias=False)  # sigmoid gate
        self.out_proj   = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.d_head ** -0.5

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        q = self.query_proj(self.query)                        # (n_heads, d_head)
        k = self.key_proj(embeddings)                          # (B, L, d_head)
        v = self.value_proj(embeddings)                        # (B, L, d_head)
        g = torch.sigmoid(self.gate_proj(embeddings))          # (B, L, d_head)

        attn = torch.einsum('hd,bld->bhl', q, k) * self.scale  # (B, n_heads, L)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).bool(), -1e9)
        attn_weights = F.softmax(attn, dim=-1)                 # (B, n_heads, L)

        gated_v = g * v                                        # (B, L, d_head)
        pooled = torch.einsum('bhl,bld->bhd', attn_weights, gated_v)  # (B, n_heads, d_head)
        pooled = pooled.reshape(pooled.shape[0], -1)           # (B, d_model)
        return self.out_proj(pooled)


class ProjectionHead(nn.Module):
    """Projects concatenated global + DBD features to hidden dim."""

    def __init__(self, config: TFScopeConfig):
        super().__init__()
        input_dim = config.esm_embed_dim * 2  # concat of global + dbd
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.proj_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.proj_hidden_dim),
            nn.Dropout(config.proj_dropout),
        )

    def forward(self, global_feat: torch.Tensor, dbd_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_feat: (B, embed_dim)
            dbd_feat: (B, embed_dim)

        Returns:
            projected: (B, proj_hidden_dim)
        """
        return self.net(torch.cat([global_feat, dbd_feat], dim=-1))
