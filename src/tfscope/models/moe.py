import torch
import torch.nn as nn
import torch.nn.functional as F

from tfscope.config import TFScopeConfig


class ExpertMLP(nn.Module):
    def __init__(self, hidden_dim: int, expert_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expert_hidden),
            nn.GELU(),
            nn.Linear(expert_hidden, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUExpert(nn.Module):
    """SwiGLU expert: W_down(silu(W_gate·x) ⊙ W_up·x).

    From DeepSeek-V2/V3 and LLaMA. More expressive than a 2-layer MLP:
    the multiplicative gate selects which features each expert passes through.
    """

    def __init__(self, hidden_dim: int, expert_hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, expert_hidden, bias=False)
        self.w_up   = nn.Linear(hidden_dim, expert_hidden, bias=False)
        self.w_down = nn.Linear(expert_hidden, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class PrototypeDictionary(nn.Module):
    """Learnable dictionary of binding concepts, ProtoT-inspired (arXiv 2602.11852).

    Provides interpretability: at inference, prototype_weights[i] shows how
    much binding concept i contributes to TF i's representation.
    Each prototype can be decoded through the shared PWM head to visualise
    the nucleotide preference pattern it encodes.
    """

    def __init__(self, hidden_dim: int, n_prototypes: int):
        super().__init__()
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, hidden_dim) * 0.02
        )
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(self, x: torch.Tensor):
        """Returns (retrieved, weights): retrieved is (B, hidden), weights is (B, n_proto)."""
        q = self.query_proj(x)                                 # (B, hidden)
        attn = q @ self.prototypes.T * self.scale              # (B, n_proto)
        weights = F.softmax(attn, dim=-1)                      # (B, n_proto)
        retrieved = weights @ self.prototypes                  # (B, hidden)
        return retrieved, weights


# ── Family embedding ──────────────────────────────────────────────────────────

class LearnedFamilyEmbedding(nn.Module):
    """Fallback discrete embedding — used when no pre-computed vectors exist."""
    def __init__(self, num_families: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_families, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, family_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(family_id)


class SemanticFamilyEmbedding(nn.Module):
    """Family embedding from pre-computed ProTrek text + ESM-2 sequence vectors.

    The frozen buffer holds one vector per family (text ++ seq embeddings).
    A small trainable projection adapts it to the model's family_embed_dim.
    Works for unseen families at LOFO time as long as a description exists.
    """

    def __init__(self, family_vectors: torch.Tensor, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        in_dim = family_vectors.shape[1]
        self.register_buffer("vectors", family_vectors)   # (F, in_dim) — frozen
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, family_id: torch.Tensor,
                family_vec: torch.Tensor = None) -> torch.Tensor:
        """Project a family vector to the model's family_embed_dim.

        family_id : (B,) index into the frozen 34-family buffer (the trained path).
        family_vec: (B, in_dim) optional raw text++ESM-2 vector for an UNSEEN family
                    not in the buffer — built on the fly from a description and/or the
                    protein's mean ESM-2 embedding. When given, it bypasses the lookup
                    so any novel family routes through the same trained projection +
                    semantic gate. Must match the buffer's in_dim.
        """
        if family_vec is not None:
            return self.proj(family_vec.to(self.vectors.dtype))
        return self.proj(self.vectors[family_id])


def build_family_embedding(config: TFScopeConfig) -> nn.Module:
    """Return SemanticFamilyEmbedding if the pre-computed file exists,
    otherwise fall back to LearnedFamilyEmbedding."""
    path = getattr(config, "family_embedding_path", None)
    if path and __import__("os").path.isfile(path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        vectors = data["embeddings"].float()          # (F, text_dim + seq_dim)
        print(f"[MoE] Loaded semantic family embeddings {vectors.shape} from {path}")
        return SemanticFamilyEmbedding(vectors, config.family_embed_dim)
    else:
        if path:
            print(f"[MoE] family_embedding_path not found ({path}) — using learned embedding")
        return LearnedFamilyEmbedding(config.num_families, config.family_embed_dim)


def load_semantic_family_vectors(config: TFScopeConfig):
    """Load the (F, in_dim) semantic family vectors for the dual-family head.
    Prefers dual_family_semantic_path (kept separate from the MoE's family_embedding_path
    so the MoE can stay learned); falls back to family_embedding_path."""
    import os
    path = getattr(config, "dual_family_semantic_path", "") or getattr(config, "family_embedding_path", None)
    if path and os.path.isfile(path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        return data["embeddings"].float()
    return None


class DualFamilyConditioner(nn.Module):
    """Fuse learned-id (in-distribution identity) + semantic (text/ESM-2) family
    signals into one conditioning vector, gated by homology (e.g. top retrieval
    cosine): high homology -> lean learned; low homology / OOD -> lean semantic.
    `family_vec` lets callers pass an arbitrary semantic vector at inference
    (e.g. a design's nearest-homolog text), bypassing the id lookup.
    """

    def __init__(self, num_families: int, semantic_vectors, d_cond: int):
        super().__init__()
        self.learned = nn.Embedding(num_families, d_cond)
        nn.init.normal_(self.learned.weight, std=0.02)
        self.has_semantic = semantic_vectors is not None
        if self.has_semantic:
            self.register_buffer("sem_vectors", semantic_vectors)          # (F, in_dim) frozen
            self.sem_proj = nn.Sequential(
                nn.Linear(semantic_vectors.shape[1], d_cond), nn.GELU(),
                nn.LayerNorm(d_cond))
        self.gate = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, family_id, homology=None, family_vec=None):
        cl = self.learned(family_id)                                       # (B, d_cond)
        if not self.has_semantic:
            return cl
        sv = family_vec if family_vec is not None else self.sem_vectors[family_id]
        cs = self.sem_proj(sv.to(cl.dtype))                                # (B, d_cond)
        h = homology if homology is not None else cl.new_ones(cl.size(0), 1)
        g = torch.sigmoid(self.gate(h.to(cl.dtype)))                       # (B,1) ~1 in-dist
        return g * cl + (1.0 - g) * cs


# ── FiLM conditioning ─────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    def __init__(self, feature_dim: int, family_embed_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(family_embed_dim, feature_dim)
        self.beta_net  = nn.Linear(family_embed_dim, feature_dim)
        nn.init.ones_(self.gamma_net.weight);  nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight);  nn.init.zeros_(self.beta_net.bias)

    def forward(self, features: torch.Tensor,
                family_embed: torch.Tensor) -> torch.Tensor:
        gamma = 1.0 + self.gamma_net(family_embed)
        beta  = self.beta_net(family_embed)
        return gamma * features + beta


# ── Gating with semantic routing bias ────────────────────────────────────────

class FamilyAwareGating(nn.Module):
    """MoE gate with a semantic routing bias.

    Replaces the old nn.Embedding(num_families, num_experts) family_bias
    with a dot-product between the family embedding and per-expert prototype
    vectors.  This generalises continuously to unseen families — the gate
    naturally routes a new family to experts that already handle
    semantically similar families.
    """

    def __init__(self, input_dim: int, family_embed_dim: int, num_experts: int):
        super().__init__()
        self.projection = nn.Linear(input_dim + family_embed_dim, 256)
        self.gate = nn.Linear(256, num_experts)
        # Learnable expert prototypes in family semantic space
        self.expert_prototypes = nn.Parameter(
            torch.randn(num_experts, family_embed_dim) * 0.02
        )

    def forward(self, x: torch.Tensor, family_embed: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.projection(torch.cat([x, family_embed], dim=-1)))
        logits = self.gate(h)
        # Semantic bias: cosine similarity between family embedding and expert prototypes
        semantic_bias = (F.normalize(family_embed, dim=-1)
                         @ F.normalize(self.expert_prototypes, dim=-1).T)  # (B, E)
        return logits + semantic_bias


# ── MOE block ─────────────────────────────────────────────────────────────────

class MOEBlock(nn.Module):
    def __init__(self, config: TFScopeConfig):
        super().__init__()
        self.config      = config
        self.num_experts = config.num_experts
        self.top_k       = config.top_k

        # Routed experts (SwiGLU, DeepSeek-V2/V3)
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.proj_hidden_dim, config.expert_hidden_dim)
            for _ in range(config.num_experts)
        ])

        # Shared experts — always active, capture universal binding features
        self.shared_experts = nn.ModuleList([
            SwiGLUExpert(config.proj_hidden_dim, config.expert_hidden_dim)
            for _ in range(config.n_shared_experts)
        ])

        # Prototype dictionary — interpretable binding concepts (ProtoT, arXiv 2602.11852)
        self.proto = (
            PrototypeDictionary(config.proj_hidden_dim, config.n_prototypes)
            if config.n_prototypes > 0 else None
        )

        self.family_embed = build_family_embedding(config)
        self.film   = FiLMLayer(config.proj_hidden_dim, config.family_embed_dim)
        self.gating = FamilyAwareGating(
            config.proj_hidden_dim, config.family_embed_dim, config.num_experts
        )
        self._aux = {}

    def forward(self, x: torch.Tensor, family_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         (B, hidden_dim)
            family_id: (B,) integer family labels

        Returns:
            output: (B, hidden_dim)
        """
        family_emb = self.family_embed(family_id)              # (B, family_embed_dim)

        # ── Shared experts (always fire) ──────────────────────────────────────
        shared_out = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x)

        # ── Top-k routed experts ──────────────────────────────────────────────
        gate_logits  = self.gating(x, family_emb)             # (B, num_experts)
        gate_weights, top_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        gate_weights = F.softmax(gate_weights, dim=-1)         # (B, top_k)

        routed_out = torch.zeros_like(x)
        for k in range(self.top_k):
            for i in range(self.num_experts):
                mask = (top_indices[:, k] == i)
                if mask.any():
                    h = self.experts[i](x[mask])
                    h = self.film(h, family_emb[mask])
                    routed_out[mask] += gate_weights[mask, k:k+1] * h

        # ── Prototype retrieval (interpretability) ────────────────────────────
        proto_weights = None
        proto_out = torch.zeros_like(x)
        if self.proto is not None:
            proto_out, proto_weights = self.proto(x)          # (B, hidden), (B, n_proto)

        output = x + shared_out + routed_out + proto_out

        self._aux = {
            'gate_logits':    gate_logits,
            'top_indices':    top_indices,
            'family_id':      family_id,
            'proto_weights':  proto_weights,               # (B, n_proto) or None
        }
        return output

    @property
    def aux_dict(self):
        return self._aux
