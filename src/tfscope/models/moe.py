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
        self.moe_residual = bool(getattr(config, "moe_residual", True))

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

        output = shared_out + routed_out + proto_out
        if self.moe_residual:
            output = output + x

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


# ── Per-residue fine-grained MoE (DeepSeekMoE-style) ──────────────────────────

class ResidueMoE(nn.Module):
    """Token-level Mixture-of-Experts applied to DBD residue embeddings.

    Motivation (vs the pooled MOEBlock): the pooled MoE makes ONE routing
    decision per protein (~881 total on our data), which is far too few for
    specialization to emerge — every past pooled-MoE run either collapsed to
    uniform routing or, when forced to specialize by CE supervision, LOST
    accuracy. Real MoE (Mixtral, DeepSeek-V2/V3, AIDO.Protein) routes PER TOKEN
    over the sequence, i.e. thousands of decisions per example. This block moves
    the MoE into a per-residue FFN over the DBD so routing has signal to learn
    from, and lets specialization emerge instead of being supervised.

    DeepSeekMoE recipe:
      - `n_shared` always-on shared experts absorb universal base-readout
        chemistry so routed experts stop being redundant (shared isolation).
      - `num_experts` fine-grained routed SwiGLU experts, top-k per token.
      - Router sees the token feature + the protein's family embedding as a soft
        bias (no CE routing supervision — emergent).
      - Standard FFN residual (out = x + shared + routed).

    Load balance is computed at the TOKEN level (defensible, unlike the 41x
    protein-imbalanced pooled case) via the existing load_balance_loss, fed the
    flattened DBD-token gate logits through the model aux dict.
    """

    def __init__(self, config: TFScopeConfig):
        super().__init__()
        d = config.esm_embed_dim
        eh = config.expert_hidden_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.n_shared = config.n_shared_experts

        self.experts = nn.ModuleList(
            [SwiGLUExpert(d, eh) for _ in range(self.num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLUExpert(d, eh) for _ in range(self.n_shared)]
        )
        self.norm = nn.LayerNorm(d)

        self.family_embed = build_family_embedding(config)
        fam_dim = config.family_embed_dim
        # Per-token router: token feature ++ family embedding -> expert logits.
        self.router = nn.Sequential(
            nn.Linear(d + fam_dim, 256), nn.GELU(),
            nn.Linear(256, self.num_experts),
        )
        # Semantic bias — cosine(family_emb, expert prototype); generalises to
        # unseen families (same idea as FamilyAwareGating).
        self.expert_prototypes = nn.Parameter(
            torch.randn(self.num_experts, fam_dim) * 0.02
        )
        self._aux = {}

    def forward(self, x: torch.Tensor, family_id: torch.Tensor,
                dbd_mask: torch.Tensor):
        """
        x:         (B, L, D) residue embeddings (DBD-indicator already added)
        family_id: (B,) integer family labels
        dbd_mask:  (B, L) bool — True at DBD residues

        Returns (refined (B, L, D), aux dict). Only DBD residues are routed and
        updated; non-DBD tokens pass through unchanged.
        """
        B, L, D = x.shape
        xn = self.norm(x)
        fam = self.family_embed(family_id)                       # (B, fam_dim)
        fam_tok = fam.unsqueeze(1).expand(B, L, -1)              # (B, L, fam_dim)

        # Router logits per token (feature + family bias)
        logits = self.router(torch.cat([xn, fam_tok], dim=-1))  # (B, L, E)
        sem_bias = (F.normalize(fam_tok, dim=-1)
                    @ F.normalize(self.expert_prototypes, dim=-1).T)  # (B, L, E)
        logits = logits + sem_bias

        gate_w, top_idx = torch.topk(logits, self.top_k, dim=-1)     # (B, L, k)
        gate_w = F.softmax(gate_w, dim=-1)

        # Shared experts (always on)
        shared_out = torch.zeros_like(x)
        for e in self.shared_experts:
            shared_out = shared_out + e(xn)

        # Routed experts — flatten tokens, process per-expert on its assigned set.
        xf = xn.reshape(B * L, D)
        top_flat = top_idx.reshape(B * L, self.top_k)
        w_flat = gate_w.reshape(B * L, self.top_k)
        routed = torch.zeros_like(xf)
        for k in range(self.top_k):
            for i in range(self.num_experts):
                m = (top_flat[:, k] == i)
                if m.any():
                    routed[m] += w_flat[m, k:k+1] * self.experts[i](xf[m])
        routed = routed.reshape(B, L, D)

        out = x + shared_out + routed                            # FFN residual
        # Only DBD residues are refined; leave the rest as the raw input.
        m = dbd_mask.unsqueeze(-1).to(out.dtype)
        out = m * out + (1.0 - m) * x

        # ── aux for token-level balance loss + interpretability ──────────────
        dm = dbd_mask.reshape(B * L)
        sel = dm.nonzero(as_tuple=True)[0]
        gate_logits_dbd = logits.reshape(B * L, self.num_experts)[sel]  # (N, E)
        top_idx_dbd = top_flat[sel]                                     # (N, k)
        fam_flat = family_id.view(B, 1).expand(B, L).reshape(B * L)[sel]
        self._aux = {
            'gate_logits': gate_logits_dbd,
            'top_indices': top_idx_dbd,
            'family_id':   fam_flat,
            'token_family_id': fam_flat,       # explicit alias for post-hoc analysis
        }
        return out, self._aux

    @property
    def aux_dict(self):
        return self._aux
