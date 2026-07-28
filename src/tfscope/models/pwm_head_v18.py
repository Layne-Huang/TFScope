"""v18 contact-aware, mutation-sensitive PWM head.

Motivation
----------
The legacy ``PWMRegressionHead`` cross-attention from PWM positions to ESM-DBD
residues was shown to be degenerate:

  * rank-1 collapse — every PWM column attends to the same handful of residues;
  * an attention sink on the terminal residue;
  * zero mass on specificity-switching residues (KLF4 K409, MyoD L122), so the
    model is mutation-blind (WT vs mutant PWM Pearson r = 1.0000).

v18 keeps the existing head as a *prior branch* (the family/RAG predictor that
gets the average motif right) and adds a **contact-aware residual branch**:

  logits = z_prior + λ · Δz_contact

The contact branch is engineered to perform genuine residue→base recognition:

  * **cosine attention** (+ LayerNorm on K/V) removes the high-norm hub-residue
    shortcut that caused the sink;
  * **amino-acid-identity values** — V_i carries an explicit embedding of the
    residue identity, so a point mutation moves the value vector and therefore
    Δz, *provided* the column attends to that residue;
  * a small **residual gate** λ keeps the prior intact early in training;
  * anti-collapse regularisers (row-diversity, hub penalty) and an optional
    family-canonical **contact-prior supervision** (v18b) force the attention to
    spread across, and land on, real recognition residues.

The head exposes ``_last_attn`` (B, Lq, Lk) and ``_last_key_mask`` (B, Lk) so the
loss can apply the attention regularisers and contact supervision.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfscope.config import TFScopeConfig
from tfscope.models.heads import PWMRegressionHead


# ESM-2 alphabet size (esm2_t33_650M_UR50D has 33 tokens)
_ESM_VOCAB = 33


class ContactCrossAttention(nn.Module):
    """Cosine cross-attention from PWM-position queries to DBD-residue keys.

    Returns the attended context and the (single-head) attention weights so the
    training loop can regularise / supervise the attention pattern.
    """

    def __init__(self, config: TFScopeConfig):
        super().__init__()
        d = config.pwm_hidden_dim
        self.d = d
        self.n_heads = max(1, int(config.v18_attn_heads))
        assert d % self.n_heads == 0, "pwm_hidden_dim must divide v18_attn_heads"
        self.dh = d // self.n_heads
        self.temp = float(config.v18_cosine_temp)
        self.use_aa_value = bool(config.v18_aa_value)

        # ── hard-sparse normalizer (recognition-contact sparse attention) ──
        self.sparse_mode = str(getattr(config, "v18_attn_sparse", "softmax"))
        if self.sparse_mode == "entmax_learn":
            # learnable per-head alpha in (1, 2]; alpha = 1 + sigmoid(raw) -> raw≈inf gives 2,
            # init so that alpha == v18_attn_alpha_init
            import math
            a0 = float(getattr(config, "v18_attn_alpha_init", 1.5))
            raw0 = math.log((a0 - 1.0) / max(1e-4, 2.0 - a0))   # inverse of 1+sigmoid
            self._alpha_raw = nn.Parameter(torch.full((self.n_heads,), raw0))

        self.kv_norm = nn.LayerNorm(config.esm_embed_dim) if config.v18_kv_layernorm else nn.Identity()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(config.esm_embed_dim, d)
        self.v_proj = nn.Linear(config.esm_embed_dim, d)
        if self.use_aa_value:
            # explicit residue-identity signal so mutations move the value vector
            self.aa_embed = nn.Embedding(_ESM_VOCAB, d)
            nn.init.normal_(self.aa_embed.weight, std=0.02)
        self.out_norm = nn.LayerNorm(d)

        # dual-family: project the fused family vector into a key/value "family token"
        self.use_dual_family = bool(getattr(config, "use_dual_family", False))
        if self.use_dual_family:
            self.fam_kv = nn.Linear(int(getattr(config, "dual_family_dim", 64)), d)

    def _normalize(self, scores):
        """Map attention logits -> weights with the configured normalizer.
        softmax is dense; entmax variants yield exactly-zero weights (sparse contacts).
        Masked entries are -inf, which all variants map to 0."""
        mode = self.sparse_mode
        if mode == "softmax":
            return F.softmax(scores, dim=-1)
        from entmax import entmax15, sparsemax, entmax_bisect
        if mode == "entmax15":
            return entmax15(scores, dim=-1)
        if mode == "sparsemax":
            return sparsemax(scores, dim=-1)
        if mode == "entmax_learn":
            alpha = 1.0 + torch.sigmoid(self._alpha_raw)          # (H,) in (1,2)
            a = alpha.view(1, -1, 1, 1).expand_as(scores[..., :1]).squeeze(-1)  # (B?,H,Lq)
            # entmax_bisect needs alpha broadcastable to scores; use per-(B,H,Lq) tensor
            a = alpha.view(1, self.n_heads, 1, 1).expand(scores.shape[0], self.n_heads,
                                                          scores.shape[2], 1)
            return entmax_bisect(scores, alpha=a, dim=-1, n_iter=24)
        raise ValueError(f"unknown v18_attn_sparse mode: {mode}")

    def forward(self, q, esm, tokens, key_valid, contact_bias=None, fam_ctx=None):
        """
        q:            (B, Lq, d)        queries (one per PWM column)
        esm:          (B, Lk, esm_dim)  ESM residue embeddings
        tokens:       (B, Lk) long      amino-acid token ids (for identity value)
        key_valid:    (B, Lk) bool      True for DBD residues (others masked out)
        contact_bias: (B, Lk) or None   soft additive logit bias (recognition prior)

        Returns:
            ctx:  (B, Lq, d)
            attn: (B, Lq, Lk)   attention weights averaged over heads
        """
        B, Lq, _ = q.shape
        Lk = esm.shape[1]
        H, dh = self.n_heads, self.dh

        kv = self.kv_norm(esm)
        K = self.k_proj(kv)
        V = self.v_proj(kv)
        if self.use_aa_value:
            V = V + self.aa_embed(tokens.clamp(min=0, max=_ESM_VOCAB - 1))
        Q = self.q_proj(q)

        # (B, H, L, dh)
        Qh = Q.view(B, Lq, H, dh).transpose(1, 2)
        Kh = K.view(B, Lk, H, dh).transpose(1, 2)
        Vh = V.view(B, Lk, H, dh).transpose(1, 2)

        # dual-family: prepend a "family token" to keys/values so every PWM
        # position attends to the fused family/semantic signal (deep injection)
        n_fam = 0
        if fam_ctx is not None and getattr(self, "use_dual_family", False):
            fk = self.fam_kv(fam_ctx).view(B, 1, H, dh).transpose(1, 2)   # (B,H,1,dh)
            Kh = torch.cat([fk, Kh], dim=2)
            Vh = torch.cat([fk, Vh], dim=2)
            key_valid = F.pad(key_valid, (1, 0), value=True)             # family token always visible
            if contact_bias is not None:
                contact_bias = F.pad(contact_bias, (1, 0), value=0.0)
            n_fam, Lk = 1, Lk + 1

        # cosine attention logits
        Qn = F.normalize(Qh, dim=-1)
        Kn = F.normalize(Kh, dim=-1)
        scores = self.temp * torch.matmul(Qn, Kn.transpose(-1, -2))   # (B, H, Lq, Lk)

        if contact_bias is not None:
            scores = scores + contact_bias.view(B, 1, 1, Lk)

        mask = (~key_valid).view(B, 1, 1, Lk)
        scores = scores.masked_fill(mask, float('-inf'))
        # guard: a row with no valid key → uniform zeros (sample has no DBD)
        no_valid = (~key_valid).all(dim=1).view(B, 1, 1, 1)
        scores = torch.where(no_valid.expand_as(scores), torch.zeros_like(scores), scores)

        attn = self._normalize(scores)                               # (B, H, Lq, Lk) sparse or dense
        attn = torch.nan_to_num(attn, nan=0.0)
        ctx = torch.matmul(attn, Vh)                                  # (B, H, Lq, dh)
        ctx = ctx.transpose(1, 2).reshape(B, Lq, self.d)
        ctx = self.out_norm(ctx)

        attn_diag = attn.mean(dim=1)                                  # (B, Lq, Lk) average heads
        if n_fam:                                                     # drop family-token column;
            attn_diag = attn_diag[:, :, n_fam:]                       # renormalize over DBD residues
            attn_diag = attn_diag / attn_diag.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return ctx, attn_diag


class PWMHeadV18(nn.Module):
    """Prior branch (legacy head) + contact-aware residual branch."""

    def __init__(self, config: TFScopeConfig):
        super().__init__()
        self.config = config
        self.max_length = config.max_motif_length
        d = config.pwm_hidden_dim

        # ── prior branch: the existing v10/v14 head (family/RAG predictor) ──
        self.prior_head = PWMRegressionHead(config)

        # ── contact-aware residual branch ──
        self.q_pos = nn.Parameter(torch.randn(1, self.max_length, config.pwm_pos_embed_dim) * 0.02)
        self.q_build = nn.Sequential(
            nn.Linear(config.proj_hidden_dim + config.pwm_pos_embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, d),
        )
        self.contact_attn = ContactCrossAttention(config)

        self.use_contact_code = bool(config.v18_contact_code)
        if self.use_contact_code:
            # neural contact-code: Δz value from [context, family] → 4 bases
            self.fam_embed = nn.Embedding(config.num_families, 32)
            self.code_mlp = nn.Sequential(
                nn.LayerNorm(d + 32), nn.Linear(d + 32, d), nn.GELU(), nn.Linear(d, 4)
            )
        else:
            self.contact_out = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4)
            )

        # ── dual-family fusion: learned-id + semantic, gated by homology ──
        self.use_dual_family = bool(getattr(config, "use_dual_family", False))
        if self.use_dual_family:
            from tfscope.models.moe import DualFamilyConditioner, load_semantic_family_vectors
            dcond = int(getattr(config, "dual_family_dim", 64))
            self.fam_cond = DualFamilyConditioner(
                config.num_families, load_semantic_family_vectors(config), dcond)
            self.dual_code_mlp = nn.Sequential(
                nn.LayerNorm(d + dcond), nn.Linear(d + dcond, d), nn.GELU(), nn.Linear(d, 4))

        self.log_lambda = nn.Parameter(torch.tensor(math.log(config.v18_delta_scale_init)))
        # contact-bias scale: fixed float, or a learnable scalar (init = configured value)
        self.bias_learnable = bool(getattr(config, "v18_contact_bias_learnable", False))
        _init = float(config.v18_contact_bias_scale)
        if self.bias_learnable:
            self.bias_scale_param = nn.Parameter(torch.tensor(_init if _init > 0 else 1.0))
            self.bias_scale = None
        else:
            self.bias_scale = _init

        # diagnostics
        self._last_attn = None
        self._last_key_mask = None

    def forward(self, x, esm_embeddings=None, dbd_mask=None, sequence_tokens=None,
                recog_prior=None, family_id=None,
                retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None,
                trust_scores=None, retrieval_reference_mask=None,
                homology=None, family_vec=None, bias_prior=None):
        B = x.shape[0]

        # 1) prior logits from the legacy head (de-novo + optional RAG log-prior)
        z_prior = self.prior_head(
            x, esm_embeddings=esm_embeddings, dbd_mask=dbd_mask,
            retrieved_pwms=retrieved_pwms, retrieved_masks=retrieved_masks,
            retrieved_sims=retrieved_sims, trust_scores=trust_scores,
            retrieval_reference_mask=retrieval_reference_mask,
        )                                                            # (B, 4, L)

        # 2) contact-aware residual
        if esm_embeddings is None or dbd_mask is None:
            self._last_attn = None
            return z_prior

        xq = x.unsqueeze(1).expand(B, self.max_length, -1)
        q = self.q_build(torch.cat([xq, self.q_pos.expand(B, -1, -1)], dim=-1))

        # contact bias: prefer the predicted prior from the contact head (bias_prior);
        # fall back to the supervision recog_prior. recog_prior itself stays the
        # (true-contact) target for the supervision loss — kept separate on purpose.
        contact_bias = None
        prior_for_bias = bias_prior if bias_prior is not None else recog_prior
        bs = self.bias_scale_param if self.bias_learnable else self.bias_scale
        use_bias = (self.bias_learnable or (self.bias_scale is not None and self.bias_scale > 0))
        if use_bias and prior_for_bias is not None:
            contact_bias = bs * prior_for_bias

        if sequence_tokens is None:
            sequence_tokens = torch.zeros(B, esm_embeddings.shape[1],
                                          dtype=torch.long, device=esm_embeddings.device)

        c_fam = None
        if self.use_dual_family:
            c_fam = self.fam_cond(family_id, homology, family_vec)   # (B, dcond)

        ctx, attn = self.contact_attn(
            q, esm_embeddings, sequence_tokens, dbd_mask.bool(), contact_bias, fam_ctx=c_fam)

        if self.use_dual_family:
            ctx_f = torch.cat([ctx, c_fam.unsqueeze(1).expand(B, self.max_length, -1)], dim=-1)
            dz = self.dual_code_mlp(ctx_f).permute(0, 2, 1)          # (B, 4, L)
        elif self.use_contact_code:
            fam = self.fam_embed(family_id) if family_id is not None else \
                torch.zeros(B, 32, device=x.device)
            ctx_f = torch.cat([ctx, fam.unsqueeze(1).expand(B, self.max_length, -1)], dim=-1)
            dz = self.code_mlp(ctx_f).permute(0, 2, 1)               # (B, 4, L)
        else:
            dz = self.contact_out(ctx).permute(0, 2, 1)              # (B, 4, L)

        dz = dz - dz.mean(dim=1, keepdim=True)                       # zero-mean per base (identifiability)
        lam = torch.exp(self.log_lambda)

        self._last_attn = attn                                       # (B, Lq, Lk)
        self._last_key_mask = dbd_mask.bool()
        return z_prior + lam * dz
