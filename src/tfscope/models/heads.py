import torch
import torch.nn as nn
import torch.nn.functional as F

from tfscope.config import TFScopeConfig
from tfscope.models.alignment import align_batch_torch


class ContactPredHead(nn.Module):
    """Frozen linear probe: ESM residue embedding → P(DNA-contact) per residue.

    Warm-started from the saved LogisticRegression contact probe (ESM-2 layer-33,
    OOF AUROC 0.946). At every forward pass it reads the model's own ESM
    embeddings and emits a per-residue contact prior, so the v18 contact bias
    works for ANY sequence (seen, held-out, or de novo design) with no external
    structure — the sequence-only replacement for a co-crystal-derived prior.
    Frozen by default (matches "use the probe as-is"); the learnable bias scale
    downstream calibrates how much the model trusts it.
    """

    def __init__(self, config: TFScopeConfig):
        super().__init__()
        self.linear = nn.Linear(config.esm_embed_dim, 1)
        path = getattr(config, "contact_probe_path", "")
        if path and __import__("os").path.isfile(path):
            import joblib
            d = joblib.load(path)
            clf = d["model"]
            W = torch.tensor(clf.coef_, dtype=torch.float32)          # (1, D)
            b = torch.tensor(clf.intercept_, dtype=torch.float32)     # (1,)
            with torch.no_grad():
                self.linear.weight.copy_(W)
                self.linear.bias.copy_(b)
            print(f"[ContactPredHead] warm-started from {path} (dim {W.shape[1]})")
        else:
            print(f"[ContactPredHead] probe path not found ({path}) — random init")
        for p in self.parameters():                                  # frozen
            p.requires_grad = False

    def forward(self, esm_embeddings: torch.Tensor) -> torch.Tensor:
        """(B, L, D) → (B, L) per-residue contact probability in [0, 1]."""
        return torch.sigmoid(self.linear(esm_embeddings).squeeze(-1))


class PositionGateHead(nn.Module):
    """Predict motif occupancy with a legacy or contiguous-span parameterization.

    ``independent`` preserves checkpoint compatibility with the original
    per-position MLP. ``span`` predicts differentiable start and length values
    and renders them into one contiguous soft interval. The latter removes
    fragmented masks by construction and lets length gradients flow directly.
    """

    def __init__(self, config: TFScopeConfig):
        super().__init__()
        self.max_length = config.max_motif_length
        self.min_length = min(config.min_motif_length, self.max_length)
        self.mode = getattr(config, "gate_mode", "independent")
        self.temperature = getattr(config, "span_gate_temperature", 0.5)
        if self.mode not in {"independent", "span"}:
            raise ValueError(f"Unknown gate_mode={self.mode!r}")
        output_dim = self.max_length if self.mode == "independent" else 2
        self.net = nn.Sequential(
            nn.Linear(config.proj_hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )
        self._last_span = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, max_length) gate logits (pre-sigmoid)."""
        raw = self.net(x)
        if self.mode == "independent":
            self._last_span = None
            return raw

        # A continuous interval constrained to fit in [0, max_length].
        max_start = max(self.max_length - self.min_length, 0)
        start = max_start * torch.sigmoid(raw[:, 0])
        available = self.max_length - start
        length = self.min_length + (available - self.min_length) * torch.sigmoid(
            raw[:, 1]
        )
        end = start + length

        positions = torch.arange(
            self.max_length, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        temp = max(float(self.temperature), 1e-3)
        left = torch.sigmoid((positions + 0.5 - start.unsqueeze(1)) / temp)
        right = torch.sigmoid((end.unsqueeze(1) - positions - 0.5) / temp)
        occupancy = (left * right).clamp(1e-6, 1.0 - 1e-6)
        self._last_span = {
            "start": start,
            "length": length,
            "end": end,
        }
        return torch.logit(occupancy)


class PWMRegressionHead(nn.Module):
    def __init__(self, config: TFScopeConfig):
        super().__init__()
        self.max_length = config.max_motif_length
        self.config = config

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.max_length, config.pwm_pos_embed_dim) * 0.02
        )

        self.pos_projection = nn.Sequential(
            nn.Linear(config.proj_hidden_dim + config.pwm_pos_embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.pwm_hidden_dim),
            nn.GELU(),
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.pwm_hidden_dim, num_heads=config.pwm_attn_heads,
            dropout=0.1, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(config.pwm_hidden_dim)

        # Cross-attention: each PWM position attends to ESM2 DBD residues.
        # Each position i learns which specific amino acids drive its nucleotide preference.
        if config.pwm_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=config.pwm_hidden_dim,
                num_heads=config.pwm_attn_heads,
                kdim=config.esm_embed_dim,
                vdim=config.esm_embed_dim,
                dropout=0.1,
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(config.pwm_hidden_dim)
        else:
            self.cross_attn = None

        self.use_retrieval  = getattr(config, "use_retrieval", False)
        self.residual_prior = getattr(config, "residual_prior", False)
        self.positionwise_retrieval_gate = getattr(
            config, "positionwise_retrieval_gate", False
        )
        self.align_retrieved_pwms = getattr(
            config, "align_retrieved_pwms", False
        )

        if self.use_retrieval and not self.residual_prior:
            # v9/v10: de-novo + confidence-gated log-prior
            d = config.pwm_hidden_dim
            self.retrieval_beta = nn.Parameter(torch.ones(1))
            self.neighbour_proj = nn.Linear(4, d)
            self.k_embed_nb     = nn.Embedding(config.retrieval_k, d)
            self.pos_embed_nb   = nn.Embedding(config.max_motif_length, d)
            self.conf_scale  = nn.Parameter(torch.tensor(10.0))
            self.conf_thresh = nn.Parameter(torch.tensor(0.5))

        if self.use_retrieval and self.residual_prior:
            # v12: output = log(prior) + alpha * delta(protein, prior)
            # log_alpha starts at -2 → alpha≈0.13, so model starts near pure retrieval
            # and gradually learns to add protein-driven corrections.
            d = config.pwm_hidden_dim
            self.log_alpha = nn.Parameter(torch.tensor(-2.0))
            # Encode each prior column (4 bases) → hidden for cross-attention
            self.prior_proj = nn.Linear(4, d)
            self.prior_cross_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=config.pwm_attn_heads,
                dropout=0.1, batch_first=True,
            )
            self.prior_cross_attn_norm = nn.LayerNorm(d)

        self.nucleotide_head = nn.Linear(config.pwm_hidden_dim, 4)  # A, C, G, T

    @staticmethod
    def _aggregate_prior(retrieved_pwms, retrieved_masks, retrieved_sims):
        """Similarity-weighted average of K retrieved PWMs → (B, 4, L) prior."""
        B, K, _, L = retrieved_pwms.shape
        valid = (retrieved_masks.sum(dim=2) > 0).float()
        w = retrieved_sims.clamp(min=0.0) * valid
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        prior = (retrieved_pwms * w.view(B, K, 1, 1)).sum(dim=1)
        prior = prior / prior.sum(dim=1, keepdim=True).clamp(min=1e-8)
        has_any = (valid.sum(dim=1) > 0).view(B, 1, 1)
        return torch.where(has_any, prior, torch.full_like(prior, 0.25))

    def forward(self, x: torch.Tensor,
                esm_embeddings: torch.Tensor = None,
                dbd_mask: torch.Tensor = None,
                retrieved_pwms: torch.Tensor = None,
                retrieved_masks: torch.Tensor = None,
                retrieved_sims: torch.Tensor = None,
                trust_scores: torch.Tensor = None,
                retrieval_reference_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:                (B, hidden_dim)            MoE output
            esm_embeddings:   (B, L, esm_embed_dim)      ESM2 residue embeddings (optional)
            dbd_mask:         (B, L) bool                True for DBD positions (optional)
            retrieved_pwms:   (B, K, 4, max_length)      raw retrieved-neighbour PWMs (optional)
            retrieved_masks:  (B, K, max_length)         1.0 = valid neighbour position (optional)
            retrieved_sims:   (B, K)                     cosine sims of each neighbour (optional)

        Returns:
            logits: (B, 4, max_length)
        """
        import torch.nn.functional as F
        B = x.shape[0]

        x_expanded = x.unsqueeze(1).expand(B, self.max_length, -1)
        pos = self.pos_embed.expand(B, -1, -1)
        h = self.pos_projection(torch.cat([x_expanded, pos], dim=-1))

        # Self-attention across PWM positions
        h_norm = self.attn_norm(h)
        h_attn, _ = self.self_attention(h_norm, h_norm, h_norm)
        h = h + h_attn

        # Cross-attention to ESM2 DBD embeddings
        if self.cross_attn is not None and esm_embeddings is not None:
            key_padding_mask = None
            if dbd_mask is not None:
                key_padding_mask = ~dbd_mask.bool()
            h_cross, _ = self.cross_attn(
                self.cross_attn_norm(h),
                esm_embeddings,
                esm_embeddings,
                key_padding_mask=key_padding_mask,
            )
            h = h + h_cross

        # ── v12 residual-prior path ──────────────────────────────────────────
        if self.use_retrieval and self.residual_prior and retrieved_pwms is not None:
            # 1. Aggregate neighbours → probability prior (B, 4, L)
            prior = self._aggregate_prior(retrieved_pwms, retrieved_masks, retrieved_sims)

            # 2. Cross-attend PWM positions to prior columns so delta can see
            #    what the retrieval prior suggests at each position
            prior_keys = self.prior_proj(prior.permute(0, 2, 1))    # (B, L, d)
            h_p, _ = self.prior_cross_attn(
                self.prior_cross_attn_norm(h), prior_keys, prior_keys
            )
            h = h + h_p

            # 3. Protein-driven delta on top of prior
            delta = self.nucleotide_head(h).permute(0, 2, 1)        # (B, 4, L)

            # 4. Residual output: log(prior) + alpha * delta
            #    When alpha→0 or delta≈0, output collapses to the prior.
            log_prior = torch.log(prior.clamp(min=1e-6))             # (B, 4, L)
            alpha = torch.exp(self.log_alpha)                        # scalar
            return log_prior + alpha * delta

        delta_logits = self.nucleotide_head(h).permute(0, 2, 1)    # (B, 4, L)

        # ── v9/v10: de-novo + confidence-gated log-prior ─────────────────────
        if self.use_retrieval and retrieved_pwms is not None:
            Bk, K, _, L = retrieved_pwms.shape
            assert Bk == B and L == self.max_length
            if self.align_retrieved_pwms:
                reference_mask = retrieval_reference_mask
                if reference_mask is None:
                    reference_mask = torch.ones(
                        B, L, device=delta_logits.device, dtype=delta_logits.dtype
                    )
                retrieved_pwms, retrieved_masks = align_batch_torch(
                    retrieved_pwms,
                    retrieved_masks,
                    F.softmax(delta_logits.detach(), dim=1),
                    reference_mask,
                    max_shift=getattr(
                        self.config, "retrieval_alignment_max_shift", 10
                    ),
                    min_overlap=getattr(
                        self.config, "retrieval_alignment_min_overlap", 4
                    ),
                    return_masks=True,
                )

            # Encode each neighbour column: (B, K, L, 4) → (B, K, L, d)
            nb = self.neighbour_proj(retrieved_pwms.permute(0, 1, 3, 2))
            k_ids   = torch.arange(K, device=nb.device).view(1, K, 1).expand(B, K, L)
            pos_ids = torch.arange(L, device=nb.device).view(1, 1, L).expand(B, K, L)
            nb = nb + self.k_embed_nb(k_ids) + self.pos_embed_nb(pos_ids)        # (B, K, L, d)

            # Per-position attention scores: at each L, dot(h[B,L,d], nb[B,K,L,d]) → (B,L,K)
            scores = torch.einsum('bld,bkld->blk', h, nb)
            valid_pos = (retrieved_masks > 0.5).permute(0, 2, 1)                 # (B, L, K)
            scores = scores.masked_fill(~valid_pos, float('-inf'))
            # Guard: position where ALL K are invalid → softmax over -inf = NaN.
            # Replace with zeros (uniform attn) and we'll zero contribution below via has_any_pos.
            has_any_pos = valid_pos.any(dim=-1)                                  # (B, L)
            scores = torch.where(has_any_pos.unsqueeze(-1).expand_as(scores),
                                  scores, torch.zeros_like(scores))
            attn = F.softmax(scores, dim=-1)                                     # (B, L, K)
            base_attn = attn

            # v10: weight attention by learned trust scores (per-neighbour).
            # Untrustworthy neighbours get down-weighted at attention time.
            if trust_scores is not None:
                # trust_scores: (B, K) in [0,1]
                attn = attn * trust_scores.unsqueeze(1)                           # (B, L, K)
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            # Combined log-prior (per-position weighted sum of K log-PWMs)
            log_pwms = torch.log(retrieved_pwms.clamp(min=1e-6))                 # (B, K, 4, L)
            combined_log = torch.einsum('blk,bkcl->bcl', attn, log_pwms)         # (B, 4, L)
            pos_gate = has_any_pos.float().unsqueeze(1)                           # (B, 1, L)
            combined_log = combined_log * pos_gate

            # β gate: v10 uses learned trust instead of cos-sim. If no neighbour is
            # trustworthy → β → 0 → fall back on de-novo sequence/delta pathway.
            donor_confidence = (
                trust_scores
                if trust_scores is not None
                else retrieved_sims.clamp(min=0.0)
            )
            if self.positionwise_retrieval_gate:
                # Expected donor confidence under the sequence-conditioned
                # attention at each motif position. This avoids applying one
                # strong donor's prior to unsupported columns elsewhere.
                gate_input = torch.einsum(
                    "blk,bk->bl", base_attn, donor_confidence
                )                                                               # (B, L)
                gate_input = gate_input * has_any_pos.float()
            else:
                gate_input = donor_confidence.max(dim=-1).values                 # (B,)
            beta_gated = self.retrieval_beta * torch.sigmoid(
                self.conf_scale * (gate_input - self.conf_thresh))
            if self.positionwise_retrieval_gate:
                beta_gated = beta_gated * has_any_pos.float()                    # (B, L)
            else:
                sample_has_any = (retrieved_masks.sum(dim=(1, 2)) > 0).float()
                beta_gated = beta_gated * sample_has_any                         # (B,)

            # stash for diagnostics (Feature 6); guard against NaN
            combined_log = torch.nan_to_num(combined_log, nan=0.0, posinf=0.0, neginf=0.0)
            self._last_beta_gated = beta_gated.detach()
            if self.positionwise_retrieval_gate:
                return delta_logits + beta_gated.unsqueeze(1) * combined_log
            return delta_logits + beta_gated.view(-1, 1, 1) * combined_log

        return delta_logits
