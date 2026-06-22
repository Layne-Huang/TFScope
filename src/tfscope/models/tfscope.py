import torch
import torch.nn as nn

from tfscope.config import TFScopeConfig
from tfscope.models.backbone import Backbone, DummyBackbone
from tfscope.models.pooling import GatedAttentionPooling, ProjectionHead
from tfscope.models.moe import MOEBlock
from tfscope.models.heads import PositionGateHead, PWMRegressionHead
from tfscope.models.pwm_head_v18 import PWMHeadV18
from tfscope.models.retrieval import TrustPredictor
from tfscope.models.register_head import RegisterHead
from tfscope.losses.registration import export_registered_predictions


class TFScopeModel(nn.Module):
    def __init__(self, config: TFScopeConfig, use_dummy_backbone: bool = False):
        super().__init__()
        self.config = config

        # Encoder
        if use_dummy_backbone:
            self.backbone = DummyBackbone(config)
        else:
            self.backbone = Backbone(config)

        # DBD position indicator — learned embedding added to DBD residues before pooling
        self.dbd_indicator = nn.Parameter(torch.zeros(1, 1, config.esm_embed_dim))

        # Dual-stream gated attention pooling (NeurIPS 2024 best paper)
        self.global_pool = GatedAttentionPooling(config)
        self.dbd_pool    = GatedAttentionPooling(config)
        self.projection  = ProjectionHead(config)

        # MOE
        self.moe = MOEBlock(config)

        # Output heads
        self.gate_head = PositionGateHead(config)   # replaces MotifLengthHead
        self.use_v18 = getattr(config, "pwm_head_v18", False)
        self.pwm_head = PWMHeadV18(config) if self.use_v18 else PWMRegressionHead(config)
        self.use_register_head = getattr(config, "register_head", False)
        if self.use_register_head:
            self.register_head = RegisterHead(config)

        self.use_retrieval = getattr(config, "use_retrieval", False)
        if self.use_retrieval:
            # v10: learned trust predictor — replaces the cos-sim-based confidence gate.
            self.trust_predictor = TrustPredictor(
                query_dim=config.proj_hidden_dim,
                max_motif_length=config.max_motif_length,
                hidden_dim=128,
            )

    @staticmethod
    def _aggregate_prior(retrieved_pwms, retrieved_masks, retrieved_sims):
        """Similarity-weighted average of K retrieved PWMs → (B, 4, L) prior.
        Falls back to uniform 0.25 where a sample has no valid neighbour."""
        B, K, _, L = retrieved_pwms.shape
        valid = (retrieved_masks.sum(dim=2) > 0).float()              # (B, K)
        w = retrieved_sims.clamp(min=0.0) * valid                     # (B, K)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)            # normalise
        prior = (retrieved_pwms * w.view(B, K, 1, 1)).sum(dim=1)      # (B, 4, L)
        # renormalise columns to sum to 1
        prior = prior / prior.sum(dim=1, keepdim=True).clamp(min=1e-8)
        # samples with no valid neighbour → uniform
        has_any = (valid.sum(dim=1) > 0).view(B, 1, 1)
        uniform = torch.full_like(prior, 0.25)
        return torch.where(has_any, prior, uniform)

    def forward(self, sequence_tokens, dbd_mask, family_id,
                retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None,
                recog_prior=None, family_vec=None, homology=None):
        """
        Args:
            sequence_tokens: (B, L) tokenized protein sequence
            dbd_mask: (B, L) boolean, True for DBD positions
            family_id: (B,) integer family labels

        Returns:
            gate_logits: (B, max_motif_length) pre-sigmoid position gate logits
            pwm_logits:  (B, 4, max_motif_length) nucleotide logits (all positions)
            aux: dict with gate_logits, top_indices, family_id
        """
        embeddings = self.backbone(sequence_tokens)               # (B, L, embed_dim)

        # Inject DBD indicator: signal to pooling which residues are in the DBD
        dbd_emb = embeddings + dbd_mask.unsqueeze(-1).float() * self.dbd_indicator

        global_feat = self.global_pool(dbd_emb)                  # (B, embed_dim)
        dbd_feat    = self.dbd_pool(dbd_emb, dbd_mask)           # (B, embed_dim)
        combined    = self.projection(global_feat, dbd_feat)     # (B, hidden_dim)

        moe_out = self.moe(combined, family_id)                   # (B, hidden_dim)

        gate_logits = self.gate_head(moe_out)                     # (B, max_length)

        # Retrieval inputs (v10): trust-gated ICL.
        trust_logits = None
        ret_pwms, ret_masks, ret_sims = None, None, None
        if self.use_retrieval and retrieved_pwms is not None:
            # Classifier-free guidance: occasionally zero retrieval to keep de-novo pathway functional.
            drop_p = self.config.retrieval_dropout
            if self.training and drop_p > 0:
                B = retrieved_pwms.shape[0]
                drop = (torch.rand(B, device=retrieved_pwms.device) < drop_p)
                if drop.any():
                    retrieved_masks = retrieved_masks.clone()
                    retrieved_masks[drop] = 0.0
                    retrieved_sims  = retrieved_sims.clone()
                    retrieved_sims[drop]  = 0.0
            # Predict per-neighbour trust (B, K); supervised by auxiliary loss at training.
            trust_logits = self.trust_predictor(combined, retrieved_pwms, retrieved_sims)
            ret_pwms, ret_masks, ret_sims = retrieved_pwms, retrieved_masks, retrieved_sims

        head_kwargs = dict(
            esm_embeddings=embeddings,
            dbd_mask=dbd_mask,
            retrieved_pwms=ret_pwms,
            retrieved_masks=ret_masks,
            retrieved_sims=ret_sims,
            trust_scores=(torch.sigmoid(trust_logits) if trust_logits is not None else None),
            retrieval_reference_mask=(gate_logits.sigmoid() > 0.5).to(
                gate_logits.dtype
            ),
        )
        if self.use_v18:
            # homology signal for dual-family gating: top retrieval cosine if available
            if homology is None and retrieved_sims is not None:
                homology = retrieved_sims.clamp(min=0.0).max(dim=1, keepdim=True).values
            head_kwargs.update(sequence_tokens=sequence_tokens,
                               recog_prior=recog_prior, family_id=family_id,
                               homology=homology, family_vec=family_vec)
        pwm_logits = self.pwm_head(moe_out, **head_kwargs)        # internal frame
        register_logits = (
            self.register_head(moe_out) if self.use_register_head else None
        )

        aux = dict(self.moe.aux_dict) if isinstance(self.moe.aux_dict, dict) else {}
        if self.use_v18 and self.pwm_head._last_attn is not None:
            aux['attn']         = self.pwm_head._last_attn        # (B, Lq, Lk)
            aux['attn_key_mask'] = self.pwm_head._last_key_mask   # (B, Lk)
            aux['recog_prior']   = recog_prior                    # (B, Lk) or None
        if trust_logits is not None:
            aux['trust_logits']    = trust_logits
            aux['retrieved_pwms']  = ret_pwms
            aux['retrieved_masks'] = ret_masks
            # Diagnostics (Feature 6): beta gate value per sample
            if hasattr(self.pwm_head, "_last_beta_gated"):
                aux['beta_gated'] = self.pwm_head._last_beta_gated
        if register_logits is not None:
            aux["register_logits"] = register_logits
            aux["internal_gate_logits"] = gate_logits
            aux["internal_pwm_logits"] = pwm_logits
            if not self.training:
                gate_logits, pwm_logits = export_registered_predictions(
                    gate_logits,
                    pwm_logits,
                    register_logits,
                    self.config.registration_max_shift,
                )
        return gate_logits, pwm_logits, aux

    def infer_mask(self, gate_logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert gate logits to a hard binary mask at inference time."""
        return (gate_logits.sigmoid() > threshold).float()
