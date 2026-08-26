"""v26 config. Deliberately small: no family, source, retrieval or provenance fields exist."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field


@dataclass
class V26Config:
    # encoder
    esm_model: str = "esm2_t33_650M_UR50D"
    esm_embed_dim: int = 1280
    esm_layers_to_average: int = 4
    freeze_encoder: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_n_layers: int = 6
    d_model: int = 512
    dropout: float = 0.1
    refiner_layers: int = 2
    refiner_heads: int = 8

    # mixture of experts (sequence-conditioned; set use_moe=False for the dense control)
    use_moe: bool = True
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    top_k: int = 2
    expert_hidden: int = 512
    balance_loss_weight: float = 0.01

    # context
    use_flank: bool = False
    flank_dropout: float = 0.4
    flank_gate_bias_init: float = -3.0
    use_partners: bool = False
    partner_dropout: float = 0.3
    partner_gate_bias_init: float = -3.0
    max_partners: int = 3

    # heads
    min_motif_length: int = 4
    max_motif_length: int = 42
    pwm_attn_heads: int = 4
    lambda_contact_init: float = 0.1
    use_contact_head: bool = True

    # loss weights (2-D contact deliberately 0; see docs/v26_contact_2d_decision.md)
    w_pwm: float = 1.0
    w_length: float = 0.05
    w_contact1d: float = 0.1
    w_contact2d: float = 0.0
    w_recognition_prior: float = 0.02
    w_context_consistency: float = 0.05

    # v24-parity loss terms. v24 trained with --latent-registration, --topbase-weight 0.1 and
    # --pwm-cov-r-weight 0.25; v26 had none of them, and v24 beats v26 by 0.23 cov_r on the SAME
    # clean split, so these are the prime suspects for the gap.
    w_registration: float = 0.0        # >0 enables registration-aware PWM loss
    registration_max_shift: int = 6
    registration_rc: bool = True
    w_topbase: float = 0.0
    topbase_margin: float = 2.0
    w_covr: float = 0.0
    # v24's three sharpness terms (combined weight 1.1). v26 omitted all three and its ic_mae is
    # 0.81-1.03 vs v24's 0.49-0.62 -- predicted PWMs are far too flat.
    w_ic: float = 0.0          # v24 pwm_ic_weight 0.5
    w_ic_pcc: float = 0.0      # v24 pwm_ic_pcc_weight 0.5  (per-COLUMN Pearson, IC-weighted)
    w_entropy: float = 0.0     # v24 pwm_entropy_weight 0.1

    def to_dict(self):
        return asdict(self)
