from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TFScopeConfig:
    # Encoder
    esm_model: str = "esm2_t33_650M_UR50D"
    esm_embed_dim: int = 1280          # ESM-2 650M d_model
    esm_layers_to_average: int = 4     # weighted avg of last N layers
    freeze_encoder: bool = True
    lora_rank: int = 0          # 0 = disabled; >0 injects LoRA into last lora_n_layers
    lora_alpha: float = 16.0    # LoRA scaling: delta = (alpha/rank) * B @ A @ x
    lora_n_layers: int = 6      # number of ESM-2 tail layers to inject LoRA into

    # Attention pooling
    pool_n_heads: int = 8
    pool_d_query: int = 64

    # Projection
    proj_hidden_dim: int = 512
    proj_dropout: float = 0.1

    # MOE
    num_experts: int = 12
    expert_hidden_dim: int = 2048      # expansion factor 4x
    top_k: int = 2                     # top-k routing
    capacity_factor: float = 1.25
    n_shared_experts: int = 1          # DeepSeek-style always-active shared experts
    n_prototypes: int = 32             # interpretable prototype dictionary (ProtoT, arXiv 2602.11852)

    # Family conditioning
    num_families: int = 10             # 8 core + "other" + "multi-domain"
    family_embed_dim: int = 64
    # Path to pre-computed semantic family embeddings (ProTrek text + ESM-2 seq).
    # If set and file exists, SemanticFamilyEmbedding is used; otherwise falls
    # back to LearnedFamilyEmbedding.
    family_embedding_path: str = (
        "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/family_embeddings.pt"
    )

    # Output heads
    max_motif_length: int = 20         # all outputs are padded to this length
    min_motif_length: int = 4          # kept for data validation only
    pwm_attn_heads: int = 4
    pwm_pos_embed_dim: int = 64
    pwm_hidden_dim: int = 128
    pwm_cross_attn: bool = True        # cross-attend PWM positions to ESM2 DBD embeddings

    # ── v18: contact-aware, mutation-sensitive PWM head ───────────────────────
    # v18a (attention repair): replace the degenerate dot-product cross-attention
    # with a cosine, LayerNorm'd, amino-acid-identity-aware *residual* branch on
    # top of the existing v10/v14 prior head, plus anti-collapse regularisation.
    pwm_head_v18: bool = False         # enable the v18 contact-aware residual head
    v18_attn_heads: int = 1            # single-head cross-attention (interpretable)
    v18_cosine_temp: float = 10.0      # temperature for cosine attention logits
    v18_kv_layernorm: bool = True      # LayerNorm ESM embeddings before K/V projection
    # Hard-sparse residue->base attention (recognition-contact sparse attention).
    #   "softmax"    : dense (default, == current model)
    #   "entmax15"   : fixed alpha=1.5, exactly-sparse
    #   "sparsemax"  : alpha=2.0, sparsest
    #   "entmax_learn": learnable per-head alpha in (1,2] via 1+sigmoid (alpha->1 recovers softmax)
    v18_attn_sparse: str = "softmax"
    v18_attn_alpha_init: float = 1.5   # init alpha for entmax_learn (sigmoid^-1 applied internally)
    # Contact-distillation: KL(structural per-base target || attention row), applied only on the
    # PDB-structured subset (samples that carry a contact_target). 0 = off (no-op).
    contact_distill_weight: float = 0.0
    contact_targets_path: str = "data/contact_maps/contact_targets.json"
    v18_aa_value: bool = True          # add amino-acid-identity embedding to values (mutation signal)
    v18_delta_scale_init: float = 0.1  # initial residual gate λ (exp(log_lambda))
    v18_row_div_weight: float = 0.05   # row-diversity loss (penalise rank-1 collapse)
    v18_hub_weight: float = 0.05       # hub penalty (penalise residues over-attended across columns)
    v18_hub_frac: float = 0.34         # u_max = hub_frac * (#valid motif columns)
    v18_freeze_prior: bool = False     # train only the contact branch (freeze v14 prior)
    # v18b (contact supervision): pull attention onto family-canonical recognition residues.
    v18_contact_supervision: bool = False
    v18_contact_weight: float = 0.3    # weight on contact-prior cross-entropy
    v18_contact_bias_scale: float = 0.0  # additive soft bias of recog prior into attn logits (0 = off)
    v18_contact_code: bool = False     # use family/aa contact-code MLP for Δz values
    recognition_prior_path: str = "data/contact_maps/recognition_residues.json"

    # Retrieval augmentation (v8 RAG-TFScope)
    use_retrieval: bool = False        # enable cross-attention to retrieved PWMs
    residual_prior: bool = False       # v12: output = log(prior) + alpha*delta instead of de-novo+log-prior
    retrieval_k: int = 3               # top-K nearest neighbours per sample
    retrieval_dropout: float = 0.20    # v10: moderate CFG (0.40 in v9 was too aggressive)
    trust_loss_weight: float = 0.5     # v10: weight on trust-predictor auxiliary BCE loss
    retrieval_hidden_dim: int = 128    # d for retrieved-PWM token embeddings (matches pwm_hidden_dim)
    retrieval_index_path: str = (
        "data/processed/tf_nn_index.json"
    )
    # Robust-RAG training augmentations (v17) — TRAIN split only, never at eval.
    full_retrieval_dropout: float = 0.0   # prob to disable ALL retrieval for a sample
    neighbor_dropout: float = 0.0         # per-neighbour independent drop prob
    hard_negative_rate: float = 0.0       # prob to inject hard-negative neighbour(s)
    hard_negative_per_sample: int = 1     # how many neighbours to replace with hard negs
    all_bad_case_rate: float = 0.0        # prob to replace ALL neighbours with bad ones

    # Loss
    gate_loss_weight: float = 1.0      # weight on position gate BCE loss
    gate_ordinal_weight: float = 0.05  # penalty for non-monotone gates
    balance_loss_weight: float = 0.05
    diversity_loss_weight: float = 0.01
    pwm_l1_weight: float = 1.0         # primary PWM loss: plain L1 matching DeepPBS formula
    pwm_ic_weight: float = 0.5         # IC-matching term |IC(target) - IC(pred)|
    pwm_entropy_weight: float = 0.1    # entropy penalty to prevent flat predictions
    pwm_ic_pcc_weight: float = 0.0     # v14: IC-weighted per-column (1-Pearson) — targets base composition
    pwm_topbase_weight: float = 0.0    # v14: top-base margin loss on high-IC positions
    pwm_topbase_margin: float = 2.0    # logit margin for top-base loss
    pwm_topbase_ic_thresh: float = 0.5 # only apply top-base loss where target IC > this (bits)
    pwm_contrastive_weight: float = 0.0  # DPAC-style in-batch contrastive (anti family-collapse)
    pwm_contrastive_tau: float = 0.1     # temperature for InfoNCE over PWM column-cosine similarity

    # Training
    learning_rate: float = 3e-4
    lora_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    total_steps: int = 20000
    batch_size: int = 32
    max_grad_norm: float = 1.0
    seed: int = 42

    @property
    def length_range(self) -> Tuple[int, int]:
        return (self.min_motif_length, self.max_motif_length)

    @property
    def num_length_classes(self) -> int:
        """Kept for backward-compat; architecture no longer uses it."""
        return self.max_motif_length - self.min_motif_length + 1
