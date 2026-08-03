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
    use_cached_esmc: bool = False
    esmc_cache_dir: str = "/data1/leihuang/TFScope_store/esmc_emb"

    # Attention pooling
    pool_type: str = "gated_attention"  # "gated_attention" (default) or "mean".
                                        # "mean" = masked mean pool (ICLR baseline B2,
                                        # "frozen ESM + mean pool + MLP").
    pool_n_heads: int = 8
    pool_d_query: int = 64

    # Projection
    proj_hidden_dim: int = 512
    proj_dropout: float = 0.1

    # MOE
    use_moe: bool = True               # False -> bypass MoE entirely (identity).
                                       # ICLR necessity audit B5 ("v24 without MoE").
                                       # protein granularity: moe_out = combined;
                                       # residue granularity: dbd_emb passes through
                                       # unrouted. No MoE aux losses are emitted, so
                                       # the balance/diversity/route terms are skipped.
    num_experts: int = 12
    expert_hidden_dim: int = 2048      # expansion factor 4x
    top_k: int = 2                     # top-k routing
    capacity_factor: float = 1.25
    n_shared_experts: int = 1          # DeepSeek-style always-active shared experts
    moe_residual: bool = True          # add the input skip (x+...) around the MoE block.
                                       # False -> output = shared+routed+proto only (forces the
                                       # router to carry signal; tests expert specialization).
    route_supervision_weight: float = 0.0  # CE(gate_logits, family_id) routing supervision —
                                           # with a mode-relabeled parquet (family_id == mode),
                                           # pushes expert i to own recognition-mode i.
    n_prototypes: int = 32             # interpretable prototype dictionary (ProtoT, arXiv 2602.11852)
    moe_granularity: str = "protein"   # "protein" -> pooled MOEBlock (1 routing decision/protein);
                                       # "residue" -> per-DBD-token ResidueMoE (DeepSeekMoE-style,
                                       # ~50-70 decisions/protein so specialization can emerge).

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
    gate_mode: str = "independent"     # "independent" (legacy) or contiguous "span"
    span_gate_temperature: float = 0.5
    motif_overflow_policy: str = "warn"  # "error", "warn", or explicit "truncate"
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
    # ── integrated contact-predictor head → contact bias (sequence-only) ──
    contact_pred_head: bool = False        # add frozen ESM→contact linear probe; its per-residue
                                           # P(contact) feeds the v18 contact bias (works on unseen TFs)
    contact_probe_path: str = ""           # joblib LogisticRegression to warm-start the head
    v18_contact_bias_learnable: bool = False  # learn the bias scale (init = v18_contact_bias_scale)
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
    # ── dual-family fusion (ProtDAT-style deep, gated): learned-id + semantic family ──
    use_dual_family: bool = False      # fuse learned-id + semantic family in the v18 head
    dual_family_dim: int = 64          # dim of the fused family conditioning vector
    dual_family_semantic_path: str = ""  # semantic vectors for the dual head (separate from MoE's
                                         # family_embedding_path so the MoE can stay learned)

    # Retrieval augmentation (v8 RAG-TFScope)
    use_retrieval: bool = False        # enable cross-attention to retrieved PWMs
    residual_prior: bool = False       # v12: output = log(prior) + alpha*delta instead of de-novo+log-prior
    retrieval_k: int = 3               # top-K nearest neighbours per sample
    retrieval_dropout: float = 0.20    # v10: moderate CFG (0.40 in v9 was too aggressive)
    trust_loss_weight: float = 0.5     # v10: weight on trust-predictor auxiliary BCE loss
    aligned_trust_target: bool = False
    trust_rank_loss_weight: float = 0.0
    trust_rank_margin: float = 0.1
    positionwise_retrieval_gate: bool = False
    align_retrieved_pwms: bool = False
    retrieval_alignment_max_shift: int = 10
    retrieval_alignment_min_overlap: int = 4
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
    # Couples the gate to the EVAL protocol. At eval (train.py) the gate picks
    # which columns are scored (`active = gate > 0.5` -> pred_core -> align_pwm),
    # and align_pwm reports per-column r over the overlap only -- so a SHORTER
    # gate is scored on fewer, easier columns and gets a higher r. BCE against
    # the GT mask alone does not counter that. This penalises |soft_len - gt_len|
    # directly. Suggested range 0.05-0.1; 0.0 keeps the old behaviour.
    gate_length_weight: float = 0.0
    pwm_cov_r_weight: float = 0.0       # differentiable full-core r x soft-coverage objective
    pwm_core_ic_thresh: float = 0.25    # bits; shared train/eval informative-core threshold
    # Two-chain (heterodimer) input: when True and a row has a partner_sequence,
    # feed ESM `chain1 + <eos> separator + partner_DBD` and mark BOTH chains'
    # residues in dbd_mask, so the PWM head can attend to both protomers and
    # place a two-half-site motif (NR direct repeats, bZIP/MAF, POU-SOX). Only
    # heterodimer rows are affected; every other row is unchanged single-chain.
    two_chain_input: bool = False
    require_multichain_eligible: bool = False  # legacy checkpoints used every available partner
    chain_id_embedding: bool = False
    # Max protomers fed (self + up to max_chains-1 partners). 2 = dimer (legacy);
    # 4 = tetramer, needed for p53 and HSF/NF-Y/IRF multimers (order-aware v23).
    max_chains: int = 2
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
    # V19 E4: latent target registration over offset x reverse-complement states.
    latent_registration: bool = False
    registration_max_shift: int = 10
    registration_min_overlap: int = 4
    registration_temperature: float = 0.1
    registration_coverage_penalty: float = 0.5
    registration_anchor_path: str = ""
    register_head: bool = False
    register_loss_weight: float = 0.5

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
