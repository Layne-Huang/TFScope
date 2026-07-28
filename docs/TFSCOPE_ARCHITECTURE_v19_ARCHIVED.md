# TFScope — Current Architecture

*Sequence-only transcription-factor DNA-binding-specificity (PWM) predictor.*
Reference: last updated 2026-07-06. All line references point to `src/tfscope/`.

TFScope maps a protein's amino-acid sequence (DNA-binding domain, DBD-cropped) to a
position weight matrix (PWM) over `{A, C, G, T}`, plus a soft per-position gate that
defines the motif's extent. It is built on a **frozen ESM-2 encoder** with light LoRA
adapters, gated attention pooling, an optional **mixture-of-experts** block, and a
**contact-aware cross-attention PWM head** supervised with recognition-residue contacts.

```
tokens (B,L) ──► ESM-2 650M (frozen + LoRA tail)                       backbone.py
                     │  weighted avg of last 4 layers
                     ▼
             residue embeddings (B, L, 1280)
                     │  + DBD indicator on DBD residues
                     ▼
        ┌─────────── MoE (per-protein OR per-residue) ─────────┐        moe.py
        │  protein-mode: applied AFTER pooling (pooled vector)  │
        │  residue-mode: applied HERE, per DBD token            │
        └───────────────────────┬──────────────────────────────┘
                     ▼           │(refined residue reps feed pooling + head keys)
       ┌─────────────┴─────────────┐
       │ global gated-attn pool     │  dbd gated-attn pool                pooling.py
       └─────────────┬─────────────┘
                     ▼
        ProjectionHead  → combined (B, 512)                              pooling.py
                     │
         (protein-mode MoE here) → moe_out (B, 512)
                     ├───────────────► PositionGateHead → gate (B, 20)   heads.py
                     ▼
        PWMHeadV18(moe_out, keys=residue embeddings) → pwm (B, 4, 20)    pwm_head_v18.py
```

---

## 1. Encoder — `models/backbone.py`

- **Model:** `esm2_t33_650M_UR50D` (1280-d, 33 layers), **frozen** (`freeze_encoder=True`).
- **LoRA adapters** (`LoRALinear`, backbone.py:12): low-rank deltas on `q_proj`/`v_proj`
  of the **last `lora_n_layers` (=6)** transformer layers. `y = Wx + (α/r)·BA·x`.
  Recipe uses `rank=16, alpha=32`. This is the only part of ESM that trains (~0.49 M params).
- **Gradient-split forward** (`_forward_esm_split`, backbone.py:82): layers `0..(n-6)` run
  under `no_grad` and are detached; the last 6 run with gradients + activation
  checkpointing to bound memory.
- **Output:** softmax-weighted average of the **last 4 layer** representations
  (`esm_layers_to_average=4`, learned `layer_weights`), `<cls>` stripped → `(B, L, 1280)`.

> Input is always **DBD-cropped** (models trained with `dbd_start=0` on all rows).
> Full-length input is out-of-distribution and degenerates. See memory
> `tfscope-dbd-cropped-input`.

## 2. DBD indicator + pooling — `tfscope.py:86`, `models/pooling.py`

- A learned `dbd_indicator` vector is added to residues inside the DBD (signals the
  pooler which positions are the domain).
- **Two `GatedAttentionPooling` heads** (pooling.py:49), 8-head attention with a
  learned query and a **per-position sigmoid gate on values** (NeurIPS-2024 gated
  attention; suppresses attention-sink residues):
  - `global_pool` over all residues,
  - `dbd_pool` masked to DBD residues only.
- **`ProjectionHead`** (pooling.py:89): concat(global, dbd) `(B, 2560)` → GELU → LayerNorm
  → Dropout → `combined (B, 512)` (`proj_hidden_dim=512`).

## 3. Mixture-of-Experts — `models/moe.py`

Selected by `config.moe_granularity ∈ {"protein", "residue"}` (`tfscope.py:35`).

### 3a. Protein-mode (default, legacy) — `MOEBlock` (moe.py:221)
Applied **after pooling** to the single pooled vector `combined (B, 512)` — i.e. **one
routing decision per protein**.
- `num_experts=12` SwiGLU routed experts + `n_shared_experts=1` always-on shared expert.
- `FamilyAwareGating` (moe.py:191): `logits = gate(x, family_emb) + cos(family_emb, expert_prototypes)`; top-`k=2`.
- FiLM family conditioning + a `PrototypeDictionary` (interpretability).
- Residual: `out = x + shared + routed + proto`.
- **Known behaviour:** this block *collapses* to uniform routing (it is not a bottleneck;
  the head reads ESM directly). It contributes a small **capacity/ensemble** gain, not
  specialization. See memory `moe-collapse-fig2e`, `family-conditioning-vestigial`.

### 3b. Residue-mode (new, DeepSeekMoE-style) — `ResidueMoE` (moe.py:309)
Applied **before pooling**, as a per-DBD-**token** FFN — i.e. **~50–70 routing decisions
per protein** so specialization can *emerge* rather than being supervised.
- **2 shared** SwiGLU experts (always on; absorb universal base-readout chemistry) +
  **8 fine-grained** routed SwiGLU experts, top-2 per token (`expert_hidden_dim=512`).
- Per-token router: `Linear([token_feat ‖ family_emb]) + cos(family_emb, expert_prototypes)`;
  **no CE routing supervision** (emergent). Family embedding is a soft bias only.
- FFN residual `out = x + shared + routed`; only DBD tokens are updated, others pass through.
- The **refined residue reps replace `esm_embeddings`** (tfscope.py) so they feed **both**
  the pooling **and** the PWM-head cross-attention keys → the MoE is a genuine bottleneck.
- Aux (flattened over DBD tokens) flows to the **token-level** load-balance loss; the
  entropy-maximizing `family_diversity_loss` is turned **off** in this mode.
- **Result (held-out val-as-test, ep~90):** gate oracle-r **0.704** / panel-r **0.681** /
  top-1 0.742 — beats the no-MoE 1-expert baseline (0.694/0.672) and ties the collapsed
  combined model (0.714/0.694) within noise; first MoE variant that neither collapses nor
  loses accuracy while learning real per-token routing. See memory `moe-collapse-fig2e`
  (Experiment 4).

## 4. Position-gate head — `models/heads.py:9` (`PositionGateHead`)
MLP `512 → 256 → 128 → max_motif_length(=20)` producing per-position **gate logits**
(pre-sigmoid). Supervised by BCE against the binary motif mask + an ordinal
regularizer (contiguous, left-aligned motif prior). Defines where the motif is; no
hard length bin.

## 5. PWM head (v18, contact-aware) — `models/pwm_head_v18.py`
`logits = z_prior + λ · Δz_contact` (pwm_head_v18.py:282). Two branches:

**Prior branch** — `PWMRegressionHead` (heads.py:39): per-position query
(`combined` broadcast + learned positional embedding) → self-attention across the 20
PWM columns → optional cross-attention to ESM DBD residues → `nucleotide_head → (B,4,20)`.
Gets the *average* motif right. (Optional RAG/retrieval log-prior path exists but the
production combined model runs **retrieval off**.)

**Contact-aware residual branch** — `ContactCrossAttention` (pwm_head_v18.py:48):
- **Single-head cosine** cross-attention (temp 10) from each PWM column (query) to DBD
  residues (keys), with **LayerNorm on K/V** — removes the high-norm hub-residue sink
  that made the legacy head rank-1-collapsed and mutation-blind.
- **Amino-acid-identity values** (`aa_embed` added to V): a point mutation moves the
  value vector, so `Δz` responds to mutations *if* the column attends to that residue
  (enables the MyoD1/KLF4 specificity-switch predictions).
- `Δz` is **zero-meaned per base** (identifiability); gated by a small learned `λ`
  (`exp(log_lambda)`, init 0.1) so the prior dominates early in training.
- Exposes `_last_attn (B,Lq,Lk)` + `_last_key_mask` so the loss can regularize/supervise
  the attention. Optional `recog_prior` additive **contact bias** on the attention logits.

> A `DualFamilyConditioner` fusion (learned-id + semantic, homology-gated) exists on this
> branch but is **off by default** in the combined model — it regressed vs combined.
> See memory `dual-family-vs-combined`, `family-conditioning-vestigial`.

## 6. Loss — `losses/tfscope_loss.py`
Composite (weights from the combined recipe):
- **Gate BCE** + ordinal contiguity term (`gate_loss_weight`, `gate_ordinal_weight`).
- **PWM**: L1 + information-content (IC) + IC-Pearson (`ic-pcc-weight 0.5`) + top-base
  margin (`topbase-weight 0.1`) + entropy regularizer.
- **v18 attention regularizers**: row-diversity (anti rank-1 collapse) + hub penalty.
- **Contact supervision** (`v18-contact-weight 0.3`): pushes cross-attention onto true
  recognition residues (`recognition_residues_cluster40trainonly.json`, **train-only** to
  avoid leakage). This is the key ingredient behind the DeepPBS-competitive result.
- **MoE aux** (moe.py path): Switch **load-balance** (`balance_loss_weight`) +
  `family_diversity_loss` (entropy-max; **0 in residue-mode**) + optional CE
  route-supervision (0 by default).

## 7. Training recipe (combined / production) — `scripts/run_v19_combined_fm_deeppbs_contact.sh`
- Data `tf_pwm_combined_fm_deeppbs.parquet`, split `combined_fm_deeppbs/split.json`
  (DeepPBS structure-level fold for a fair TFScope-vs-DeepPBS comparison).
- `lr 4.5e-4`, `lora-lr 7.5e-6`, `lora rank16/alpha32/6 layers`, warmup 150, bf16 + TF32.
- Gene-balanced sampling; `--eval-oracle-r` (oracle-aligned per-column Pearson r selects
  `ckpt_best`); early-stop patience 30.
- **Residue-MoE variant:** `scripts/run_v19_residue_moe.sh` — same recipe + `--moe-granularity
  residue --num-experts 8 --n-shared-experts 2 --top-k 2 --expert-hidden-dim 512
  --balance-loss-weight 0.01 --diversity-loss-weight 0.0`. Single-GPU pinned by UUID.

## 8. Key config values — `config.py`
| field | value | note |
|---|---|---|
| `esm_model` | esm2_t33_650M_UR50D | frozen |
| `esm_embed_dim` | 1280 | |
| `lora_rank / alpha / n_layers` | 16 / 32 / 6 | recipe overrides |
| `proj_hidden_dim` | 512 | MoE/head input |
| `pool_n_heads` | 8 | gated attention |
| `num_experts / n_shared / top_k` | 12 / 1 / 2 | protein-mode default |
| `max_motif_length` | 20 | all PWMs padded to 20 |
| `pwm_hidden_dim / pos_embed / attn_heads` | 128 / 64 / 4 | prior branch |
| `v18_attn_heads / cosine_temp / aa_value` | 1 / 10 / True | contact branch |
| `v18_delta_scale_init` | 0.1 | residual gate λ init |
| `num_families` | 10 | learned family embedding |

## 9. What is *not* the lever (honest notes)
- Accuracy is **ESM-sequence-driven**; the family head is near-inert and family
  conditioning barely changes the predicted PWM (`family-conditioning-vestigial`).
- The protein-mode MoE **collapses**; its edge is capacity, not specialization.
- The **residue-mode MoE** is the first that learns real routing at parity accuracy —
  its scientific value is emergent expert↔recognition-chemistry structure (analysis
  pending), not an accuracy boost.
- The decisive accuracy ingredient is the **contact-supervised v18 cross-attention head**,
  which makes TFScope competitive with structure-based DeepPBS from sequence alone.
