# TFScope — Architecture & Results (as of 2026-06-11)

Seed model for predicting transcription-factor (TF) binding specificity (the **initial PWM**)
directly from the protein sequence of the DNA-binding domain (DBD). This document is the
single-file reference for the current model design, training setup, and benchmark results.

---

## 1. Problem & framing

- **Input:** the DBD + flanking linkers of a TF (median ~111 aa), tokenized as ESM-2 amino-acid IDs,
  plus a `dbd_mask` (which residues are in the DBD) and an integer `family_id`.
- **Output:** a position weight matrix (PWM), `(4, L)` over `{A,C,G,T}`, left-anchored, with a
  per-position **gate** that decides how many columns are informative (motif length is *predicted*,
  not fixed).
- **Niche vs. DeepPBS:** DeepPBS needs a 3D protein–DNA structure. TFScope is **sequence-only** and
  predicts the PWM from the DBD sequence alone. The core idea is **retrieval-augmented prediction
  (RAG):** condition on the PWMs of the nearest training TFs (like AlphaFold templates / rCLAMPS),
  so the model refines a strong prior rather than predicting from scratch.

### Data
- Parquet: `data/processed/tf_pwm_aug_dbd_canon_trim.parquet` — 4,247 PWM rows (sequence-augmented),
  PWMs canonicalized + left-anchored + IC-trimmed; motif length median 9 (range 1–20, capped at
  `max_motif_length=20`). Stored `sequence` is the DBD+linkers (`dbd_mask` all-True in this parquet).
- The stored PWM is `float32` bytes reshaped to `(4, motif_length)`, ACGT order.

---

## 2. Architecture

End-to-end forward pass (`src/tfscope/models/tfscope.py::TFScopeModel`):

```
sequence_tokens, dbd_mask, family_id, [retrieved_pwms, retrieved_sims]
        │
        ▼
 ┌──────────────────────┐
 │ ESM-2 650M backbone  │  frozen, last 6 layers get LoRA (q_proj,v_proj), rank 16, α 32
 │ (esm2_t33_650M_UR50D)│  learned softmax over last 4 layers → (B, L, 1280)
 └──────────┬───────────┘
            │  + dbd_indicator (learned vector added to DBD residues)
     ┌──────┴───────┐
     ▼              ▼
 GatedAttn       GatedAttn        dual-stream pooling (NeurIPS'24 gated attention,
 (global)        (DBD-masked)     sigmoid value-gate → attention-sink-free)
     └──────┬───────┘
            ▼
      ProjectionHead              concat(global,dbd) 2560 → 512 (GELU+LN+dropout)
            │  combined (B,512)
            ▼
 ┌──────────────────────────────────────────────┐
 │ Family-conditioned MoE (DeepSeek-style)        │
 │  • SwiGLU experts (routed, top-k) + shared     │
 │  • FamilyAwareGating: gate logits + cosine     │
 │    (family_emb · expert_prototype) routing bias│
 │  • FiLM(family_emb) modulates each expert      │
 │  • PrototypeDictionary (32 concepts, interp.)  │
 └──────────┬─────────────────────────────────────┘
            │ moe_out (B,512)
       ┌────┴───────────────┐
       ▼                    ▼
 PositionGateHead      PWMHeadV18  ──────────────► pwm_logits (B,4,20)
 → gate_logits                ▲
   (B,20)                     │ retrieval: TrustPredictor (B,K)
                              └── retrieved_pwms (B,K,4,20), sims (B,K)
```

### 2.1 Encoder — `backbone.py`
- **ESM-2 650M** (`esm2_t33_650M_UR50D`), **frozen**. **LoRA** adapters (`LoRALinear`) injected into
  `q_proj`/`v_proj` of the **last 6 transformer layers** (rank 16, α 32). Gradient is split at the
  LoRA boundary: layers below run under `no_grad`; the last 6 use activation checkpointing so only
  ~6 layers' activations are stored (this is the main memory/throughput driver).
- Output = learned **softmax-weighted average of the last 4 layers** → `(B, L, 1280)`.

### 2.2 Pooling — `pooling.py`
- **Dual-stream `GatedAttentionPooling`**: one global stream over all residues, one DBD-masked
  stream. Multi-head attention with a learned query **plus a per-position sigmoid gate on the value
  vectors** (NeurIPS-2024 gated attention) to suppress uninformative residues / prevent attention
  sinks. A learned `dbd_indicator` vector is added to DBD residues before pooling.
- `ProjectionHead`: concat(global, DBD) `2×1280=2560 → 512` (GELU + LayerNorm + dropout 0.1).

### 2.3 Mixture-of-Experts — `moe.py`
- **SwiGLU experts** (DeepSeek-V2/V3 style: `W_down(silu(W_gate·x) ⊙ W_up·x)`), `num_experts` routed
  + `n_shared_experts=1` always-on.
- **`FamilyAwareGating`**: standard gate logits **plus a semantic routing bias** =
  cosine(family_embedding, learned per-expert prototype). This routes *unseen* families to experts
  that already handle semantically-similar families (generalizes continuously, unlike a discrete
  family→expert table).
- **FiLM** conditioning: `family_emb` produces per-expert `(γ, β)` that modulate expert outputs.
- **Family embedding** is either `SemanticFamilyEmbedding` (frozen ProTrek-text ++ ESM-2 vectors,
  small trainable projection — works for unseen families) or a learned fallback. **Key point:** the
  MoE does **not classify** the family — it is *told* `family_id` and uses it for **conditioned
  routing**. Quality is therefore bounded by the upstream Pfam labels.
- **`PrototypeDictionary`** (32 prototypes): interpretable "binding concepts," soft-attention readout
  for analysis. Output is `x + shared + routed + proto` (residual).

### 2.4 Output heads — `heads.py`, `pwm_head_v18.py`
- **`PositionGateHead`**: `512 → 256 → 128 → 20` logits; per-position sigmoid gate supervised by the
  binary `pwm_mask` (BCE) + an ordinal term encouraging contiguous left-aligned motifs. Replaces
  discrete length classification.
- **`PWMHeadV18`** (current head) = **prior branch + contact-aware residual**:
  `logits = z_prior + λ · Δz_contact`, with `λ = exp(log_lambda)` initialized at 0.1 (prior dominates
  early).
  - **Prior branch** = the legacy `PWMRegressionHead` (see 2.5): per-position queries (self-attn +
    cross-attn to ESM-DBD residues) + the RAG log-prior path.
  - **Contact-aware residual** (`ContactCrossAttention`): **cosine** cross-attention (+ LayerNorm on
    K/V) from each PWM column to DBD residues, with **amino-acid-identity values** (an `nn.Embedding`
    over the residue token added to V) so that a **point mutation moves the value vector** and hence
    Δz. Designed to fix the documented degeneracy of the legacy cross-attention (rank-1 collapse,
    terminal-residue attention sink, mutation-blindness). Supports **hard-sparse normalizers**
    (`softmax` | `entmax15` | `sparsemax` | learnable-α `entmax_bisect`) — sparse modes give
    exactly-zero attention weights ("contacts"). **The current best checkpoints use `softmax`**;
    entmax hard-sparsity is the planned Step-2 novelty (separate run).
  - Δz is **zero-meaned per base** (identifiability). Exposes `_last_attn`/`_last_key_mask` for the
    attention regularizers.

### 2.5 Retrieval (RAG) — `retrieval.py` + the prior branch
- **Index (offline):** `build_tf_embeddings.py` → ESM-2 layer-33 **DBD-masked mean-pool** per TF →
  `npz`. `build_nn_index.py` → cosine-NN, **leave-one-out**, donor pool = `train ∪ val`, **top-K=3**.
  Index used here: `data/processed/tf_nn_index_cluster40.json`.
- **`TrustPredictor`** (learned, replaces a raw cosine gate): from `(query_features, retrieved_PWM_k,
  cos_sim_k)` it predicts **"will neighbour k's PWM transfer to this query?"** Supervised at training
  by the true per-column Pearson r between the neighbour PWM and the target (`compute_true_trust`).
  This matters because same-family DBDs look near-identical in embedding space (cos≈0.95) even when
  their PWMs differ — cosine alone over-trusts them.
- **Fusion (prior branch, `PWMRegressionHead`):** per PWM position, attention over the K neighbour
  columns (weighted by `trust_scores`), producing a combined **log-prior** = trust-weighted sum of
  the K log-PWMs. A learned **β-gate** = `β·σ(conf_scale·(max_trust − conf_thresh))` decides how much
  to lean on retrieval; if no neighbour is trustworthy, β→0 and the model falls back to the de-novo
  sequence pathway. Output = `delta_logits + β·combined_log`.
- **Classifier-free guidance:** during training, `retrieval_dropout=0.15` of samples have their
  retrieval zeroed so the de-novo pathway stays functional for TFs with no good neighbour.

### 2.6 Loss — `losses/tfscope_loss.py`
Weighted sum (config weights in §3):
- **Gate:** BCE on per-position gate + ordinal regularizer (`gate_ordinal_weight`).
- **PWM content:** L1/MAE (`pwm_l1_weight`), KL(target‖pred) (`pwm_*`), entropy (sharpening),
  **IC-matching** (`pwm_ic_weight`), **IC-weighted (1−Pearson r)** per column (`pwm_ic_pcc_weight`),
  **top-base hinge margin** at high-IC positions (`pwm_topbase_weight`/`margin`).
- **Trust:** auxiliary supervision of `TrustPredictor` (`trust_loss_weight`).
- **MoE:** load-balance + diversity (`balance_loss_weight`, `diversity_loss_weight`).
- **Optional:** DPAC-style in-batch **PWM-contrastive** (`pwm_contrastive_weight`, default 0 —
  anti family-collapse), and v18 attention regularizers (row-diversity, hub penalty) +
  optional contact-prior supervision.

---

## 3. Training setup (current best: `cluster40_v18a_rag`)

| | |
|---|---|
| Split | `cluster40` (CD-HIT 40% identity, honest OOD): **2,983 train / 625 val / 639 test** PWMs; **957 / 168 / 222** unique genes |
| Encoder | ESM-2 650M frozen; LoRA rank 16, α 32, last 6 layers; LoRA-lr 1e-5 |
| MoE (best) | 10 families, **12 experts**, top-k, expert hidden 2048, 1 shared, 32 prototypes |
| Head | `PWMHeadV18` (`--pwm-head-v18`), softmax attention, λ init 0.1 |
| Retrieval | k=3, dropout 0.15, `tf_nn_index_cluster40.json`, trust_loss 0.5 |
| Optim | AdamW, lr 6e-4, wd 0.01, warmup 500 steps, cosine; batch 128; max-grad-norm 1.0 |
| Loss wts | l1 1.0, ic 0.5, entropy 0.1, ic_pcc 0.5, topbase 0.1 (margin 2.0), gate 1.0, balance 0.05, diversity 0.01 |
| Schedule | up to 200 epochs, early-stop patience 30 on **val oracle-r** (eval every 5 epochs on 100 TFs) |

**Dataset is tiny:** ~24 gradient steps/epoch (batch 128 over ~2,983 augmented PWMs / 957 genes),
so the whole 200-epoch schedule is only ~4,800 steps.

### Evaluation protocol — `scripts/eval_oracle_r_testset.py`
All numbers below use the **same protocol** on the cluster40 test set, with the **≥4-position rule**
(motifs whose IC≥0.25 core spans <4 positions are excluded; 639→**636**):
- **gate-r** (deployable boundary): pred = columns where gate>0.5; ±10 offset + RC aligned to the
  IC≥0.25 target core; per-column Pearson r. *Primary metric.*
- **panel-r** (length-oracle): pred = `pwm_mask` window, same alignment.
- **canon_fixed_r** (fully deployable): **no alignment freedom** — measures absolute registration.
- Plus MAE (mean over elements), RMSE, CE, KL, top-1 base acc, AUC, F1, MCC.

> Note on MAE conventions: our panel MAE is mean-over-elements and oracle-aligned. The **DeepPBS**
> convention is `mean(sum_over_4_bases(|pred−true|))` in a fixed frame (≈4× larger). DeepPBS reports
> MAE 0.553 / r 0.70 on its own structure-based split (not threshold-matched to cluster40).

---

## 4. Results — cluster40 OOD test (n=636, one protocol)

> **Historical benchmark warning:** the table in this section predates the V19
> clean grouped split and corrected LoRA checkpoint saving. Do not use it as
> evidence for a V19 gain.

| model | gate-r ↑ | gate-med | panel-r ↑ | MAE ↓ | top1 ↑ | AUC ↑ | MCC ↑ | canon-r ↑ |
|---|---|---|---|---|---|---|---|---|
| **TFScope-RAG** (10-fam/12-exp) | **0.592** | **0.563** | **0.545** | **0.191** | **0.631** | 0.781 | 0.441 | 0.136 |
| baseline (no retrieval) | 0.535 | 0.506 | 0.513 | 0.203 | 0.610 | 0.776 | 0.434 | 0.122 |
| rebin34 / 16-expert (RAG) | 0.574 | 0.541 | 0.529 | 0.196 | 0.615 | **0.789** | **0.454** | 0.111 |
| Stage-B pretrain (DPAC+HT-SELEX) | 0.531 | 0.504 | 0.495 | 0.211 | 0.597 | 0.759 | 0.419 | 0.155 |
| Stage-B pretrain (DPAC) | 0.508 | 0.495 | 0.479 | 0.210 | 0.592 | 0.734 | 0.396 | **0.176** |
| contrastive-aux | 0.471 | 0.443 | 0.460 | 0.219 | 0.571 | 0.717 | 0.331 | 0.140 |

**Headline:** TFScope-**RAG is the best model** — wins all correlation metrics + MAE/top1.
Retrieval adds **+0.057 gate-r** over the identical no-retrieval baseline (0.592 vs 0.535).

### 4.1 V19 clean-split seed-42 publication candidate

V19 uses:

- `data/processed/splits/cluster40_clean/split.json`;
- `data/processed/tf_nn_index_cluster40_clean.json`;
- train-only validation/test retrieval;
- corrected checkpoints that retain trained LoRA tensors.

The validation-locked candidate composes the corrected E2 fixed-frame model
with E5b motif content using family-specific weights. On the held-out test set
over 195 evaluable genes:

| metric | corrected E2 | composition | paired delta | paired-gene 95% CI |
|---|---:|---:|---:|---:|
| panel-r | 0.4938 | **0.5454** | **+0.0516** | **[+0.0312, +0.0740]** |
| canon-r | 0.1573 | 0.1527 | -0.0046 | [-0.0228, +0.0125] |
| aligned DeepPBS-scale MAE | 0.8392 | 0.8361 | -0.0031 | [-0.0208, +0.0135] |
| fixed-frame MAE | 1.1444 | **1.1168** | **-0.0276** | **[-0.0446, -0.0111]** |
| RMSE | 0.3186 | **0.3070** | **-0.0115** | **[-0.0180, -0.0059]** |
| CE | 1.5619 | **1.4309** | **-0.1310** | **[-0.1738, -0.0916]** |
| KL | 1.1015 | **0.9729** | **-0.1285** | **[-0.1703, -0.0894]** |

The panel-r gain is primarily `C2H2_long` (`+0.2045`, 95% CI
`[+0.1384,+0.2730]`, paired permutation `p<1e-4`), with a smaller significant
`bHLH` gain (`p=0.0117`). Overall panel-r, fixed MAE, RMSE, CE, and KL pass
paired sign-flip permutation tests. Canon-r, aligned MAE, AUC, top1, F1, and
MCC do not show statistically supported improvements.

This is a two-model composition, not a single-model replacement. Report its
inference cost and the corrected E2 baseline. The user-fixed scope uses only
seed 42, which must be disclosed as a limitation.

DeepPBS's published MAE `0.553` and r `0.70` use its own structure-based split
and are not directly comparable to this sequence-only clean split.

### Key findings (ablations)
1. **Retrieval is the lever, not capacity.** RAG > baseline by +0.057. Every attempt to improve the
   *model* instead of the *donor pool* barely moves or hurts the headline metric.
2. **Finer taxonomy / more experts hurts.** The 34-family full-Pfam rebin + 16-expert MoE (matched
   capacity) tests **below** the 10-family/12-expert RAG at every checkpoint (peak 0.578 at epoch 25
   vs 0.592). It slightly sharpens the gate (best AUC/MCC) but degrades per-column PWM content.
3. **Pretraining doesn't transfer.** DPAC and DPAC+HT-SELEX Stage-B pretraining land at/below
   baseline on oracle metrics (they *do* help absolute registration → higher canon-r, but not
   specificity). Contrastive-aux is worst.
4. **DPAC as retrieval donors is harmful** (separate pure-retrieval diagnostic, no training):
   top-1 retrieved oracle-r 0.618→0.545. Root cause: DPAC proteins are bona-fide DBDs (96% cos≥0.9
   to a real TF) but their "motif" is a **single crystallized DNA fragment**, a garbage pseudo-PWM
   that displaces the correct real-TF donor. *A single bound sequence is the wrong supervision target.*
5. **Registration is the dominant error for everyone.** `canon_fixed_r` (no alignment) collapses to
   0.11–0.18 for all models (and is the same regime DeepPBS sits in) — the ±10/RC alignment recovers
   ~0.40 of correlation. Fixing register (structure/contacts) is the biggest open lever.

### Training dynamics — why RAG peaks early (~epoch 9–25)
- On the rebin34 run, `ckpt_best.pt` = **epoch 9** (val oracle-r 0.791); the honest **test** peak is
  **epoch 25** (gate-r 0.578); everything from epoch 50 on degrades to ~0.55.
- Mechanism: retrieval hands the model a near-complete answer, so the learnable task collapses to
  "trust the right neighbour + tiny refinement" — saturated in a few hundred steps. `L_pwm` barely
  moves (1.86→1.79) while `L_gate` wiggles. After saturation, training memorizes train-specific
  residuals that don't transfer to held-out families (cluster40 is OOD), so val rises while train
  falls. The best val is reached **during** warmup, before peak LR.
- **Val→test gap is large and expected** for RAG (e.g. val 0.79 → test 0.57). Only test numbers count.

---

## 5. Threshold context (is cluster40 "too hard"?)
Standard sequence-identity split thresholds: 25% (remote homology), **30% (canonical hard
generalization)**, 40% (moderate), 50%/90% UniRef (light dedup). cluster40 is **moderate — one notch
more lenient than the 30% gold standard** — so it is a fair, defensible OOD benchmark, not extreme.
Caveat: for short, highly-conserved DBDs, 40% identity is genuinely diverged, and the task has an
intrinsic ceiling (related DBDs can bind differently; registration caps everyone). Recommended for
the paper: report a **difficulty curve** (cluster30 / 40 / 70-90) — RAG's gain should grow as the
split loosens and shrink toward baseline at 30%.

---

## 6. Status & open levers
- **Best config to keep:** `cluster40_v18a_rag` — 10 families, 12 experts, `PWMHeadV18` (softmax),
  RAG k=3. (`results/cluster40_panel/min4_rag.json`.)
- **Dead ends (recorded):** rebin34/MoE-16, DPAC/HT-SELEX Stage-B pretraining, DPAC retrieval donors,
  contrastive-aux.
- **Untested sibling:** Step-2 **entmax hard-sparse** contact attention distilled from real PDB
  co-crystals (`train_cluster40_v18a_rebin34_e16_sparse_distill.sbatch`) — the actual planned novelty.
- **Highest-value next experiments:** (a) attack **registration** (the dominant error) via structural
  contacts / AF3; (b) push the retrieval *ceiling* (donor quality), e.g. `retrieval_dropout` sweep to
  force more sequence reliance; (c) the cluster30 difficulty-curve point.

---

### File map
| Component | File |
|---|---|
| Model assembly | `src/tfscope/models/tfscope.py` |
| ESM-2 + LoRA encoder | `src/tfscope/models/backbone.py` |
| Gated dual-stream pooling | `src/tfscope/models/pooling.py` |
| Family-conditioned MoE | `src/tfscope/models/moe.py` |
| Gate + legacy PWM/RAG head | `src/tfscope/models/heads.py` |
| Contact-aware v18 head | `src/tfscope/models/pwm_head_v18.py` |
| TrustPredictor (RAG) | `src/tfscope/models/retrieval.py` |
| Losses | `src/tfscope/losses/tfscope_loss.py` |
| PWM alignment (±10+RC) | `src/tfscope/models/alignment.py` |
| Training | `scripts/train.py`, `scripts/train_cluster40_v18a_rebin34_e16.sbatch` |
| Eval (this protocol) | `scripts/eval_oracle_r_testset.py`, `scripts/eval_full_metrics.py` |
| Embeddings / NN index | `scripts/build_tf_embeddings.py`, `scripts/build_nn_index.py` |
| Result JSONs | `results/cluster40_panel/*.json` (+ `SUMMARY.md`) |
