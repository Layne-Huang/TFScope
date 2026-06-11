# TFScope Seed Model — Architecture Overview

**Synthesis of Phase 1 Brainstorm Sessions**
**Date:** 2026-04-16

---

## 1. Executive Summary

TFScope predicts TF-DNA binding specificity directly from protein amino acid sequence, without requiring structural information. The seed model (Module 1) encodes a TF protein sequence using a pretrained protein language model, extracts both global and DNA-binding-domain-focused features, routes them through a Mixture-of-Experts layer conditioned on DBD family identity, and outputs a candidate motif length and coarse nucleotide preference profile (initial PWM).

**Key novelty vs. DeepPBS (Nature Methods 2024):** DeepPBS requires protein-DNA structures as input. TFScope predicts from sequence alone — scalable to the full repertoire of TFs across species, including those with no structural data.

---

## 2. Architecture Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT                                        │
│  Protein sequence  +  DBD boundary mask  +  DBD family label       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ESM C 600M ENCODER (frozen)                       │
│  Full protein sequence → per-residue embeddings (1152-dim)         │
│  Weighted average of last 4 layers (learnable weights)             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    residue embeddings [B, L, 1152]
                               │
               ┌───────────────┴───────────────┐
               │                               │
               ▼                               ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│  GLOBAL STREAM           │   │  DBD STREAM                          │
│  Attention pooling       │   │  Attention pooling                   │
│  (all residues, 8-head)  │   │  (DBD-masked only, 8-head, separate) │
│  → [B, 1152]             │   │  → [B, 1152]                         │
└───────────┬──────────────┘   └─────────────────┬────────────────────┘
            │                                     │
            └──────────────┬──────────────────────┘
                           │  Concatenate
                           ▼
              ┌────────────────────────┐
              │  MLP Projection        │
              │  2304 → 512            │
              │  Linear+GELU+LN+Drop   │
              └───────────┬────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MOE BLOCK (12 experts)                            │
│                                                                     │
│  Family Embedding (64-dim)                                          │
│       │                                                             │
│       ├──→ Family-Aware Gating: Linear(576→256→12) + family_bias   │
│       │         Top-2 sparse routing                                │
│       │                                                             │
│       └──→ FiLM conditioning inside each expert                     │
│              gamma * expert_output + beta                            │
│                                                                     │
│  Expert MLP: 512→2048→GELU→2048→512 (×12 experts, ~25.2M params)  │
│  Load balance loss + Diversity loss + Capacity factor 1.25          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                          [B, 512]
                               │
               ┌───────────────┴───────────────┐
               │                               │
               ▼                               ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│  MOTIF LENGTH HEAD       │   │  PWM REGRESSION HEAD                 │
│  512→256→128→17          │   │  512+pos_embed → 128-dim per pos     │
│  (classes for 4-20bp)    │   │  4-head self-attention across pos    │
│  CE + label_smooth=0.1   │   │  → Linear(128, 4) per position       │
│                          │   │  Variable-length mask [B, 4, 20]     │
│  Output: [B, 17] logits  │   │  KL divergence loss                  │
│                          │   │  Output: [B, 4, 20] logits           │
└──────────────────────────┘   └──────────────────────────────────────┘
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │  LOSS: Uncertainty-weighted     │
              │  σ_L² × L_length + log σ_L     │
              │  σ_P² × L_pwm + log σ_P        │
              │  + 0.01 × L_balance             │
              │  + 0.005 × L_diversity          │
              └────────────────────────────────┘
```

---

## 3. Design Decision Summary

| Component | Decision | Rationale | Key Alternative Considered |
|-----------|----------|-----------|---------------------------|
| **Encoder** | ESM C 600M (frozen) | Purpose-built for embeddings; no multi-track overhead | ESM3 1.4B (generative, wasteful) |
| **Layer extraction** | Weighted avg last 4 layers | Multi-level info without dimension explosion | Last layer only; concatenation |
| **Encoder training** | Frozen → probe → LoRA | Avoid catastrophic forgetting | Full fine-tuning (risky) |
| **Global pooling** | Multi-head attn, learned query, all residues | Can focus on important positions | Mean pooling (undifferentiated) |
| **DBD pooling** | Same arch, separate params, DBD mask | Focuses on recognition determinants | Shared params |
| **Stream combination** | Concat + MLP (2304→512) | Preserves both global and local | Addition (loses info) |
| **MOE type** | Sparse top-2, 12 experts | Family specialization with knowledge transfer | Dense routing (diluted); hard 1:1 mapping (dead experts) |
| **Conditioning** | FiLM + family-aware gating bias | Parameter-efficient; biologically motivated | Hypernetworks (expensive, unstable) |
| **Load balancing** | Auxiliary loss + diversity loss + stratified sampling | Multi-pronged for severe imbalance | Single balance loss only |
| **Length loss** | CE with label_smoothing=0.1 | Adjacent lengths near-equivalent | Regression (no ordinal structure) |
| **PWM loss** | KL divergence, position-masked | Natural metric for probability distributions | MSE on probabilities |
| **Task balancing** | Uncertainty weighting (Kendall 2018) | Automatic, learned | Hand-tuned lambda |
| **PWM head** | Self-attention across positions | Captures inter-position dependencies | Independent per-position MLP |
| **Motif length** | 17 classes [4-20bp] | Family-conditional in practice | Regression |
| **DBD families** | 8-10 core + "other", C2H2 subdivided | ~85% TF coverage | All ~60 families (sparse) |
| **Training data** | JASPAR 2024 + CIS-BP experimental | Quality + coverage | CIS-BP all (leakage risk) |
| **Evaluation** | LOFO cross-validation (primary) | Cross-family generalization | Random split only |

---

## 4. Input/Output Specification

### Inputs

| Input | Shape | Type | Source |
|-------|-------|------|--------|
| `protein_sequence` | `(B,)` | string (amino acids) | UniProt |
| `dbd_mask` | `(B, L)` | bool | InterPro/Pfam annotations |
| `family_id` | `(B,)` | int [0..11] | DBD family classification |

### Outputs

| Output | Shape | Description |
|--------|-------|-------------|
| `length_logits` | `(B, 17)` | Logits for motif lengths 4..20 |
| `pwm_logits` | `(B, 4, 20)` | Logits for A/C/G/T at each position |
| `predicted_length` | `(B,)` | argmax of length_logits |
| `predicted_pwm` | `(B, 4, L_pred)` | Softmax of pwm_logits, masked by predicted length |

---

## 5. Recommended Implementation Order

### Phase 2a: Data Pipeline (~2-3 days)
1. Download JASPAR 2024 + CIS-BP experimental PWMs
2. Map TFs to UniProt sequences, InterPro DBD annotations, DBD family labels
3. Build dataset class with family-stratified sampling
4. Create train/val/test splits (LOFO + identity-based + within-family)

### Phase 2b: Encoder + Pooling (~2-3 days)
1. Set up ESM C 600M inference (frozen)
2. Implement weighted layer averaging
3. Implement dual-stream attention pooling
4. Run diagnostic probes (t-SNE, linear probe)

### Phase 2c: MOE + Output Heads (~3-4 days)
1. Implement MOE block with gating, FiLM, load balancing
2. Implement motif length head and PWM regression head
3. Implement multi-task loss with uncertainty weighting
4. End-to-end training pipeline

### Phase 2d: Ablation + Evaluation (~3-5 days)
1. Run ablation experiments (routing, expert count, conditioning, features)
2. Full LOFO cross-validation
3. Comparison baselines (DeepPBS with predicted structures, homology transfer)
4. Generate figures

---

## 6. Key Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **ESM C embeddings lack DBD specificity** | Medium | High | Diagnostic probing first; LoRA fine-tuning if needed |
| **MOE experts don't specialize meaningfully** | Medium | Medium | Ablation with dense routing as fallback; monitor expert utilization |
| **Severe family imbalance causes poor rare-family performance** | High | Medium | Stratified sampling + diversity loss + oversampling |
| **Nature Methods sees seed model as incomplete** | Medium | High | Frame as standalone tool + component of broader framework; compare against DeepPBS with predicted (not experimental) structures |
| **PWM independence assumption limits accuracy** | Low | Low | Acceptable for seed model; self-attention in PWM head partially addresses |
| **Dimerization not modeled** | Known | Medium | Restrict evaluation to monomers/homodimers; acknowledge limitation |

---

## 7. Phase 2 Readiness Checklist

- [ ] ESM C installed and inference tested (`pip install esm`)
- [ ] JASPAR 2024 data downloaded and parsed
- [ ] CIS-BP experimental PWMs extracted (excluding inferred)
- [ ] UniProt-to-DBD-family mapping built
- [ ] InterPro DBD boundary annotations extracted
- [ ] `tfscope` conda/mamba environment set up with PyTorch, esm
- [ ] GPU access confirmed (single A100 or equivalent)
- [ ] DeepPBS baseline available for comparison
- [ ] Evaluation metrics implemented (Tomtom, Pearson, KL)
- [ ] LOFO split strategy implemented

---

## 8. Total Parameter Budget

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| ESM C 600M encoder | 600M | No (frozen) |
| Layer averaging weights | 4 | Yes |
| Global attention pooling | ~1.5M | Yes |
| DBD attention pooling | ~1.5M | Yes |
| Projection (2304→512) | ~1.2M | Yes |
| MOE experts (12×) | ~25.2M | Yes |
| Gating + FiLM + family embed | ~0.5M | Yes |
| Motif length head | ~0.2M | Yes |
| PWM regression head | ~0.5M | Yes |
| **Total new trainable** | **~30.6M** | **Yes** |
| **Total model** | **~630M** | 30.6M trainable |

---

*Documents generated from Phase 1 brainstorm sessions with tf-binding-expert, protein-lm-expert, and moe-architect agents.*
