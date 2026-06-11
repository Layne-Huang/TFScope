# Brainstorm Session #2: Encoder Design — Protein Language Model Strategy

**Agent:** protein-lm-expert
**Date:** 2026-04-16
**Topic:** ESM model selection, embedding extraction, pooling design, compute budget

---

## 1. Model Selection: ESM C 600M (Not ESM3)

### Why ESM C, Not ESM3

- **ESM3** is a generative multimodal model (sequence + structure + function tracks). Its multi-track encoding adds computational overhead without benefit for pure representation extraction.
- **ESM C** is purpose-built for representation learning — clean Pre-LN transformer with rotary embeddings, SwiGLU activations, no biases, 2048 token context.

### ESM C Model Specifications

| Model | Params | Layers | d_model | Heads | Context |
|-------|--------|--------|---------|-------|---------|
| esmc-300m | 300M | 30 | 960 | 15 | 2048 |
| **esmc-600m** | **600M** | **36** | **1152** | **18** | **2048** |
| esmc-6b | 6B | 80 | 2560 | 40 | 2048 |

**Recommendation: esmc-600m.** ESM C 300M matches ESM2 650M; ESM C 600M rivals ESM2 3B and approaches ESM2 15B performance. Single-GPU friendly (1.2GB fp16).

### ESM3 Reference (if needed)

| Model | Params | Notes |
|-------|--------|-------|
| esm3-sm-open-v1 | 1.4B | Open weights, only practical choice |
| esm3-medium-2024-08 | 7B | |
| esm3-large-2024-03 | 98B | Impractical |

---

## 2. Layer Selection: Weighted Average of Last 4 Layers

Different layers capture different abstraction levels:
- **Lower layers:** local physicochemical properties
- **Middle layers:** secondary structure, contact patterns
- **Upper layers:** functional and family-level information

**Strategy:** Learnable scalar weights, softmax-normalized, over last 4 layers (indices 32-35).

```python
# For ESM C 600M (36 layers, d_model=1152)
layer_weights = nn.Parameter(torch.zeros(4))  # learnable, softmax-normalized
weights = F.softmax(layer_weights, dim=0)
# embedding = sum(weights[i] * layer_output[32+i] for i in range(4))
# Output dim: 1152 (no expansion)
```

This preserves all information without quadrupling the dimension (as concatenation would). More robust to noise in individual layers.

---

## 3. Frozen vs. Fine-tuned Encoder: Three-Phase Approach

### Phase 1 — Frozen Baseline

Freeze all ESM C parameters. Train only pooling heads, projection layers, MOE, and output heads (~2-5M trainable params).

### Phase 2 — Diagnostic Probing

Before fine-tuning, run diagnostic experiments:
1. **t-SNE/UMAP** of frozen embeddings colored by DBD family — are clusters separated?
2. **Linear probe:** linear classifier on frozen embeddings → DBD family. >90% accuracy means encoder already captures family distinctions.
3. **Fine-grained probe:** within a single family, can a linear probe predict specificity subtypes?

### Phase 3 — LoRA (Only If Probing Shows Gap)

- Apply LoRA to Q, K, V, O projection matrices in last 12 layers
- Rank r=8 or r=16, alpha=16 or 32
- Learning rate: 1e-5 (vs 1e-3 for new pooling/MOE params)
- Adds ~0.5-2M trainable parameters

**Why not full fine-tuning:** 600M params with full fine-tuning risks catastrophic forgetting. LoRA provides sufficient adaptation for the narrow domain shift.

---

## 4. Attention Pooling Design

### Architecture: Multi-Head Attention with Learned Query

```python
class AttentionPooling(nn.Module):
    def __init__(self, d_model=1152, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 144

        self.query = nn.Parameter(torch.randn(n_heads, 64))
        self.query_proj = nn.Linear(64, d_model // n_heads, bias=False)
        self.key_proj = nn.Linear(d_model, d_model // n_heads, bias=False)
        self.value_proj = nn.Linear(d_model, d_model // n_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = (d_model // n_heads) ** -0.5

    def forward(self, embeddings, mask=None):
        # embeddings: (B, seq_len, 1152)
        q = self.query_proj(self.query)                    # (n_heads, d_head)
        k = self.key_proj(embeddings)                      # (B, seq_len, d_head)
        v = self.value_proj(embeddings)                    # (B, seq_len, d_head)

        attn = torch.einsum('hd,bld->bhl', q, k) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), -1e9)
        attn_weights = F.softmax(attn, dim=-1)

        pooled = torch.einsum('bhl,bld->bhd', attn_weights, v)
        pooled = pooled.reshape(pooled.shape[0], -1)       # (B, d_model)
        return self.out_proj(pooled)
```

### Dual-Stream Setup

**Global stream:** mask = all ones (all residues). Captures cofactor, oligomerization, regulatory context.

**DBD stream:** mask = DBD boundary positions only. Focuses on recognition determinants.

Separate `AttentionPooling` instances (not shared weights) so they learn different attention patterns.

### Stream Combination

```python
global_feat = global_pooling(esm_embeddings)           # (B, 1152)
dbd_feat = dbd_pooling(esm_embeddings, dbd_mask)       # (B, 1152)
combined = torch.cat([global_feat, dbd_feat], dim=-1)  # (B, 2304)
projected = nn.Sequential(
    nn.Linear(2304, 512),
    nn.GELU(),
    nn.LayerNorm(512),
    nn.Dropout(0.1)
)(combined)                                             # (B, 512)
```

---

## 5. Input Representation

### Sequence Input
- Full TF protein sequence (all residues including regulatory domains, DBD, linkers)
- ESM C context: 2048 tokens. Most human TFs are 300-1500 residues — well within limit
- For rare cases >2048 residues, truncate to central region containing DBD + flanking

### DBD Boundary Annotation
- From InterPro/Pfam/UniProt feature tables
- Represented as boolean mask tensor: `dbd_mask[batch, seq_len]`
- For TFs with multiple DBDs (e.g., C2H2 arrays), unified mask covers all DBD positions

### Preprocessing Pipeline
1. Fetch TF sequence from UniProt by accession ID
2. Fetch DBD annotations from InterPro/Pfam
3. Construct DBD mask tensor
4. Feed sequence to ESM C → extract weighted layer-average embeddings
5. Apply dual-stream attention pooling

---

## 6. Computational Budget

### Inference (Single Protein)

| Component | Estimate |
|-----------|----------|
| Model params (fp16) | ~1.2 GB |
| GPU memory with activations (1000 residues) | ~3-4 GB |
| Forward pass (A100) | ~50-80 ms |
| Forward pass (RTX 3090) | ~100-150 ms |

### Training

| Configuration | GPU Memory | Training Time (10K TFs, 100 epochs) |
|--------------|------------|--------------------------------------|
| Frozen encoder, batch=32, A100 | ~6-8 GB | ~2-4 hours |
| LoRA fine-tuning, batch=32, A100 | ~12-16 GB | ~8-16 hours |

### Development Workflow
- **ESM C 300M** for rapid prototyping (fast iteration, fits any GPU)
- **ESM C 600M** for actual experiments and final model
- Same API, trivial to swap

---

## 7. Design Decision Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Encoder | ESM C 600M | Purpose-built for representation learning; no multi-track overhead |
| Embedding dim | 1152 | ESM C 600M d_model |
| Layer extraction | Weighted avg of last 4 | Preserves multi-level info without dimension explosion |
| Encoder training | Frozen → probe → LoRA | Avoid catastrophic forgetting; adapt only if needed |
| Global pooling | Multi-head attn, learned query | Superior to mean pooling; can focus on important residues |
| DBD pooling | Same arch, separate params, DBD mask | Focuses on recognition determinants |
| Stream combination | Concat + MLP (2304→512) | Preserves both global and local information |
| Input | Full sequence + DBD annotations | Full context; DBD boundaries from InterPro |
