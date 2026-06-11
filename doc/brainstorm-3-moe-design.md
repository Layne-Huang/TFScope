# Brainstorm Session #3: MOE Architecture Design

**Agent:** moe-architect
**Date:** 2026-04-16
**Topic:** MOE layer, routing, conditioning, loss design, output heads, training details

---

## 1. MOE Architecture: 12 Experts, Top-2 Sparse Routing

### Architecture Choice

**Sparse MOE** (not dense). With ~12 DBD families of vastly different sizes, sparse routing provides the right inductive bias. Dense routing would dilute specialization.

### Expert Specification

```
Expert MLP: Linear(512, 2048) → GELU → Linear(2048, 512)
Parameters per expert: ~2.1M
Total MOE parameters: ~25.2M (12 × 2.1M)
Activation: GELU (consistent with ESM C encoder)
Dropout: 0.1 on expert output
Residual: at MOE block level (not inside expert)
```

### MOE Block Forward Pass

```python
def moe_forward(x, family_id, experts, gating, film):
    # x: [B, 512], family_id: [B]

    # 1. Compute gating weights
    gate_logits = gating(x, family_id)               # [B, 12]
    gate_weights, top_indices = torch.topk(gate_logits, k=2)  # [B, 2]
    gate_weights = F.softmax(gate_weights, dim=-1)

    # 2. Dispatch to top-2 experts
    output = torch.zeros_like(x)
    for k in range(2):
        for i in range(12):
            mask = (top_indices[:, k] == i)
            if mask.any():
                h = experts[i](x[mask])              # expert MLP
                h = film(h, family_id[mask])          # FiLM conditioning
                output[mask] += gate_weights[mask, k:k+1] * h

    # 3. Residual connection
    return x + output
```

### Why 12 Experts (Not 1-per-family)

Allows cross-family knowledge transfer when biologically appropriate (e.g., bHLH and bZIP share basic region contacts). Avoids dead experts for rare families.

---

## 2. Conditioning: FiLM + Family-Aware Gating Bias

### Family Embedding

```python
class FamilyEmbedding(nn.Module):
    def __init__(self, num_families=12, embed_dim=64):
        self.embedding = nn.Embedding(num_families, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, family_id):
        return self.embedding(family_id)  # [B, 64]
```

### FiLM Modulation (Inside Each Expert)

After expert MLP produces output, apply family-specific scale and shift:

```python
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim=512, family_embed_dim=64):
        self.gamma_net = nn.Linear(family_embed_dim, feature_dim)
        self.beta_net = nn.Linear(family_embed_dim, feature_dim)
        # Init gamma near 1, beta near 0
        nn.init.ones_(self.gamma_net.weight)
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, features, family_embed):
        gamma = 1.0 + self.gamma_net(family_embed)   # [B, 512]
        beta = self.beta_net(family_embed)             # [B, 512]
        return gamma * features + beta
```

**Why FiLM:** Parameter-efficient (2 linear layers). Biologically motivated — same protein fold produces different specificities depending on family-specific grammar.

---

## 3. Routing Function: Family-Aware Gating

```python
class FamilyAwareGating(nn.Module):
    def __init__(self, input_dim=512, family_embed_dim=64, num_experts=12):
        self.projection = nn.Linear(input_dim + family_embed_dim, 256)
        self.gate = nn.Linear(256, num_experts)
        self.family_bias = nn.Embedding(num_families, num_experts)

    def forward(self, x, family_id, family_embed):
        h = torch.cat([x, family_embed], dim=-1)      # [B, 576]
        h = F.gelu(self.projection(h))                # [B, 256]
        logits = self.gate(h)                          # [B, 12]
        logits = logits + self.family_bias(family_id)  # soft family-expert alignment
        return logits
```

**Family-specific bias** softly encourages router to assign inputs to experts that have seen their family before, without hard-coding the mapping. Critical for rare families.

**Why top-2 (not top-1):** Top-1 risks routing oscillation and gives no gradient to second-best expert. Top-2 provides backup expert, important for families at expert boundaries.

---

## 4. Load Balancing

Four-pronged approach for the severe family imbalance (C2H2: ~700 vs HMG: ~20):

### (a) Auxiliary Load-Balance Loss (Switch Transformer)

```python
def load_balance_loss(gate_logits, top_indices, num_experts=12, alpha=0.01):
    B = gate_logits.shape[0]
    expert_mask = torch.zeros(B, num_experts)
    for k in range(2):
        expert_mask.scatter_(1, top_indices[:, k:k+1], 1.0)
    f = expert_mask.mean(dim=0)                        # fraction routed to each expert
    P = F.softmax(gate_logits, dim=-1).mean(dim=0)     # mean router probability
    return alpha * num_experts * (f * P).sum()
```

### (b) Family Diversity Loss

```python
def family_diversity_loss(gate_logits, family_id, num_families=12, num_experts=12, alpha=0.005):
    probs = F.softmax(gate_logits, dim=-1)
    loss = 0.0
    for f in range(num_families):
        mask = (family_id == f)
        if mask.sum() > 1:
            mean_probs = probs[mask].mean(dim=0)
            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
            loss -= entropy  # maximize entropy
    return alpha * loss / num_families
```

### (c) Capacity Factor: 1.25

Each expert handles 1.25× its "fair share." Tokens exceeding capacity pass through via residual.

### (d) Family-Stratified Sampling

- Minimum 2 samples per family per batch
- Oversample rare families with replacement
- Sample weight: `1 / sqrt(count_family)`

---

## 5. Multi-Task Loss: Uncertainty Weighting

```python
class TFScopeLoss(nn.Module):
    def __init__(self):
        self.log_sigma_length = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_pwm = nn.Parameter(torch.tensor(0.0))

    def forward(self, pred_length, pred_pwm, target_length, target_pwm,
                pwm_mask, gate_logits, top_indices, family_id):

        # Task 1: Motif length classification (CE with label smoothing)
        L_length = F.cross_entropy(pred_length, target_length, label_smoothing=0.1)

        # Task 2: PWM regression (KL divergence, masked)
        pred_probs = F.log_softmax(pred_pwm, dim=1)
        kl_per_pos = F.kl_div(pred_probs, target_pwm, reduction='none').sum(dim=1)
        valid_counts = pwm_mask.sum(dim=1, keepdim=True).clamp(min=1)
        L_pwm = ((kl_per_pos * pwm_mask).sum(dim=1) / valid_counts.squeeze()).mean()

        # Auxiliary: load balance + diversity
        L_balance = load_balance_loss(gate_logits, top_indices)
        L_diversity = family_diversity_loss(gate_logits, family_id)

        # Uncertainty-weighted combination
        prec_l = torch.exp(-self.log_sigma_length)
        prec_p = torch.exp(-self.log_sigma_pwm)

        total = (prec_l * L_length + self.log_sigma_length +
                 prec_p * L_pwm + self.log_sigma_pwm +
                 L_balance + L_diversity)

        return total
```

---

## 6. Output Heads

### Motif Length Classification

```python
class MotifLengthHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=17):  # lengths 4..20
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)  # [B, 17]
```

### PWM Regression (with Self-Attention)

```python
class PWMRegressionHead(nn.Module):
    def __init__(self, input_dim=512, max_length=20):
        self.max_length = max_length
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, 64) * 0.02)
        self.pos_projection = nn.Sequential(
            nn.Linear(input_dim + 64, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU()
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(128)
        self.nucleotide_head = nn.Linear(128, 4)  # A, C, G, T

    def forward(self, x, motif_length=None):
        B = x.shape[0]
        x_expanded = x.unsqueeze(1).expand(B, self.max_length, -1)  # [B, 20, 512]
        pos = self.pos_embed.expand(B, -1, -1)                       # [B, 20, 64]
        h = self.pos_projection(torch.cat([x_expanded, pos], dim=-1)) # [B, 20, 128]

        # Self-attention captures inter-position dependencies
        h_norm = self.attn_norm(h)
        h_attn, _ = self.self_attention(h_norm, h_norm, h_norm)
        h = h + h_attn  # residual

        logits = self.nucleotide_head(h).permute(0, 2, 1)  # [B, 4, 20]

        # Variable-length masking
        if motif_length is not None:
            mask = torch.zeros(B, self.max_length, device=x.device)
            for i, l in enumerate(motif_length):
                mask[i, :l] = 1.0
            logits = logits * mask.unsqueeze(1)
        return logits
```

**Why self-attention in PWM head:** Adjacent positions are correlated (dinucleotide dependencies, periodic zinc finger contacts, palindromic bZIP motifs). Position-independent MLP would miss these.

---

## 7. Training Details

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | Standard for transformers |
| Learning rate (heads) | 3e-4 | New parameters |
| Learning rate (LoRA) | 1e-5 | If fine-tuning encoder |
| Weight decay | 0.01 | |
| Beta1, Beta2 | 0.9, 0.98 | Beta2=0.98 following ESM configs |
| LR schedule | Cosine annealing + warmup | |
| Warmup steps | 2000 | ~10% of total |
| Total steps | 20000 | Sufficient for ~1500-2000 TFs |
| Batch size | 32 | Single A100 |
| Gradient clipping | 1.0 (global norm) | Critical for MOE stability |
| Dropout | 0.1 (all layers) | |
| Precision | BF16 mixed | Stable with MOE |
| Random seed | 42 (3 runs for error bars) | |

### Monitoring During Training

Log every 100 steps:
- Per-expert utilization (detect collapse)
- Per-family routing distribution
- Expert output norms (detect dead experts)
- Gating entropy (measure specialization)
- Per-task losses and learned uncertainty weights

### Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| Expert collapse | One expert >50% traffic | Increase alpha_balance, check sampling |
| Dead experts | <1% traffic for 1000 steps | Reinitialize weights, increase diversity loss |
| Routing oscillation | Top-2 indices change frequently | Add gating temperature annealing |
| Family ignoring | All families route to same expert | Increase family_bias init magnitude |

---

## 8. Ablation Experiment Plan

Run before committing to full training (~1-2 hours each on single A100):

| Experiment | Variants | Metric |
|-----------|----------|--------|
| **Routing** | Top-1 / Top-2 (proposed) / Dense soft | Expert diversity + val PWM KL |
| **Expert count** | 8 / 12 (proposed) / 16 | Per-family val performance |
| **Conditioning** | FiLM only / Bias only / Both (proposed) | Rare-family performance |
| **Feature extraction** | Last layer only / Last-4 avg (proposed) | Silhouette score + val PWM KL |

---

## 9. Proposed File Organization

```
src/tfscope/
  models/
    moe.py           # MOE block, gating network, FiLM
    experts.py       # Expert MLP definition
    heads.py         # MotifLengthHead, PWMRegressionHead
    backbone.py      # ESM C feature extraction + projection
    tfscope.py       # Full model combining all components
  losses/
    tfscope_loss.py  # Multi-task loss with uncertainty weighting
    balance.py       # Load balance + diversity losses
  data/
    dataset.py       # TF dataset with family-stratified sampling
  train/
    trainer.py       # Training loop with MOE monitoring
    config.py        # All hyperparameters as dataclass
```
