# TFScope Loss Functions

All losses are defined in `src/tfscope/losses/`. The total objective is a weighted sum of five terms:

```
L_total = w_gate · σ_gate⁻¹ · L_gate + log σ_gate
        +           σ_pwm⁻¹  · L_pwm  + log σ_pwm
        + L_balance
        + L_diversity
```

where `σ_gate`, `σ_pwm` are learned uncertainty parameters (Kendall & Gal 2018).

---

## 1. Gate Loss — `L_gate`

**File:** `tfscope_loss.py:33–41`  
**Config:** `gate_loss_weight = 1.0`, `gate_ordinal_weight = 0.05`

### 1a. Binary Cross-Entropy (per position)

The `PositionGateHead` outputs a logit per position `i ∈ {1…20}`. The target is the binary PWM mask (1 = motif position, 0 = padding).

$$L_{\text{BCE}} = -\frac{1}{B \cdot L} \sum_{b,i} \left[ m_{b,i} \log \sigma(g_{b,i}) + (1 - m_{b,i}) \log(1 - \sigma(g_{b,i})) \right]$$

where $g_{b,i}$ is the gate logit and $m_{b,i} \in \{0,1\}$ is the mask target.

### 1b. Ordinal Regularization

Encourages the gate to be **monotonically decreasing** (left-aligned contiguous motifs), encoding the biological prior that TF binding sites are uninterrupted sequences.

$$L_{\text{ord}} = \frac{1}{B(L-1)} \sum_{b,i} \text{ReLU}\!\left(\sigma(g_{b,i+1}) - \sigma(g_{b,i})\right)$$

Any "rise" in gate probability (position $i+1$ being more active than $i$) is penalized.

**Combined gate loss:**

$$L_{\text{gate}} = L_{\text{BCE}} + 0.05 \cdot L_{\text{ord}}$$

---

## 2. PWM Loss — `L_pwm`

**File:** `tfscope_loss.py:44–51`

KL divergence between the predicted nucleotide distribution and the target PWM, averaged only over **valid (motif) positions** determined by the ground-truth mask.

$$L_{\text{pwm}} = \frac{1}{B} \sum_b \frac{1}{|\mathcal{V}_b|} \sum_{i \in \mathcal{V}_b} \text{KL}\!\left(q_{b,i} \,\|\, p_{b,i}\right)$$

where:
- $p_{b,i} = \text{softmax}(\text{logits}_{b,:,i})$ — predicted nucleotide distribution at position $i$
- $q_{b,i}$ — target PWM column at position $i$ (normalized frequency)
- $\mathcal{V}_b = \{i : m_{b,i} = 1\}$ — valid motif positions for sample $b$

KL divergence expands to:

$$\text{KL}(q \| p) = \sum_{n \in \{A,C,G,T\}} q_n \log \frac{q_n}{p_n}$$

Padding positions (uniform 0.25 target) are excluded by the mask, so they do not dilute the gradient signal.

---

## 3. Uncertainty Weighting — Kendall & Gal 2018

**File:** `tfscope_loss.py:53–58`

Instead of fixed loss weights, two scalar parameters `log_sigma_gate` and `log_sigma_pwm` are **learned jointly** with the model. The combined loss is:

$$L = w_g \cdot e^{-s_g} L_{\text{gate}} + s_g + e^{-s_p} L_{\text{pwm}} + s_p$$

where $s_g = \log \sigma_{\text{gate}}$, $s_p = \log \sigma_{\text{pwm}}$.

The log terms prevent the model from trivially minimizing loss by making $\sigma \to \infty$. As training progresses, the model automatically down-weights the noisier task.

> **Reference:** Kendall A & Gal Y (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR.

---

## 4. MoE Load Balance Loss — `L_balance`

**File:** `balance.py:5–14`  
**Config:** `balance_loss_weight = 0.05`

Switch Transformer–style loss that prevents expert collapse (all tokens routing to the same expert).

$$L_{\text{balance}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e \cdot P_e$$

where:
- $E = 12$ — number of experts
- $f_e$ — fraction of tokens actually dispatched to expert $e$ (hard top-k assignment)
- $P_e$ — mean soft routing probability for expert $e$ across the batch
- $\alpha = 0.05$

Minimizing $\sum_e f_e P_e$ encourages uniform utilization. The product of hard ($f_e$) and soft ($P_e$) assignments makes this differentiable.

> **Reference:** Fedus W, Zoph B & Shazeer N (2022). *Switch Transformers: Scaling to Trillion Parameter Models.* JMLR.

---

## 5. Family Diversity Loss — `L_diversity`

**File:** `balance.py:17–33`  
**Config:** `diversity_loss_weight = 0.01`

Encourages different samples **within the same TF family** to be routed to different experts, so that experts can specialize across families rather than within them.

For each family $f$, compute the mean routing distribution over all samples in that family, then **maximize its entropy**:

$$L_{\text{diversity}} = -\frac{\alpha}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} H\!\left(\bar{p}_f\right), \quad \bar{p}_f = \frac{1}{|\mathcal{B}_f|}\sum_{b \in \mathcal{B}_f} \text{softmax}(g_b)$$

$$H(\bar{p}_f) = -\sum_e \bar{p}_{f,e} \log \bar{p}_{f,e}$$

High entropy means the family's samples are spread across many experts; low entropy means they all collapse to the same expert. The negative sign turns maximization into minimization.

---

## Summary Table

| Loss | Symbol | Supervision target | Weight |
|---|---|---|---|
| Gate BCE | $L_{\text{BCE}}$ | Binary position mask | 1.0 (× learned $\sigma_g^{-1}$) |
| Gate ordinal regularization | $L_{\text{ord}}$ | Monotone gate prior | 0.05 |
| PWM KL divergence | $L_{\text{pwm}}$ | Target PWM columns | learned $\sigma_p^{-1}$ |
| Uncertainty log terms | $s_g + s_p$ | — (regularization) | 1.0 |
| MoE load balance | $L_{\text{balance}}$ | Uniform expert usage | 0.05 |
| Family diversity | $L_{\text{diversity}}$ | Diverse intra-family routing | 0.01 |

---

## Notes on Current Behavior

- **`L_pwm` barely decreases during training** (0.71 → 0.64 over 40 epochs). The PWM head may benefit from: (1) a dedicated IC-weighted loss term that up-weights conserved positions, (2) adding MAE as an auxiliary PWM loss, or (3) a higher effective learning rate on the PWM head.
- **`L_gate` dominates early training** due to `gate_loss_weight = 1.0`. The uncertainty weighting should balance them automatically over time, but `sigma_gate` and `sigma_pwm` should be monitored in the training logs.
- **No explicit length regression loss** — length is inferred from the gate. A direct L1 length head could reduce `length_mae` (currently 1.37 bp on training data).

---

## Comparison: DeepPBS Loss Functions

**Source:** `/n/home13/leihuang/project/DeepPBS/deeppbs/nn/trainer.py`

DeepPBS uses a simpler two-term objective with a distinctly different philosophy from TFScope:

```
L_total = mse_weight × L_L1  +  ic_weight × L_IC
```

### DeepPBS Loss 1 — L1 (MAE) on softmax probabilities

```python
l1_loss(softmax(pred[mask]), target[mask])   # trainer.py:364
```

Direct mean absolute error between predicted and target PWM probability values, averaged over all valid positions and all 4 nucleotides. **This is exactly the `pwm_mae` metric we just added to TFScope's evaluation — but DeepPBS uses it as a training loss.**

### DeepPBS Loss 2 — Information Content Matching (IC Loss)

```python
|KL(log_background, target[mask]) - KL(log_background, softmax(pred[mask]))|
# where background = [0.25, 0.25, 0.25, 0.25]   trainer.py:351-358
```

This is the key novelty. Each term `KL(background ∥ p)` is the **information content** of distribution `p` relative to uniform background — the higher the IC, the more specific the motif column.

$$L_{\text{IC}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \left| \text{IC}(q_i) - \text{IC}(\hat{p}_i) \right|$$

$$\text{IC}(p) = \text{KL}(p \| \text{uniform}) = \sum_n p_n \log \frac{p_n}{0.25} = 2 - H(p) \text{ bits}$$

**This loss does not penalize predicting the wrong nucleotide** — it only penalizes predicting the wrong *level of specificity* at each position. A conserved A predicted as C is fine as long as the model is equally confident. This is a softer, shape-focused supervision signal.

### Key architectural difference

DeepPBS feeds both strands of DNA simultaneously (`y_pwm0`, `y_pwm1` stacked), and a commented-out symmetry loss (`symm_loss`) would have enforced reverse-complement consistency — disabled in current training but noteworthy as a biological prior.

---

## Comparison Table: TFScope vs DeepPBS

| Aspect | TFScope | DeepPBS |
|---|---|---|
| **Primary PWM loss** | KL(target ∥ pred) | L1 / MAE |
| **IC supervision** | None (only evaluation metric) | Explicit IC-matching loss |
| **Gate / length** | BCE per position + ordinal regularization | Not needed (fixed-length output) |
| **Loss balancing** | Learned uncertainty weights (Kendall & Gal) | Fixed scalar weights |
| **MoE regularization** | Load balance + family diversity | None |
| **Strand symmetry** | None | Defined but disabled |

### What TFScope could adopt from DeepPBS

1. **IC loss as an additional PWM term** — add `|IC(target_pos) - IC(pred_pos)|` alongside the existing KL loss. This would directly optimize the `ic_pearson` metric and give the PWM head a cleaner gradient at conserved positions where KL divergence may be numerically unstable.

2. **L1 as training loss** — replace or supplement KL divergence with L1 on softmax probabilities. L1 is more robust to outliers than KL (which explodes when `pred → 0` at positions where `target > 0`), and it directly optimizes `pwm_mae`.

3. **IC-weighted L1** — weight the L1 loss by the target IC per position (up-weight conserved positions). DeepPBS defines this but currently disables it via `rescaleWeight` returning a constant 1 — it is a promising direction.

**Suggested combined loss for TFScope:**

```
L_pwm_new = λ_kl · KL(target ∥ pred)   [current]
           + λ_l1 · L1(softmax(pred), target)   [from DeepPBS]
           + λ_ic · |IC(target) - IC(pred)|      [from DeepPBS]
```
