# v18 Plan: Contact-Aware and Mutation-Sensitive TFScope PWM Head

**Version name:** `v18_contact_aware_mutation_sensitive`  
**Primary goal:** Fix the degenerate PWM-head cross-attention and make TFScope sensitive to specificity-switching variants, while preserving or improving absolute PWM prediction performance.

---

## 0. Motivation

Current TFScope versions show a serious failure mode in the PWM head:

1. **Position-independent cross-attention**  
   Every output PWM position attends to the same small set of DBD residues. The attention matrix is approximately rank-1.

2. **Input-independent / mutation-blind behavior**  
   For specificity-changing variants such as **KLF4 K409Q** and **MyoD L122R**, the WT and mutant attention maps are identical, and the predicted PWMs are also identical.

3. **Causal residues are ignored**  
   The mutated base-contacting residue receives zero attention mass.

4. **The problem is not caused by RAG**  
   A no-retrieval de novo model shows the same collapse, indicating that the degeneracy is intrinsic to the unconstrained PWM cross-attention head and the current PWM-only training objective.

Therefore, v18 should not simply tune retrieval, augmentation, or physics calibration. The main change should be:

> Replace the unconstrained PWM-to-DBD cross-attention with a **contact-aware residual correction head** that is explicitly forced to read plausible DNA-recognition residues.

---

# 1. v18 Core Hypothesis

The current model can predict average family-level motifs, but it does not learn the residue-to-base mapping required for variant sensitivity.

The v18 hypothesis is:

> TFScope will become mutation-sensitive if each motif position is constrained to read plausible base-contacting residues, and if mutations are modeled as residue-specific corrections to a shared base PWM prior.

In other words:

```text
v10/v14:
    protein sequence → global/family PWM prior

v18:
    global/family PWM prior
    + contact-aware residue-to-base correction
    + mutation-specific delta PWM
```

---

# 2. v18 Design Overview

## 2.1 High-level architecture

```text
Protein sequence
    ↓
ESM residue embeddings
    ↓
DBD family alignment / recognition-residue slots
    ↓
Global prior branch
    └── predicts seed/RAG PWM prior

Contact-aware branch
    ├── builds PWM-position × residue contact mask
    ├── masked/cosine cross-attention from PWM positions to DBD residues
    ├── predicts contact-based residual Δlogits
    └── optionally learns contact-code corrections

Mutation-delta branch
    ├── compares WT and mutant residue states
    ├── predicts ΔPWM only at affected contact positions
    └── suppresses retrieval prior at mutation-contacted columns

Final output
    ↓
Calibrated PWM
    + attention/contact attribution
    + mutation-effect ΔPWM
```

---

# 3. Key Architectural Change

## 3.1 From full PWM prediction to residual correction

Current PWM head implicitly tries to predict the entire PWM from cross-attention:

\[
z_{j,b}^{final} = f_{\theta}(q_j, ESM_{DBD})
\]

This allows the model to ignore contact residues and use family/global shortcuts.

v18 changes this to:

\[
z_{j,b}^{final}
=
z_{j,b}^{prior}
+
\lambda_j \Delta z_{j,b}^{contact}
+
\mu_j \Delta z_{j,b}^{mutation}
\]

where:

- `z_prior`: seed/RAG/global prior PWM logits;
- `Δz_contact`: contact-aware correction learned from DBD residues;
- `Δz_mutation`: mutation-specific correction;
- `λ_j`, `μ_j`: learned or rule-based gates.

This forces the cross-attention branch to serve a specific role:

> It should correct the base composition using contact residues, not predict the whole motif from global shortcuts.

---

# 4. Contact-Aware Cross-Attention

## 4.1 Problem with current attention

Current attention is unconstrained:

\[
A_{j,i}
=
softmax_i
\left(
\frac{Q_jK_i^T}{\sqrt{d}}
\right)
\]

where every motif position `j` can attend to every residue `i`. This causes hub-residue collapse.

## 4.2 v18 masked attention

v18 introduces a contact mask:

\[
M_{j,i}
=
\begin{cases}
0, & i \text{ can plausibly contact motif position } j \\
-\infty, & \text{otherwise}
\end{cases}
\]

Then:

\[
A_{j,i}
=
softmax_i
\left(
score(Q_j,K_i) + M_{j,i}
\right)
\]

Only plausible recognition residues are visible to each motif position.

---

# 5. Contact Mask Construction

v18 should support three contact-mask sources.

## 5.1 Source A: Family canonical contact maps

This is the most important first implementation.

### C2H2 zinc finger

For each finger, allow only canonical recognition helix residues:

```text
helix -1
helix +2
helix +3
helix +6
```

Each finger maps to approximately 3–4 DNA bases.

Example:

```text
Finger 1 recognition residues → motif positions 1–3/4
Finger 2 recognition residues → motif positions 4–6/7
Finger 3 recognition residues → motif positions 7–9/10
```

For KLF4, this should ensure that the residue corresponding to K409 is visible to the relevant motif column.

### bHLH

Use basic-region residues contacting E-box half-sites.

Example:

```text
basic-region contact residues → CANNTG / E-box core positions
```

For MyoD L122R, the mutated residue should be included in the contact mask for the affected E-box position.

### Homeodomain

Use recognition-helix positions such as:

```text
HD_pos47
HD_pos50
HD_pos51
HD_pos54
```

### bZIP

Use basic-region residues contacting half-site positions.

### ETS

Use recognition helix residues contacting the GGAA/T core.

---

## 5.2 Source B: PDB-derived contact maps

For training samples with known TF-DNA structures:

```text
direct base contact:
    protein side-chain heavy atom to DNA base heavy atom < 3.6 Å

weak base contact:
    3.6–5.0 Å

backbone contact:
    side-chain to sugar/phosphate
    not used directly for base preference, but can be auxiliary feature
```

Soft contact weight:

\[
w_{ij}
=
\exp
\left(
-\frac{d_{ij}-d_0}{\tau}
\right)
\]

where `d_ij` is the side-chain-to-base distance.

---

## 5.3 Source C: Boltz/AF3-derived predicted contact maps

This should be used later, after v18a/v18b are stable.

Pipeline:

```text
v10/v14 prior PWM
→ candidate DNA sequence
→ Boltz/AF3 TF-DNA complex
→ extract side-chain/base contacts
→ map residues to family-aligned positions
→ build contact mask
```

Important caveat:

> Do not depend on predicted structures in the first v18 implementation, because wrong seed DNA can produce wrong predicted contacts.

Use family canonical maps first.

---

# 6. Family-Aligned Recognition Residue Slots

Raw residue indices are not comparable across TFs. v18 must map raw residue positions into family-aligned recognition slots.

Examples:

```text
KLF4 raw residue K409 → C2H2 finger slot / helix position
MyoD raw residue L122 → bHLH basic-region contact slot
HOX raw residue Nxxx → Homeodomain_pos50
```

The model should use:

```text
family_id
aligned_contact_slot
amino_acid_identity
ESM_residue_embedding
```

not just raw sequence index.

Recommended intermediate table:

```text
sample_id
family
raw_residue_index
aligned_contact_slot
amino_acid
is_recognition_residue
contacted_motif_positions
```

---

# 7. Contact-Code Residual Layer

v18 should include a simple contact-code parameterization.

## 7.1 Contact-code table

Define:

\[
G(f,h,a,b)
\]

where:

- `f`: DBD family;
- `h`: family-aligned contact slot;
- `a`: amino acid;
- `b`: DNA base A/C/G/T.

Interpretation:

> In family `f`, amino acid `a` at contact slot `h` increases or decreases preference for base `b`.

## 7.2 Contact correction

For motif position `j`:

\[
\Delta z_{j,b}^{contact}
=
\sum_{i:(i,j)\in C}
A_{j,i}
\cdot
G(f,h_i,a_i,b)
\]

or neural version:

\[
\Delta z_{j,b}^{contact}
=
\sum_{i:(i,j)\in C}
A_{j,i}
\cdot
MLP([e_i, emb(h_i), emb(a_i), emb(f)])_b
\]

First version should use the simpler table/embedding version.

## 7.3 Zero-mean constraint

To make `G` identifiable:

\[
\sum_b G(f,h,a,b) = 0
\]

Implementation:

```python
G_centered = G - G.mean(dim=-1, keepdim=True)
```

---

# 8. Mutation-Specific Delta Branch

## 8.1 Why needed

For variant prediction, WT and mutant often retrieve the same prior PWM. Therefore, the difference must come from a mutation-aware branch, not from separate absolute PWM prediction.

## 8.2 Shared-prior variant mode

For WT and mutant:

```text
Use the same retrieval/global prior.
Only the mutation-delta branch is allowed to create WT-mutant differences.
```

Formula:

\[
z^{WT}
=
z^{prior}
+
\Delta z^{contact}(WT)
\]

\[
z^{Mut}
=
z^{prior}
+
\Delta z^{contact}(WT)
+
\Delta z^{mutation}
\]

or equivalently:

\[
z^{Mut}
=
z^{WT}
+
\Delta z^{mutation}
\]

## 8.3 Contact-code mutation delta

For a mutation at contact slot `h`:

\[
\Delta z_{j,b}^{mutation}
=
w_{h,j}
\left[
G(f,h,a_{mut},b)
-
G(f,h,a_{WT},b)
\right]
\]

This makes the model naturally sensitive to variants at recognition residues.

## 8.4 Mutation-aware retrieval suppression

If mutation residue `i` contacts motif position `j`, reduce the retrieval gate at that position:

\[
r_{j}^{mut}
=
r_j
\cdot
(1 - \rho m_j)
\]

where:

\[
m_j =
\max_{i \in mutated} w_{ij}
\]

Interpretation:

```text
If a mutation affects motif position j,
do not let retrieved PWM completely override the mutation signal.
```

---

# 9. Attention Degeneracy Fixes

v18 should directly address the observed rank-1 attention collapse.

## 9.1 Cosine attention

Replace dot-product attention with normalized cosine attention:

\[
score(Q_j,K_i)
=
\tau
\cdot
\frac{Q_j}{||Q_j||}
\cdot
\frac{K_i}{||K_i||}
\]

This reduces high-norm hub residue effects.

## 9.2 LayerNorm on K/V

Apply LayerNorm to ESM residue embeddings before projection:

```python
residue_emb = LayerNorm(residue_emb)
K = Wk(residue_emb)
V = Wv(residue_emb)
```

## 9.3 Remove global tokens from cross-attention

Do not allow PWM positions to attend to:

```text
CLS token
EOS token
padding token
any artificial global pooled token
```

## 9.4 Attention row-diversity loss

Prevent all PWM positions from using the same attention vector.

\[
L_{row-div}
=
\frac{1}{L(L-1)}
\sum_{j \neq k}
cos(A_j,A_k)
\]

Minimize `L_row-div`.

Use this only with contact mask, otherwise it may create artificial diversity.

## 9.5 Hub penalty

If a residue receives excessive attention across all motif positions:

\[
u_i = \sum_j A_{j,i}
\]

Penalize overly concentrated usage:

\[
L_{hub}
=
\sum_i
max(0, u_i - u_{max})^2
\]

---

# 10. Loss Functions

v18 loss should include both PWM accuracy and contact/mutation supervision.

## 10.1 PWM loss

Use existing PWM loss, but keep it balanced:

\[
L_{PWM}
=
CE(P_{target}, P_{pred})
+
\lambda_{pcc} L_{IC-PCC}
+
\lambda_{top} L_{top-base}
+
\lambda_{IC} L_{IC}
\]

Recommended:

```text
CE/KL weight: 1.0
IC-weighted PCC: 0.3–0.7
top-base margin: 0.05–0.1
IC matching: 0.2–0.4
```

## 10.2 Contact attention supervision

If target contact map is available:

\[
L_{contact}
=
KL(A^{target} || A^{pred})
\]

If only canonical contact prior is available, use softer supervision:

\[
L_{contact-prior}
=
-\sum_{j,i}
C^{prior}_{j,i}
\log A_{j,i}
\]

## 10.3 Attention row diversity

\[
L_{row-div}
=
mean_{j \neq k}
cos(A_j,A_k)
\]

## 10.4 Mutation delta loss

If WT/mutant experimental PWM pairs are available:

\[
L_{\Delta}
=
||\Delta PWM_{pred} - \Delta PWM_{true}||_1
+
\lambda
(1 - PCC(\Delta PWM_{pred}, \Delta PWM_{true}))
\]

## 10.5 Mutation attention loss

For a mutated contact residue `i_m` and affected motif positions `J_m`:

\[
L_{mut-attn}
=
-\log
\sum_{j \in J_m}
A_{j,i_m}
\]

This prevents zero attention on causal residues.

## 10.6 Sensitivity and stability losses

For contact mutations:

\[
L_{sens}
=
max(0, \epsilon - ||\Delta PWM_{pred}||_1)
\]

For non-contact mutations:

\[
L_{stable}
=
||\Delta PWM_{pred}||_1
\]

Use these carefully. Do not force every mutation to change PWM.

## 10.7 Total loss

\[
L =
L_{PWM}
+
\lambda_c L_{contact}
+
\lambda_d L_{row-div}
+
\lambda_{\Delta} L_{\Delta}
+
\lambda_m L_{mut-attn}
+
\lambda_s L_{sens/stable}
+
\lambda_{reg} ||G||_2^2
\]

Recommended initial weights:

```text
λ_c = 0.2–0.5
λ_d = 0.02–0.1
λ_Δ = 0.5 if mutant data exists, otherwise 0
λ_m = 0.1–0.3 for variant batches
λ_s = 0.05–0.1
λ_reg = 1e-4
```

---

# 11. Training Plan

## Stage 1: v18a — Attention repair only

Goal:

> Make the PWM head stop using rank-1 hub attention.

Changes:

```text
- Keep v10/v14 prior branch.
- Add contact mask.
- Add cosine attention.
- Add LayerNorm on K/V.
- Remove global tokens from cross-attention.
- Make cross-attention output residual Δz_contact only.
- Add row-diversity and hub penalties.
```

No mutation training yet.

Success criteria:

```text
KLF4 attention rank > 1
PWM positions attend to different recognition residues
mutated residue attention mass > 0 if inside contact mask
WT vs mutant attention r < 1.0
absolute PWM performance not worse than v10 by more than ~0.03 r
```

---

## Stage 2: v18b — Contact-supervised head

Goal:

> Make attention approximate a real or canonical residue-base contact map.

Changes:

```text
- Add family canonical contact supervision.
- Add PDB-derived contact supervision where available.
- Train contact-code table G.
```

Success criteria:

```text
Attention aligns with known recognition residues.
Contact-code improves base composition on high-IC positions.
v18b > v18a on mean Pearson r or at least on variant sensitivity.
Shuffled contact map negative control does not improve.
```

---

## Stage 3: v18c — Mutation-aware head

Goal:

> Predict nonzero and directionally meaningful ΔPWM for specificity-switching variants.

Changes:

```text
- Add shared-prior variant mode.
- Add mutation-delta branch.
- Add mutation attention loss.
- Add sensitivity/stability losses.
- Use WT/mutant paired data where available.
```

Success criteria:

```text
KLF4 K409Q:
    predicted PWM should no longer be identical to WT
    mutated residue should receive attention
    affected motif columns should shift in the correct direction

MyoD L122R:
    predicted PWM should show nonzero specificity shift

Contact mutations:
    ΔPWM magnitude should be higher than non-contact mutations

Non-contact mutations:
    ΔPWM should remain small
```

---

## Stage 4: v18d — Integrate with RAG++ and augmented donors

Goal:

> Preserve absolute PWM performance while retaining mutation sensitivity.

Changes:

```text
- Use v14_RAG++ as prior branch.
- Use v18 contact-aware branch as residual correction.
- In variant mode, WT/mutant share retrieval prior.
- Suppress retrieval gate at mutation-contacted motif positions.
- Optionally use augmented data only as retrieval donor, not direct target.
```

Success criteria:

```text
Absolute PWM prediction approaches or exceeds v10/v14.
Variant sensitivity remains nonzero.
Retrieval does not collapse WT/mut predictions to identical PWMs.
```

---

# 12. Implementation Checklist

## 12.1 New files

```text
models/pwm_head_v18.py
models/contact_code.py
models/contact_mask.py
scripts/build_family_contact_masks.py
scripts/extract_pdb_contacts.py
scripts/build_recognition_slots.py
scripts/train_v18a_attention_repair.py
scripts/train_v18b_contact_supervised.py
scripts/train_v18c_variant_delta.py
scripts/evaluate_v18_attention.py
scripts/evaluate_v18_variants.py
```

## 12.2 New data files

```text
data/contact_maps/family_canonical_contacts.json
data/contact_maps/pdb_contacts.parquet
data/contact_maps/recognition_slots.parquet
data/contact_maps/contact_masks_train.pt
data/contact_maps/contact_masks_test.pt
data/variants/variant_pairs.json
data/variants/variant_eval_cases.json
```

## 12.3 Required metadata per sample

```text
sample_id
protein_sequence
dbd_sequence
dbd_family
dbd_start
dbd_end
pwm
motif_length
source_id
gene_symbol
uniprot_id
family_aligned_positions
recognition_residue_indices
```

## 12.4 Required metadata per variant

```text
variant_id
gene
family
wt_sequence
mut_sequence
mutation_raw_index
mutation_dbd_index
wt_aa
mut_aa
aligned_contact_slot
affected_motif_positions
wt_pwm_exp_optional
mut_pwm_exp_optional
```

---

# 13. Evaluation Plan

## 13.1 Absolute PWM metrics

```text
Mean Pearson r
Median Pearson r
IC-weighted Pearson r
CE/KL
MAE/RMSE
Top-1 base accuracy
AUC macro OvR
```

## 13.2 Attention diagnostics

For each model:

```text
attention row rank
mean attention entropy
row-to-row attention cosine similarity
mutated residue attention mass
hub residue usage
WT vs mutant attention correlation
attention mass on canonical contact residues
```

Required cases:

```text
KLF4 WT vs K409Q
MyoD WT vs L122R
at least one homeodomain case
at least one C2H2-ZF case
at least one bZIP/bHLH case
```

## 13.3 Variant metrics

```text
WT-mutant PWM correlation
ΔPWM L1 magnitude
ΔPWM Pearson with experimental delta if available
changed-position detection accuracy
changed-base direction accuracy
contact mutation ΔPWM magnitude
non-contact mutation ΔPWM magnitude
contact/non-contact sensitivity ratio
```

## 13.4 Negative controls

```text
1. Shuffled contact masks
2. Random recognition residue slots
3. Mutation outside DBD
4. Mutation in non-contact residue
5. WT sequence duplicated as mutant
6. Same model with contact mask disabled
```

Expected:

```text
True contact-aware model should outperform shuffled/random controls.
Non-contact mutations should not create large ΔPWM.
WT duplicated as mutant should produce near-zero ΔPWM.
```

---

# 14. Concrete KLF4 and MyoD Test Requirements

## KLF4 K409Q

Known qualitative expectation:

```text
WT motif resembles:      GGGCGGGGC
K409Q mutant resembles: GGGTGGGTG
```

v18 should show:

```text
- mutated residue attention mass > 0
- WT vs mutant attention map not identical
- WT vs mutant PWM not identical
- affected columns show G/C to T-like preference shift
```

## MyoD L122R

v18 should show:

```text
- mutated residue is inside bHLH contact mask
- mutation produces nonzero ΔPWM
- attention on mutation-associated motif position changes or remains focused but value changes
```

Important distinction:

> Attention map does not necessarily need to change dramatically if the model already attends to the correct residue.  
> The key is that the value/content from the mutated residue changes and produces ΔPWM.

---

# 15. Recommended Hyperparameters

```yaml
v18:
  prior_branch:
    use_v10_prior: true
    freeze_prior_initially: true

  attention:
    type: cosine
    temperature_init: 10.0
    kv_layernorm: true
    remove_global_tokens: true
    contact_mask: true
    mask_mode: canonical_first
    allow_weak_contacts: true

  contact_code:
    use_table: true
    zero_mean_per_base: true
    l2_reg: 1e-4

  residual:
    delta_scale_init: 0.1
    gate_type: position_specific
    gate_inputs:
      - contact_confidence
      - seed_entropy
      - prior_ic
      - family_embedding
      - mutation_contact_indicator

  losses:
    pwm_ce: 1.0
    ic_pcc: 0.5
    contact_supervision: 0.3
    row_diversity: 0.05
    hub_penalty: 0.05
    mutation_attention: 0.2
    mutation_delta: 0.5
    sensitivity_stability: 0.05

  training:
    stage1_epochs: 30
    stage2_epochs: 50
    stage3_epochs: 30
    lr_contact_head: 3e-4
    lr_contact_code: 1e-3
    lr_prior_branch: 1e-5
    retrieval_dropout: 0.15
```

---

# 16. Expected Outcomes

## Short-term

v18a/v18b should fix attention collapse:

```text
attention no longer rank-1
mutated residues receive nonzero mass
different motif positions attend to different recognition residues
```

## Medium-term

v18c should become variant-sensitive:

```text
WT and mutant PWMs no longer identical
contact mutations produce localized ΔPWM
non-contact mutations remain stable
```

## Long-term

v18d should combine performance and interpretability:

```text
absolute PWM performance close to or above v10/v14
variant sensitivity preserved
residue-to-base attribution available
contact-code interpretable
```

---

# 17. Main Risks and Mitigations

## Risk 1: Contact masks are incomplete or wrong

Mitigation:

```text
- Use soft masks instead of hard masks.
- Allow weak fallback attention to nearby DBD residues.
- Compare canonical vs PDB-derived masks.
```

## Risk 2: Absolute PWM performance drops

Mitigation:

```text
- Keep v10/v14 prior branch.
- Make contact branch residual only.
- Initialize delta scale small.
- Freeze prior branch during early training.
```

## Risk 3: Mutation sensitivity becomes overactive

Mitigation:

```text
- Add stability loss for non-contact mutations.
- Use mutation delta only when mutation maps to contact slot.
- Use gate controlled by contact confidence.
```

## Risk 4: Attention diversity creates artificial mapping

Mitigation:

```text
- Do not use row-diversity alone.
- Always pair it with contact masks or contact supervision.
- Include shuffled contact negative controls.
```

---

# 18. Minimal First Experiment

If time is limited, run this first:

```text
v18a_minimal:
    - freeze v10 prior
    - contact mask using family canonical contacts
    - cosine attention
    - contact residual Δz only
    - row-diversity loss
    - train contact head only
```

Evaluate on:

```text
KLF4 WT/K409Q
MyoD WT/L122R
original benchmark subset
LSO subset
```

Success threshold:

```text
KLF4 mutated residue attention mass > 0
WT-mut PWM r < 1.0
attention rank > 1
absolute mean r drop < 0.03 relative to v10
```

If this fails, the issue is likely in the contact-mask construction or family-aligned residue mapping, not in retrieval.

---

# 19. Summary

v18 should be designed around one central correction:

> Do not hope that unconstrained cross-attention will discover residue-to-base recognition. Explicitly constrain and supervise the PWM head to read plausible recognition residues.

The recommended v18 progression is:

```text
v18a: attention repair
v18b: contact-supervised recognition head
v18c: mutation-delta branch
v18d: RAG++ integration with mutation-aware retrieval suppression
```

This plan directly targets the observed failure:

```text
rank-1 attention
hub-residue collapse
zero mass on causal residues
WT-mutant identical predictions
NoRAG collapse
```

and turns the PWM head into a mechanistically interpretable, variant-sensitive module.
