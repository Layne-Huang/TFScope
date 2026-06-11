# TFScope Experiment Summary

**Last updated:** 2026-06-03  
**Goal:** Predict TF–DNA binding specificity (PWM) from protein sequence alone, matching or exceeding structure-based DeepPBS.

> **⚠️ Sections 1–12 are the v7–v14 era and contain a stale headline** ("LSO-NN 0.812 beats DeepPBS" in §10/§12) — that figure was oracle-aligned/leaky (see §4a). The current, corrected state of the project is in **§13 (v17/v18 + fair evaluation)** below.

---

## 1. Project Architecture

```
TFScope Pipeline
├── Module 1: Seed ML Model
│   ├── Backbone: ESM-2 (650M) with LoRA fine-tuning (6 layers, rank=16)
│   ├── Pooling: Dual-stream gated attention (global + DBD-focused)
│   ├── MoE decoder: 12 experts, top-2 routing, 1 shared expert
│   ├── Family conditioning: Semantic embeddings (ProTrek + ESM-2)
│   ├── Output heads: PositionGate (motif length) + PWMRegression (4×L)
│   └── Optional: Retrieval-Augmented Generation (RAG)
│
└── Module 2: Structure-Based Refinement (Calibration)
    ├── Structure prediction: Boltz-2 (protein+DNA complex)
    ├── Physics-based scoring: MM-GBSA (OpenMM AMBER14/DNA.OL15 + GBn2)
    └── Output: Calibrated PWM via Boltzmann(−ΔΔG_bind/τ)
```

**Key data:**
- Training set: `data/processed/tf_pwm_deeppbs_only.parquet` (520 structure-derived TFs)
- Augmented set: `data/processed/tf_pwm_aug_dbd.parquet` (4247 TFs, adds CIS-BP/RCADE/MEME)
- Test split: `benchmark_no_val.json` — 130 blind test TFs (DeepPBS benchmark `id.txt`)
- Evaluation metric: Pearson r (mean per-position, averaged over TFs)

**Baseline — DeepPBS:** r=0.702, requires crystal structure of protein+DNA complex at inference.

---

## 1a. Detailed Model Architecture

### Shared Backbone (all versions)

```
Input: protein sequence (up to 1024 aa)
  ↓
ESM-2 (esm2_t33_650M_UR50D, 650M params)
  - Frozen weights + LoRA on last 6 transformer layers
  - LoRA rank=16, alpha=32 → 491K trainable params in backbone
  - Outputs: per-residue embeddings (L × 1280)
  ↓
DBD Indicator: learned embedding added to DBD residues before pooling
  ↓
Dual-Stream Gated Attention Pooling
  - global_pool:  attends over full sequence → (1280,)
  - dbd_pool:     attends only within DBD mask → (1280,)
  ↓
ProjectionHead: [global ‖ dbd] → 512-dim joint representation
  - Linear(2560→512) + LayerNorm + GELU + Dropout(0.1)
  ↓
MoE Block (DeepSeek-style)
  - 12 routed experts + 1 shared expert (always active)
  - top-2 routing per token
  - Expert hidden dim: 2048 (4× expansion)
  - Auxiliary losses: load balancing + diversity
  - Family-aware routing via SemanticFamilyEmbedding (64-dim, 10 families)
  ↓
Output Heads (in parallel):
  ┌─ PositionGateHead → (max_L,) logits → sigmoid → motif length mask
  └─ PWMRegressionHead → (4, max_L) logits → softmax → PWM
```

### PWMRegressionHead (all versions)

```
MoE output (B, 512)
  ↓ expand + concat with positional embedding (64-dim)
  ↓ pos_projection → (B, max_L, 128)
  ↓ self-attention across PWM positions
  ↓ cross-attention to ESM-2 DBD residues  [pwm_cross_attn=True]
    — each PWM position i attends to which DBD residues drive its nucleotide pref
  ↓ [RAG-specific path — see section 1b]
  ↓ nucleotide_head: Linear(128→4) → (B, 4, max_L) logits
```

### Loss Function (all versions)

```
L_total = L_pwm + L_gate + L_balance + L_diversity

L_pwm   = L1(pred_pwm, target_pwm)          [weight=1.0]
         + |IC(pred) - IC(target)|           [IC matching, weight=0.5]
         + entropy penalty                   [prevent flat preds, weight=0.1]
L_gate  = BCE(gate_logits, pwm_mask)         [motif length supervision]
         + ordinal violation penalty         [encourage contiguous motifs]
L_balance = MoE load balancing loss
L_diversity = MoE expert diversity loss
```

---

## 1b. RAG Design Evolution

### v7 (no RAG) — baseline

```
output = nucleotide_head(h)    # pure de-novo from ESM-2 features
```
No retrieval. The model predicts entirely from protein sequence.  
**Config:** `use_retrieval=False`, 520 training samples, lr=3e-4.

---

### v10 (RAG v9/v10 design) — additive log-prior

```
output = delta_logits + β_gated × combined_log_prior

where:
  delta_logits = nucleotide_head(h)            # de-novo ESM pathway
  combined_log_prior = Σ_k attn(h, nb_k) × log(retrieved_pwm_k)
  β_gated = β × sigmoid(conf_scale × (trust_score - conf_thresh))
```

Key components:
- **Neighbour projection:** each retrieved PWM column (4-dim) → d=128, with positional + K-index embeddings
- **Per-position attention:** at each output position L, attend over K=3 retrieved neighbour columns
- **TrustPredictor (v10):** learned head `f(query_features, retrieved_pwm_k, cos_sim_k)` → trust score per neighbour. Supervised by actual Pearson r between retrieved PWM and target PWM.
- **Confidence gate β:** if no neighbour is trustworthy → β→0 → fall back on de-novo
- **Retrieval dropout:** 15% of batches drop retrieval (CFG-style), keeping de-novo pathway active

**Config:** `use_retrieval=True`, `retrieval_dropout=0.15`, `retrieval_k=3`, `trust_loss_weight=0.5`,  
NN index: `tf_nn_index.json` (same source_id allowed), 520 training samples, lr=6e-4.

**Why it works:** 85% of training time the model sees retrieval → learns to use it. De-novo pathway stays functional via 15% dropout. Low retrieval dropout = critical success factor.

---

### v11 (RAG v10 + augmented data)

Same RAG design as v10, but:
- Training data: 4247 TFs (adds 3727 CIS-BP/RCADE/MEME ORIG__ entries)
- NN index: `tf_nn_index_aug.json` (includes ORIG__ entries as donors)
- More total steps (13,000 vs 3,400) and larger batch (64 vs 32)

**Result:** r=0.522 vs v10 r=0.541 — slightly worse despite more data. Augmented data does not improve performance; adding sequence-only (non-structure) training samples may hurt due to distribution shift.

---

### v12/v13 (residual prior design) — FAILED

```
output = log(prior) + α × delta(protein_features, prior)

where:
  prior = similarity-weighted average of K=3 retrieved PWMs
  delta = nucleotide_head(h + prior_cross_attn(h, prior))  # model sees prior
  α = exp(log_alpha)  [initialized to exp(-2) ≈ 0.13]
```

The idea: when α→0, output collapses to the retrieval prior → model starts near-retrieval and learns small corrections.

**Why it failed:**
1. **50% retrieval dropout** — when retrieval dropped, prior=uniform, output≈log(0.25)+0.13×delta ≈ uniform. Near-zero gradient for de-novo pathway → it never trains effectively.
2. **LSO index is harder** — same-source neighbors excluded, so priors are noisier during training.
3. **Net result:** v13 r=0.419 << v10 r=0.541.

**Config:** `use_retrieval=True`, `residual_prior=True`, `retrieval_dropout=0.50`, NN index: `tf_nn_index_lso.json`, 4117 training samples.

**Fix needed (v14):** Reduce dropout to 15–20%, or use two-stream design where de-novo and residual paths are separate and the de-novo pathway is always at full strength.

---

## 2. Test Set Leakage Categories

| Category | Definition | n | % |
|---|---|---|---|
| L2 (same source) | Same motif ID (source_id) in training | 115 | 88% |
| L1 (same gene) | Same gene, different motif | 13 | 10% |
| L0 (novel gene) | Completely unseen gene (both are RXRA) | 2 | 2% |

**Note:** L2 TFs share the same JASPAR/HOCOMOCO accession with training TFs but from different PDB structures. No direct sequence leakage; structural leakage exists via same DNA conformation.

---

## 3. Seed Model Experiments

### Model Versions

| Model | Training data | Architecture | Mean r | Median r | IC-r | MAE | AUC | CE |
|---|---|---|---|---|---|---|---|---|
| **v7 best** | 520 struct | ESM+MoE, no retrieval | 0.334 | 0.307 | 0.882 | 0.239 | 0.586 | 2.204 |
| **v10 ep100** | 520 struct | ESM+MoE + RAG (v9/v10 design) | **0.541** | 0.638 | **0.922** | **0.174** | **0.780** | 1.870 |
| **v11 ep075** | 4247 aug | ESM+MoE + RAG (v10) + aug index | 0.522 | 0.664 | 0.915 | 0.182 | 0.753 | 1.942 |
| **v13 best** | 4117 aug | ESM+MoE + residual RAG (v12) | 0.419 | 0.491 | 0.896 | 0.219 | 0.717 | 2.382 |
| **DeepPBS** | crystal struct | 3D structure-based | 0.702 | 0.728 | 0.468 | 0.138* | 0.789 | 1.163 |

*DeepPBS MAE divided by 4 for comparable scale (different normalization).

**Best seed model: v10 ep100.** Nearly ties DeepPBS on AUC (0.780 vs 0.789). Main gap is in mean Pearson r (base composition accuracy).

### Version Comparison Table

| | v7 | v10 | v11 | v13 |
|---|---|---|---|---|
| **Training samples** | 520 | 520 | 4247 | 4117 |
| **NN index** | — | tf_nn_index.json | tf_nn_index_aug.json | tf_nn_index_lso.json |
| **RAG design** | None | Additive log-prior | Additive log-prior | Residual prior |
| **Retrieval dropout** | — | 15% | 15% | 50% |
| **Trust predictor** | No | Yes | Yes | No |
| **Total steps** | 3,400 | 3,400 | 13,000 | 9,750 |
| **Batch size** | 32 | 32 | 64 | 64 |
| **Learning rate** | 3e-4 | 6e-4 | 6e-4 | 6e-4 |
| **LoRA rank** | 16 | 16 | 16 | 16 |
| **Checkpoint** | `deeppbs_v7_full/ckpt_best.pt` | `deeppbs_v10_single/ckpt_epoch100.pt` | `deeppbs_v11_aug_dbd/ckpt_epoch075.pt` | `deeppbs_v13_residual_rag_aug/ckpt_best.pt` |
| **Mean r** | 0.334 | **0.541** | 0.522 | 0.419 |

### Why v13 Underperforms v10

v13 uses the "residual prior" design (`output = log(prior) + α×delta`) with 50% retrieval dropout. When retrieval is dropped, prior = uniform and α=0.13, making the output nearly uniform → near-zero gradients → the de-novo pathway never develops strength. v10 uses additive design with 15% dropout, which works better.

### Key Insight: IC-r

Our models achieve **IC-r = 0.88–0.92** vs DeepPBS **0.468**. Our models correctly predict *which positions are information-rich* even without 3D structure. The remaining gap to DeepPBS is in exact base composition (mean r 0.54 vs 0.70).

---

## 4. Retrieval-Augmented Baseline (NN)

Pure nearest-neighbour lookup — no model, just copy the closest training TF's PWM.

> **⚠️ CORRECTION (2026-06-01):** The 0.812 LSO-NN figure below was **oracle-aligned (leaky)** — it aligned each retrieved PWM to the *target PWM itself* before correlating, which requires knowing the answer. It is NOT an achievable baseline. See the corrected numbers in the next subsection.

| Strategy (oracle-aligned, LEAKY) | All | L2 | L1 | L0 |
|---|---|---|---|---|
| Top-1 NN (same source allowed) | 0.903 | 0.919 | 0.772 | 0.846 |
| Leave-source-out (LSO) | 0.812 | 0.817 | 0.772 | 0.846 |
| Leave-gene-out | 0.771 | 0.771 | 0.754 | 0.846 |
| DeepPBS | 0.702 | 0.701 | 0.709 | 0.686 |

### 4a. Corrected LSO-NN baseline (per-column Pearson, DeepPBS-style)

Reconciled the 0.812 discrepancy. LSO top-1 NN copy, three alignment regimes:

| Variant | per-column r | Achievable at inference? |
|---|---|---|
| RAW (no alignment) | 0.334 | ✓ |
| SEED-aligned (align neighbor to v10 seed) | 0.477 | ✓ deployable |
| **ORACLE-aligned** (align to ground truth) | **0.841** | ✗ leaky — this was "0.812" |
| v10 model | 0.542 | ✓ |
| DeepPBS | 0.702 | ✓ (needs structure) |

**Corrected key findings:**
1. Deployable pure-retrieval ceiling ≈ **0.48** (seed-aligned) — BELOW v10 (0.542). Pure retrieval is NOT a shortcut past DeepPBS.
2. The oracle (0.84) proves the retrieved neighbors *contain* near-DeepPBS signal, but it is **only reachable by aligning to the answer** (leakage).
3. **Oracle best-of-K rises with K**: K=1→0.84, K=3→0.90, K=8→0.93, K=16→0.94, K=32→0.97. Alignment is the +0.51 lever; selection is +0.10.

### 4b. Alignment-teacher fusion — NEGATIVE RESULT (the retrieval path is exhausted)

Built and tested the strongest version of the plan's retrieval roadmap: K=8 neighbours, motif-aligned (offset × rev-comp), oracle-distillation teacher (selection + prior KL), position-specific mixture gate. Trained as a lightweight head on top of frozen v10 (`scripts/train_alignment_fusion.py`, `src/tfscope/models/alignment.py`, `scripts/build_aligned_retrieval.py`).

| Deployable method | per-column r |
|---|---|
| raw top-1 NN | 0.334 |
| neighbour-consensus (align to nn#1) | 0.336 |
| seed-aligned top-1 | 0.477 |
| v10 (de-novo + RAG) | 0.542 |
| **alignment-fusion head (trained)** | **0.549** |
| oracle (align to truth, LEAKY) | 0.841 |
| DeepPBS | 0.702 |

**Conclusion: retrieval is exhausted under clean LSO.** The fusion head gained only +0.007 over v10. The reason: the deployable seed-aligned prior (0.48) is *weaker than the seed itself* (0.54), so the learned gate correctly ignores it. The oracle ceiling is unreachable because alignment quality is bounded by the reference, and the best reference (the seed, 0.54) isn't good enough. **The v10_improvement_plan's retrieval-centric short-term line (P2→P3→P4→P5) chases a leakage artifact and will not reach DeepPBS.**

### 4c. Corrected direction: improve the de-novo prediction (v14)

Since retrieval caps at ~0.55, the lever is the **de-novo pathway**. v10 has IC-r=0.92 (knows *which* positions matter) but column-r=0.54 (base composition wrong). Two new loss terms target this directly (`src/tfscope/losses/tfscope_loss.py`):
- **IC-weighted per-column Pearson loss** — optimises the exact reported metric, weighted by target IC.
- **Top-base margin loss** — hinge pushing the true top base above the runner-up at high-IC positions.

v14 = v10 recipe (additive RAG, 15% dropout, 520 TFs) + these terms. Sbatch: `scripts/train_v14_icpcc.sbatch`. Baseline to beat: 0.542.

**Open decision (plan P0):** LSO (de-novo claim, retrieval won't help, must improve de-novo) vs original same-source-allowed index (raw retrieval 0.556, homology transfer, weaker novelty). Determines whether retrieval is part of the story at all.

**Files:**
- NN index (original): `data/processed/tf_nn_index_aug.json`
- NN index (LSO): `data/processed/tf_nn_index_lso.json`
- Evaluation script: `scripts/evaluate_nn_baseline.py`

---

## 5. Rosetta Calibration (Module 2 — original)

**Pipeline:** v10 seed PWM → consensus DNA → Boltz-2 (single-sequence) → Rosetta RM8B ΔΔG scan → Boltzmann PWM

**Force field:** RM8B_torsional.wts (beta_nov16 + DNA torsional terms)

| Method | Mean r | CE | AUC | Improved/130 |
|---|---|---|---|---|
| v10 seed (no calibration) | 0.541 | 1.870 | 0.780 | — |
| AF3+Rosetta (τ=1.5) | 0.301 | 2.761 | 0.659 | 31/130 |
| AF3+Rosetta (τ=4.0) | 0.288 | 2.050 | — | — |
| AF3+Rosetta (τ=10.0) | 0.271 | 1.738 | — | — |
| Crystal struct+Rosetta | 0.463 | — | — | — |

**Conclusion:** Rosetta calibration consistently hurts. Even using ground-truth crystal structures (r=0.463 < v10 seed 0.541), confirming the problem is Rosetta's energy function, not structure quality. Rosetta's `hbond_sc` and `fa_elec` terms are not calibrated for DNA base-preference discrimination.

**Boltz iPTM vs improvement correlation:** Pearson r=0.38 — weak predictor. Even Q4 (highest iPTM) only improves 24% of cases.

---

## 6. Seed-Calibration Mixture (Idea 1)

Mix v10 seed and Rosetta calibration: `final = (1−w)×seed + w×calibrated`, where `w = iPTM × (1 − seed_IC_norm)`.

| Method | Mean r | CE | MAE | AUC |
|---|---|---|---|---|
| v10 seed | 0.542 | 1.870 | 0.174 | 0.780 |
| **Mixture w_scale=0.3** | **0.540** | **1.491** | **0.174** | 0.775 |
| Mixture w_scale=1.0 | 0.528 | 1.434 | 0.182 | 0.764 |
| Rosetta cal (original) | 0.308 | 2.759 | 0.240 | 0.662 |
| DeepPBS | 0.577* | 1.163 | 0.189 | 0.789 |

*DeepPBS r recalculated from stored predictions for fair comparison.

**Conclusion:** Mixture at w_scale=0.3 preserves r and MAE of seed while reducing CE by 20% (1.870→1.491). Best calibration result with Rosetta.

---

## 7. MM-GBSA Calibration (Module 2 — improved)

**Pipeline:** v10 seed PWM → consensus DNA → Boltz-2 (single-sequence, no MSA) → PyRosetta DNA mutation → OpenMM AMBER14/DNA.OL15 + GBn2 → ΔΔG_bind = ΔE_complex − ΔE_DNA → Boltzmann PWM (τ=1.5)

**Implementation details:**
- PyRosetta mutates each DNA base (double-strand with complement) using `mutate_dna_base`
- OpenMM with AMBER14/ff14SB + DNA.OL15 + GBn2 implicit solvent (`soluteDielectric=2.0`, `solventDielectric=78.5`)
- Reference state correction: `ΔΔG_bind = ΔE_complex − ΔE_DNA` cancels intrinsic base energy differences
- Without correction: ΔΔG ≈ ±175 kcal/mol (dominated by base identity change); with correction: ±5–25 kcal/mol (binding preference signal)
- ~2.5 min/TF on A100 GPU with `min_iter=300`

**Parameter sensitivity (critical finding):**
- `min_iter=100` → noisy ΔΔG, wrong consensus (`CCTCCCGTCCCG` for CEBPB)
- `min_iter=300` → stable ΔΔG, meaningful consensus
- Different Boltz structures (same sequence, different random seed) → completely different MM-GBSA results: MM-GBSA is **highly sensitive to the predicted conformation**, not just the force field

**CEBPB single-sample result (min_iter=300, no-MSA Boltz):**

| Method | Pearson r | Δ vs seed |
|---|---|---|
| v10 seed | −0.079 | — |
| Rosetta cal | +0.060 | +0.138 |
| **MM-GBSA** | **+0.175** | **+0.254** |

---

### 7a. MM-GBSA Pilot — 28-Sample Results (COMPLETED)

Ran on 28/31 samples where Rosetta originally improved (3 failed due to NaN in energy minimization).

| Method | Mean r | Median r | Improved/28 | Mean Δr vs seed |
|---|---|---|---|---|
| v10 seed (baseline) | 0.147 | 0.041 | — | — |
| Rosetta cal (τ=1.5) | 0.181 | 0.055 | 18/28 | +0.035 |
| **MM-GBSA (τ=1.5)** | **0.025** | **0.015** | **12/28** | **−0.121** |

**MM-GBSA is overall worse than Rosetta on this pilot.** Despite CEBPB being a positive case, the majority of samples are hurt. Bimodal behaviour: a few large wins (+0.4–0.6 Δr for RXRA, ZBTB33, HSF1), but most samples degrade.

**Top 5 MM-GBSA wins:**

| Sample | Seed r | MM-GBSA r | Δr |
|---|---|---|---|
| 6fbr_B_RXRA | −0.214 | +0.411 | +0.624 |
| 6dfc_A_ZBTB33 | −0.242 | +0.337 | +0.579 |
| 5vmv_A_ZBTB33 | −0.264 | +0.251 | +0.515 |
| 4f6n_A_ZBTB33 | +0.016 | +0.450 | +0.435 |
| 7dcj_B_HSF1 | −0.262 | +0.166 | +0.428 |

**Root cause of failures:** Boltz-predicted structures are not reliable enough for single-structure MM-GBSA. When the protein-DNA interface is correct, MM-GBSA extracts real signal; when it's not, ΔΔG values are noise. Two Boltz runs of the same complex can give completely different MM-GBSA predictions.

**Key limitation:** Seed propagation problem — if the seed PWM is wrong, Boltz builds the structure around the wrong DNA, and MM-GBSA scans from that wrong starting point.

---

## 8. Iterative MM-GBSA — CEBPB Experiment (COMPLETED)

**Pipeline:** Repeat until convergence:
1. Compute argmax consensus from MM-GBSA PWM
2. Write new FASTA with new consensus DNA
3. Run Boltz-2 + MSA server (better structure quality, iPTM ~0.95)
4. Run MM-GBSA scan (min_iter=300, soluteDielectric=2.0)
5. Stop if consensus unchanged

**Two experiments on CEBPB (true motif: TTGCGCAA):**

| Experiment | Round 0 DNA | MM-GBSA consensus | Pearson r | Round 1 |
|---|---|---|---|---|
| **Wrong start** | `TATTTAAAAATA` (v10 seed, wrong) | `TATTTAAATATA` | −0.024 | NaN crash |
| **Correct start** | `TATTGCGCAATA` (true motif) | `ATAATGTACGAT` | **+0.318** | NaN crash |

**Key findings:**

1. **Wrong start cannot escape bad minimum.** Starting from AT-rich DNA, MM-GBSA consensus barely changes (`TATTTAAAAATA` → `TATTTAAATATA`). The protein is posed against the wrong DNA, so MM-GBSA correctly reads that AT is preferred — in that (wrong) structure. r=−0.024 barely improves from seed r=−0.079.

2. **Correct start gives meaningful improvement.** Starting from the true CEBPB DNA, MM-GBSA consensus `ATAATGTACGAT` achieves r=+0.318 — substantially better than seed (−0.079) and Rosetta (+0.060). The physics correctly identifies some nucleotide preferences when given a reasonable starting structure.

3. **NaN crash in round 1** — Boltz occasionally generates degenerate structures for non-natural DNA sequences (new consensus ≠ original training DNA). Need NaN guard + retry with different Boltz seed.

4. **min_iter matters:** min_iter=100 gave nonsensical `CCTCCCGTCCCG`; min_iter=300 gave physically reasonable results. Confirmed the structural sensitivity: two different Boltz runs (same sequence) gave completely different MM-GBSA outputs.

**Script:** `scripts/iterate_mmpbsa.py`

**Conclusion:** Iterative MM-GBSA works in principle (r=+0.318 from correct start) but is blocked by: (a) NaN crashes from degenerate Boltz structures, (b) inability to recover from wrong starting DNA. Needs crystal-quality structures or ensemble Boltz averaging to be reliable.

---

## 9. Boltz-2 MSA vs Single-Sequence

Testing impact of MSA on structure quality for protein-DNA complexes.

| Mode | iPTM (1a1g Egr1) |
|---|---|
| Single-sequence | 0.932 |
| **With MSA server** | **0.964** |

MSA improves iPTM by ~3%. Boltz run for all 130 test TFs with MSA pending (job 17727673, gpu_h200).

**Fix required:** Input FASTAs must use `>A|protein` (not `>A|protein|empty`) to enable MSA fetching.

---

## 10. Summary Table — All Methods

| Method | Mean r | Requires 3D structure? | Notes |
|---|---|---|---|
| DeepPBS | 0.702 | ✅ Yes (crystal) | 3D state-of-the-art |
| **NN LSO baseline** | **0.812** | ❌ No | Beats DeepPBS; no model |
| v10 seed | 0.541 | ❌ No | Best sequence-only model |
| v10 + Mixture | 0.540 | Partial (Boltz) | Better CE, same r |
| v10 + MM-GBSA (pilot 28 TF) | 0.025 | Partial (Boltz) | Worse than Rosetta overall; bimodal |
| v10 + Rosetta | 0.301 | Partial (Boltz) | Hurts performance |
| Crystal + Rosetta | 0.463 | ✅ Yes (crystal) | Worse than seed alone |
| v13 residual RAG | 0.419 | ❌ No | Underperforms v10 |
| v11 RAG | 0.522 | ❌ No | Slightly below v10 |

---

## 11. Key Files

| File | Description |
|---|---|
| `scripts/evaluate.py` | Main evaluation script |
| `scripts/run_mmpbsa_scan.py` | MM-GBSA DNA base scanning |
| `scripts/iterate_mmpbsa.py` | Iterative Boltz+MM-GBSA calibration |
| `scripts/train_v12_residual_rag.sbatch` | v12 training (residual prior, LSO index) |
| `scripts/train_v13_residual_rag_aug.sbatch` | v13 training (aug dataset) |
| `scripts/run_boltz_v10_msa.sbatch` | Boltz-2 with MSA for all 130 test TFs |
| `data/processed/tf_nn_index_lso.json` | Leave-source-out NN index (K=3) |
| `data/processed/splits/deeppbs_only/benchmark_no_val_aug.json` | Extended split (4117 train) |
| `results/mmpbsa_v10_pilot/` | MM-GBSA pilot results (31 samples) |
| `results/mmpbsa_iterate_cebpb/` | Iterative calibration results (CEBPB) |
| `results/tfscope_v10_ep100/` | v10 benchmark results |

---

## 12. Next Steps

1. **Fix NaN crash in iterative MM-GBSA** — add retry with different Boltz random seed when NaN detected; enables multi-round iteration
2. **Fix v14 architecture** — lower retrieval dropout (15–20%), two-stream design where de-novo pathway is always active
3. **Boltz MSA run** (job 17727673, pending gpu_h200) — better structures for all 130 test TFs
4. **MM-GBSA with crystal structures** — run MM-GBSA on actual crystal structures to upper-bound what physics can give
5. **Ensemble MM-GBSA** — average ΔΔG over 3–5 Boltz structures per TF to reduce structural noise
6. **Paper narrative:** Sequence-only + LSO retrieval (r=0.812) already exceeds DeepPBS (r=0.702); iterative MM-GBSA from correct start gives further gains (CEBPB r=+0.318 vs seed −0.079)

> Item 6 is **superseded** — the 0.812 was leaky. See §13 for the corrected, honest comparison.

---

## 13. v17 / v18 + Fair Evaluation (2026-06-03)

### 13.1 The degenerate cross-attention problem
The legacy PWM-head cross-attention (`pwm_head.cross_attn`) collapsed to **rank-1**: every PWM
column attends to the same ~3 DBD residues (row-constancy 0.6–0.8), with a terminal-residue
**sink**, and **zero attention mass on specificity-switching residues** (KLF4 K409, MyoD L122).
Consequence: the model is **mutation-blind** (WT vs mutant predicted PWM Pearson r = 1.000).
This is **RAG-independent** — a no-retrieval model (`deeppbs_v14_noRAG`) shows the identical
collapse, so it is intrinsic to the unconstrained cross-attention + PWM-only loss, *not* a
retrieval artifact.
Scripts: `viz_attn_compare.py`, `viz_attn_testset.py`, `attn_v18.py`, `viz_attn_v18.py`.

### 13.2 v18 design (`src/tfscope/models/pwm_head_v18.py`, `plan/v18_plan.md`)
The legacy head becomes a **frozen prior branch**; a new **contact-aware residual branch** adds
`logits = z_prior + λ·Δz_contact`. Branch = cosine cross-attention + LayerNorm on K/V +
amino-acid-identity values + row-diversity + hub penalties.
- **v18a** = attention repair only (no supervision).
- **v18b** = + family-canonical recognition-residue supervision (`build_recognition_prior.py` →
  `data/contact_maps/recognition_residues.json`; rule-based, leak-free, NOT from test PDBs).
- **v18c** (planned) = mutation-contrastive / sensitivity loss + larger learnable λ.

Both built on the **v17_200ep** prior (LGO index = leakage-free/honest), `--lora-rank 0`,
`--v18-freeze-prior` (only the 0.55M contact-branch params train).
Launch: `scripts/train_v18a.sbatch`, `scripts/train_v18b.sbatch` (gpu_test, batch 16).

### 13.3 v18a result — repair worked; mutation sensitivity did NOT (yet)
| | v17 | v18a |
|---|---|---|
| KLF4 row-constancy | 0.81 | **0.25** (collapse broken) |
| KLF4 attn entropy (/4.42) | 1.47 | **2.36** (spread, no sink) |
| KLF4 mass on K409 | **0.000** | **0.094** (now reads it) |
| KLF4 WT-vs-mut output PWM r | 0.9997 | 0.9998 (still blind) |
The attention is repaired (non-degenerate, reads the causal residue) but the output is still
mutation-blind — λ is small, the frozen prior is blind, and nothing rewards sensitivity yet.
Mutation sensitivity is a **v18c** goal, not v18a. Heatmap: `results/v18_attn/attn_v17_vs_v18a.png`.

### 13.4 Two fair evaluation protocols (every model incl. DeepPBS scored identically)
1. **Trimmed-core, offset+RC aligned** (`eval_trimmed_core.py`, `eval_full_metrics.py`): trim the
   target to its IC≥0.25 informative core, grant every model oracle offset+RC alignment. Fair
   *ranking*; absolute values are upper bounds.
2. **Canonical-fixed deployable** (`eval_canonical_registration.py`): apply the v16 canonicalize
   (trim + canonical-strand) to BOTH prediction and target, fixed scoring, no alignment. (v16 must
   use the canon parquet for retrieval donors — see `DATA_OF`. The `canon_fixed_r` inside
   `eval_full_metrics.py` has a target-strand bug — use `eval_canonical_registration.py` numbers.)

### 13.5 RESULTS — full metric panel (116 DeepPBS-covered TFs)

> **Aligner bug (found+fixed 2026-06-03):** `align_pwm` picked the shift maximising *mean* per-col
> r over the overlap with only a 2-col floor → cherry-picked tiny high-r windows (Egr1 took a
> 2/10-col overlap r=0.994 over the honest 10/10 r=0.982), inflating every motif-level number
> ~0.05. Fixed via `coverage_norm=True` (selection score = Σr/Lr). Numbers below are corrected.

**Motif-level (trimmed core IC≥0.25, offset+RC aligned) — honest v18a still beats DeepPBS on 8/11;
v18b is WORSE than v18a (contact supervision hurt accuracy):**

| Metric | v17 | v18a (honest, LGO) | v18b | DeepPBS |
|---|---|---|---|---|
| Mean Pearson r | 0.700 | **0.802** | 0.723 | 0.750 |
| Median Pearson r | 0.723 | **0.831** | 0.742 | 0.759 |
| IC-weighted r | 0.954 | **0.972** | 0.958 | 0.968 |
| MAE ↓ | 0.141 | **0.109** | 0.134 | 0.133 |
| RMSE ↓ | 0.242 | **0.184** | 0.230 | 0.204 |
| Top-1 accuracy | 0.753 | **0.836** | 0.767 | 0.793 |
| F1 (macro) | 0.646 | **0.765** | 0.674 | 0.725 |
| MCC | 0.615 | **0.765** | 0.651 | 0.698 |
| Cross-entropy ↓ | 1.14 | 0.99 | 1.13 | **0.84** |
| KL ↓ | 0.666 | 0.517 | 0.655 | **0.358** |
| AUC (macro OvR) | 0.879 | **0.939** | 0.891 | 0.937 |

DeepPBS keeps only the calibration metrics (CE/KL). v18a's **+0.10 mean r over v17 comes entirely
from the 0.55M contact branch** (prior frozen). Oracle-aligned ⇒ upper bound, applied to all
equally (fair ranking).

**Deployable — canonical-registration fixed (no oracle alignment; `eval_canonical_registration.py`):**

| | v14 (leaky) | v16 (leaky, canon-trained) | v17 | v18a | v18b | DeepPBS |
|---|---|---|---|---|---|---|
| Mean r | 0.424 | **0.503** | 0.338 | 0.420 | 0.326 | 0.419 |
| Median r | 0.560 | **0.669** | 0.368 | 0.456 | 0.346 | 0.648 |

(v18a ties DeepPBS on deployable mean; v18b is worse than v17 — contact supervision hurt the
deployable number too. The `canon_fixed` column inside `eval_full_metrics.py` uses a strand-buggy
target and should be ignored; these are authoritative.)

### 13.6 Key conclusions
1. **Registration is the dominant error for ALL methods, including DeepPBS** (~0.43 r lost from
   aligned→fixed). Improving the *emitted* register is worth more than any base-composition gain.
2. **On base composition (the seed model's job), honest leakage-free v18a exceeds DeepPBS** on
   correlation/error/classification metrics (8/11); DeepPBS retains a calibration (CE/KL) +
   deployable-register edge.
3. **v18b (contact supervision) did NOT help — it is worse than v18a** on every accuracy metric.
   Mutation diagnostic: v18b tripled KLF4 mass@K409 (0.094→0.309) but dropped MyoD mass@L122 to
   ~0.008, and BOTH stay output-mutation-blind. Marginal-pull-to-recognition-SET over-constrains.
   **v18a (repair only) is the best v18 variant.**
4. **v16 (trained in the canonical frame) wins the deployable metric** (mean+median > DeepPBS) —
   but v16 uses the leaky normal index.
5. Leakage hygiene: v14/v16 = normal index (leaky); v17/v18a/v18b = LGO (honest).

### 13.7 TODO
- **v18c** mutation-contrastive / sensitivity loss + larger learnable λ — supervision alone does
  not produce output mutation sensitivity.
- **Rethink contact supervision** — per-causal-residue, not marginal-to-set (v18b hurt accuracy).
- **v16-LGO** run to deconfound canonical-frame benefit vs retrieval leakage.
- **Gate-head / placement** fix — the remaining ~0.43 deployable registration cost.

**Key new files:** `pwm_head_v18.py`, `build_recognition_prior.py`, `eval_trimmed_core.py`,
`eval_canonical_registration.py`, `eval_full_metrics.py`, `attn_v18.py`, `viz_attn_v18.py`,
`train_v18a.sbatch`, `train_v18b.sbatch`. Checkpoints: `deeppbs_v18a_attnrepair`,
`deeppbs_v18b_contact` (training). Memory: `v18-and-fair-eval.md`.
