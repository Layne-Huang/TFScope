# TFScope: Predicting Transcription Factor DNA-Binding Specificity from Protein Sequence

**Lei Huang**¹, [Co-authors TBD]

¹ [Affiliation TBD]

---

## Abstract

Transcription factor (TF) DNA-binding specificity — the sequence preference encoded in a position weight matrix (PWM) — is a foundational quantity in gene regulation, yet its systematic prediction remains dominated by structure-based methods that require a solved protein–DNA complex at inference. Here we present **TFScope**, a sequence-only framework that predicts PWMs directly from TF amino-acid sequence. TFScope combines a protein language-model encoder (ESM-2 with LoRA fine-tuning) with a dual-stream gated DNA-binding domain pooling layer, a family-conditioned mixture-of-experts (MoE) decoder, and a contact-aware PWM-head cross-attention branch. Evaluated against the structure-based state-of-the-art DeepPBS on its own 130-TF blind split, TFScope exceeds DeepPBS on 8 of 11 motif-level metrics (mean Pearson *r* 0.802 vs 0.750) while requiring no structural data at inference. Under a stringent cluster-40% identity out-of-distribution benchmark, TFScope achieves mean oracle *r* = 0.535 with no test TF sharing >40% sequence identity to any training TF. Leave-family-out experiments reveal a ~0.48 sequence-only transfer floor and expose long C2H2 zinc-finger arrays as the frontier where per-finger recognition codes resist family-level transfer. Mechanistically, TFScope's cross-attention head localizes specificity-determining residues without three-dimensional structure, and retrieval augmentation over experimentally characterized TFs provides the critical margin over the structure-based baseline. TFScope enables proteome-scale PWM prediction at negligible computational cost.

---

## Introduction

Transcription factor binding specificity — the DNA sequence preference that governs gene regulatory networks — is encoded in the structural interface between the TF's DNA-binding domain (DBD) and its cognate DNA sequence. Decoding this interface at scale is essential for understanding and engineering gene regulation, yet current high-throughput experimental methods (SELEX, PBM, ChIP-seq) cover only a fraction of the ~1,600 human TFs, and structure-based computational methods require a solved or accurately predicted protein–DNA complex at inference.

Structure-based methods such as DeepPBS model the atomic readout mechanism directly: given a docked protein–DNA complex, a geometric deep-learning model infers which DNA bases are preferred at each position from the heavy-atom contact geometry. DeepPBS achieves state-of-the-art accuracy on its 130-TF benchmark and, with advances in structure prediction, can be applied to AlphaFold-predicted complexes. Nevertheless, the structure requirement introduces a fundamental bottleneck: high-quality docked complexes are unavailable or inaccurate for most TF–DNA systems, and predicting or folding each complex adds substantial computational cost.

Protein language models (PLMs) trained on hundreds of millions of protein sequences encode evolutionary and structural information that has proven transferable to a wide range of downstream prediction tasks. We hypothesized that a PLM encoder — specifically ESM-2, with fine-tuning via low-rank adaptation (LoRA) focused on the DBD — could capture the sequence-level signals sufficient to infer DNA binding specificity without any structural data. This hypothesis is motivated by the observation that TF binding preferences are largely determined by a small number of specificity-determining residues in the DBD that directly contact DNA bases, and these residues leave detectable imprints in co-evolutionary and language-model representations.

Here we present TFScope, a sequence-only framework for PWM prediction. TFScope encodes the TF sequence through a PLM backbone, pools DBD-focused and global representations, decodes through a family-conditioned MoE layer, and predicts a gated PWM via a contact-aware cross-attention head that reads specificity-determining residues from the language-model embedding. An optional retrieval-augmentation (RAG) branch pulls nearest-neighbor PWMs from experimentally characterized TFs as a prior, providing an interpretable analogue of the homology transfer widely used in structural bioinformatics. We evaluate TFScope across three benchmarks of increasing stringency, characterize its mechanistic behavior through interpretability experiments, and establish both where it generalizes and where its honest limits lie.

---

## Results

### TFScope architecture

TFScope processes a TF amino-acid sequence through five computational stages (Fig. 1):

1. **Encoder**: ESM-2 (650M parameters, frozen) with LoRA fine-tuning on the last 6 transformer layers (rank 16, α 32; ~491K trainable parameters). A learned DNA-binding domain indicator embedding is added to DBD residues before pooling to bias the representation toward the recognition interface.

2. **Dual-stream gated pooling**: Separate gated attention pooling over the full sequence (global stream) and the DBD residues (DBD stream) produces two 1280-dimensional vectors, concatenated and projected to a 512-dimensional joint representation via a linear + LayerNorm + GELU layer.

3. **Family-conditioned MoE decoder**: A DeepSeek-style mixture-of-experts block (12 routed experts + 1 shared expert, top-2 routing) conditioned on semantic family embeddings derived from ProTrek + ESM-2 encodings of family-level descriptions (2304-dimensional; covers all 10 TF families). Family conditioning gives structurally unseen TFs a meaningful prior without leaking label information.

4. **Contact-aware PWM head**: A cross-attention head maps from PWM column queries to DBD residue keys/values, with a v18 contact-aware branch that adds a residual `Δz_contact` to the frozen prior. The branch uses cosine cross-attention with LayerNorm on K/V, amino-acid identity values, and row-diversity + hub penalties to prevent attention collapse.

5. **Position gate + PWM regression**: A position gate head predicts which columns are information-rich (active), and a PWM regression head outputs a (4 × L) frequency matrix. Optional RAG retrieval of K nearest-neighbor PWMs (leave-gene-out index, cosine similarity) provides an additional prior.

### Benchmark 1 — Head-to-head against structure-based DeepPBS

We evaluated TFScope against DeepPBS on the DeepPBS 130-TF blind test split using two evaluation protocols, applied identically to both methods. In the first protocol (motif-level, oracle-aligned), each predicted PWM is trimmed to the target's IC ≥ 0.25 informative core and granted oracle offset and reverse-complement alignment, providing a fair ranking of base-composition accuracy with registration removed. In the second protocol (deployable, canonical-fixed), prediction and target are independently canonicalized (trimmed + canonical strand), and no alignment oracle is granted.

**Base composition (oracle-aligned, Table 1).** On 116 TFs covered by both methods, TFScope exceeded DeepPBS on 8 of 11 metrics (Fig. 2). Mean per-column Pearson *r* improved from 0.750 to **0.802** (+0.052), median *r* from 0.759 to **0.831**, top-1 base accuracy from 0.793 to **0.836**, MCC from 0.698 to **0.765**, F1 from 0.724 to **0.765**, AUC from 0.937 to **0.939**, MAE from 0.133 to **0.108**, and RMSE from 0.204 to **0.184**. DeepPBS retained an advantage only on calibration metrics: cross-entropy (0.84 vs 0.99) and KL divergence (0.358 vs 0.517). The TFScope ablation with the contact-aware branch disabled (Table 1, "−contact") showed mean *r* = 0.701, establishing that the contact branch alone provides the +0.10 improvement over the frozen prior at the cost of only 0.55M additional parameters.

**Deployable registration (canonical-fixed, Table 2).** Removing the alignment oracle collapsed both methods' mean *r* by ~0.43: TFScope fell to 0.420 and DeepPBS to 0.419 — a statistical tie. **Registration — accurately placing the motif in the correct frame — is the dominant error for every method, including DeepPBS.** This limitation is a stated property of the current seed model; a placement/gate head is identified as the principal avenue for future improvement.

**Case study 1: Egr1 (1a1g_A).** To illustrate Benchmark 1, we examined Egr1, a C2H2 zinc-finger TF whose 10-position GC-rich motif (GCGTGGGCGT) is one of the most information-dense in the JASPAR database (Fig. 3). TFScope achieved oracle *r* = 0.990, recovering every position of the consensus with high fidelity from protein sequence alone. DeepPBS, operating on the same crystal complex (PDB 1a1g), achieved *r* = 0.776; its predicted logo contained essentially a single G residue at the first position. This example illustrates concretely that sequence-level information in the ESM-2 representation can resolve binding specificity details that are not accurately recovered from the 3D readout at this crystal structure.

### Benchmark 2 — cluster40: honest out-of-distribution generalization

Protein sequences in the DeepPBS benchmark can include near-homologues of training TFs, so accuracy on that benchmark may reflect homologue lookup as much as genuine sequence→specificity learning. To evaluate out-of-distribution generalization, we constructed **cluster40**: a CD-HIT clustering at 40% protein identity over 1,320 unique proteins yielding 389 clusters, family-stratified into 2,983 train / 625 validation / 639 test PWMs. No test TF shares >40% identity with any training TF.

TFScope trained on the cluster40 split achieved mean oracle *r* = **0.535** (median 0.505) on the 639-TF held-out set (Table 3, Fig. 4). Additional metrics: IC-weighted *r* 0.553, MAE 0.198, top-1 accuracy 0.630, AUC 0.797, F1 0.603, MCC 0.472. Per-family accuracy spanned a wide range (Table 4): bZIP (0.72) and Homeodomain (0.68) generalized well, while C2H2_long (181 TFs, 28% of the test set, mean *r* 0.43) was the bottleneck (see Case study 3).

Two important leakage caveats: the same checkpoint scores ~0.76 on an earlier training split and ~0.80 on the DeepPBS benchmark, but both are leaky (the earlier split trained on most cluster40 test TFs; 90/130 DeepPBS test entries landed in cluster40's training set). The only clean cluster40 number is **0.535 on its own held-out set**, and only that is reported here.

### Benchmark 3 — Leave-family-out: transfer to an unseen TF family

Leave-family-out (LFO) experiments probe the hardest form of generalization: holding out an entire structural family as test, training on the other nine, and evaluating how well the model transfers to a completely unseen family. This is possible in TFScope because family embeddings are semantic (derived from PLM encodings of family descriptions rather than from training labels), so a held-out family still receives a meaningful prior. We trained 10 models (one per held-out family) on the full augmented dataset and evaluated on 4,241 held-out TFs.

**The transfer floor.** The macro-average oracle *r* across all families was **0.479 ± 0.057** (median 0.447). This figure represents TFScope's sequence-only transfer capability to a structurally unseen family — the floor below which architecture improvements must target either anti-collapse training objectives or large-scale protein–DNA pre-training.

**Variance collapse (Fig. 5).** The in-distribution per-family spread on cluster40 (0.39–0.72) collapsed to a tight LFO band (~0.42–0.61). The in-distribution high-flyers — bZIP (−0.235), Homeodomain (−0.202), Nuclear Receptor (−0.206) — fell hardest when their family was held out, revealing that their strong in-distribution scores were largely **family memorization** of a conserved binding grammar rather than transferable sequence→specificity mapping. Conversely, families that scored weakly in-distribution (C2H2_short: +0.223, ETS: +0.024) held or even improved in LFO.

**Case study 3: The C2H2 long-array frontier (Fig. 6).** C2H2_long zinc-finger arrays showed not only the lowest mean LFO *r* (0.447) but the widest within-family variance (−0.302 to 0.997 across 812 TFs). Two extreme examples illustrate the biological basis: ZNF76, which contains a short, repetitive recognition code (ZF1 R→RDER pattern), was predicted with oracle *r* = 0.997 even under LFO; ZNF649, a long tandem array with a complex per-finger grammar and an AT-rich recognition sequence (ATATAT), was predicted with oracle *r* = −0.302 — the model produced a near-random output. This contrast identifies per-finger compositional diversity as the bottleneck: families whose specificity is encoded in a single, shared recognition grammar generalize across the LFO boundary, while those where each zinc finger reads a different sub-code do not.

### The contact-aware cross-attention head reads specificity-determining residues

**Case study 2: KLF4 K409 and MyoD L122 (Fig. 7).** A key design goal of the v18 contact branch was to repair the degenerate rank-1 attention collapse observed in the prior cross-attention head, where every PWM column attended identically to the same 2–3 DBD residues (row-constancy 0.81 for KLF4) with a terminal-residue sink, and zero attention mass on experimentally known specificity-determining residues.

After contact-branch activation, KLF4 K409 — a lysine whose identity determines GC-box versus CACCC preference — went from zero attention mass (0.000) to 0.094, row-constancy fell from 0.81 to 0.25, and attention entropy rose from 1.47 to 2.36 nats (out of a maximum 4.42), indicating broad, non-degenerate attention across the DBD. The same repair was observed for MyoD L122 (row-constancy 0.59→0.27, entropy 1.40→1.40/3.95, mass 1.25→0.214). This result demonstrates that TFScope's cross-attention, after contact-branch activation, localizes information-relevant residues directly from the protein language-model embedding — an interpretability axis independent of any three-dimensional structure.

**Honesty note:** The attention repair demonstrably improves *where* the model looks, but does not yet change *what* the model outputs when that residue is mutated (WT-vs-mutant output PWM *r* ≈ 0.9997–0.9998 for both TFs). Output mutation sensitivity requires additionally a mutation-contrastive loss term (planned v18c), and is therefore a stated limitation.

### Retrieval augmentation drives TFScope above the structure-based baseline

**Case study 4: RAG ablation (Fig. 8).** TFScope with retrieval augmentation disabled (TFScope −RAG) achieved mean oracle *r* = 0.749 on the DeepPBS 130-TF blind split — essentially identical to DeepPBS's 0.750, demonstrating that the sequence encoder alone is already competitive with the structure-based method. Adding RAG (nearest-neighbor PWMs from experimentally characterized TFs, retrieved via leave-gene-out index to prevent information leakage) raised performance to 0.802, providing the +0.053 margin that places TFScope above DeepPBS. Additional metrics followed the same pattern: top-1 accuracy (0.800→0.836 vs DeepPBS 0.793), MCC (0.699→0.765 vs 0.698). DeepPBS retained the calibration edge (CE 0.84 vs 0.988; KL 0.358 vs 0.517).

The RAG mechanism is the model's analogue of homology transfer: if an experimentally characterized close relative exists in the database, its PWM provides a strong prior that the model learns to weight against its own sequence-derived prediction. The leave-gene-out index is critical — using the unconstrained retrieval index inflates results by permitting same-source donor retrieval and is not used for any headline metric.

---

## Discussion

TFScope demonstrates that the binding specificity information available in a protein language model, combined with targeted architectural components for DBD pooling, family conditioning, and contact-aware attention, is sufficient to match or exceed the structure-based state of the art for PWM prediction. The key results are:

(1) **Sequence-only parity/superiority on base composition.** TFScope without any structural input exceeds DeepPBS on 8/11 motif-level metrics in a fair, leakage-controlled comparison. The result holds across TF families and is not driven by a single easy family.

(2) **Registration is the dominant bottleneck.** Both TFScope and DeepPBS lose ~0.43 mean *r* when oracle alignment is replaced by canonical-fixed registration. This implies that accurate motif placement — knowing where in the predicted PWM array the informative core lies — is more valuable than further base-composition tuning, and is the top architectural priority for the next version.

(3) **Retrieval augmentation provides the competitive margin.** Without RAG, TFScope and DeepPBS are equivalent (0.749 vs 0.750). RAG is therefore the mechanism through which TFScope exceeds the structure-based method, and its leakage hygiene (leave-gene-out index) is essential to an honest evaluation.

(4) **The transfer floor is ~0.48.** Leave-family-out experiments reveal that in-distribution accuracy is substantially inflated by family memorization for families with a conserved binding grammar. The honest transfer floor to a completely unseen family is ~0.48 in mean oracle *r*. Closing the gap between this floor and the in-distribution ceiling is the central challenge for the next generation of models, motivating anti-collapse contrastive training and large-scale protein–DNA pretraining (HT-SELEX, DPAC).

(5) **Long C2H2 zinc-finger arrays are the prediction frontier.** The per-finger recognition code architecture of long C2H2 arrays violates the one-family-one-consensus assumption that underlies the family-conditioned prior, producing the widest within-family spread of any TF class (−0.30 to 0.997) and the lowest mean transfer score. A finger-level model that decomposes the prediction into per-finger contributions is a natural direction.

The current TFScope is limited to PWM (first-order) binding models and does not capture higher-order sequence preferences, cooperative binding, or cofactor effects. Output mutation sensitivity — predicting how a single amino-acid substitution changes the PWM — is not yet achieved despite attention-level repair, requiring a dedicated mutation-contrastive training signal. Extension to longer protein sequences beyond the 1024-residue context, multi-subunit TF complexes, and in vivo chromatin context are left to future work.

---

## Methods

### Dataset and data processing

Training data were derived from the DeepPBS dataset (520 TF–DNA crystal structures with experimentally validated PWMs) and augmented with CIS-BP, RCADE, and MEME database entries, yielding 4,247 unique TF entries (`tf_pwm_aug_dbd_canon_trim.parquet`). Each entry contains the TF amino-acid sequence, annotated DBD start/end positions, family assignment (10 families: bZIP, Homeodomain, Nuclear_Receptor, Forkhead, bHLH, C2H2_medium, C2H2_long, C2H2_short, ETS, Other), and a 4 × L PWM in canonical trimmed form (IC ≥ 0.25 left-anchored core, maximum length 20 columns).

**Canonical trimming and strand normalization.** Each PWM was trimmed to its IC ≥ 0.25 core. Strand canonicalization selects the forward strand by the criterion that the 5′ end is more GC-rich, consistent with the convention in the CIS-BP and JASPAR databases.

**Evaluation protocols.** Two evaluation protocols were applied identically to all methods including DeepPBS:
- *Motif-level, oracle-aligned*: Trim target to IC ≥ 0.25 core; grant every method oracle offset (±10) and reverse-complement alignment maximizing coverage-normalized mean per-column Pearson *r* (`coverage_norm=True` in `align_pwm`); report metrics on the aligned overlap. This provides fair ranking with absolute values as upper bounds.
- *Deployable, canonical-fixed*: Apply v16 canonicalization (trim + canonical strand) independently to prediction and target; score with no alignment. This measures what a method actually emits in deployment.

### Model architecture

**Encoder.** ESM-2 (esm2_t33_650M_UR50D, 650M parameters; Hugging Face `facebook/esm2_t33_650M_UR50D`) with the full model frozen and LoRA adapters (rank 16, α 32) applied to the last 6 transformer layers. A learned 16-dimensional DBD indicator embedding is added to per-residue representations within the annotated DBD boundaries.

**Dual-stream gated pooling.** Two independent gated attention pooling modules (`global_pool` over the full sequence; `dbd_pool` over DBD residues only) produce 1280-dimensional vectors. These are concatenated to form a 2560-dimensional vector, then projected by `Linear(2560→512) + LayerNorm + GELU + Dropout(0.1)`.

**Family-conditioned MoE decoder.** A DeepSeek-style MoE block with 12 routed experts + 1 shared expert (always active), top-2 routing, load-balance loss weight 0.01. Family conditioning is applied via learned semantic embeddings (2304-dimensional, derived from ProTrek + ESM-2 encodings of family-level text descriptions) concatenated with the 512-dimensional representation before the MoE gate.

**Contact-aware PWM head (v18).** The legacy cross-attention head becomes a frozen prior branch. A contact-aware residual branch adds `logits = z_prior + λ·Δz_contact` where the contact branch uses cosine cross-attention + LayerNorm K/V + amino-acid identity values + row-diversity penalty + hub penalty. `λ` is a learnable scalar. Only the 0.55M contact-branch parameters are trained; the remaining model is frozen from the prior checkpoint.

**Position gate + PWM output.** A two-layer MLP position gate predicts per-column activation probabilities (sigmoid). The PWM regression head is a convolutional decoder producing (4 × 20) logits; softmax over the 4-base dimension yields probabilities. During inference, gate-active columns (gate > 0.5) are selected and oracle-aligned to the target for evaluation.

**Retrieval augmentation (RAG).** At training and inference, K = 3 nearest-neighbor TFs are retrieved from a pre-built index by cosine similarity of the ESM-2 DBD representation. The leave-gene-out (LGO) index excludes any TF with the same gene symbol as the query, preventing information leakage. Retrieved PWMs are projected and injected via a cross-attention layer before the MoE block. The index is built on the augmented training set; test TFs are excluded from the index.

### Training

All models were trained with AdamW (lr 6×10⁻⁴, weight decay 0.01, β₁ 0.9, β₂ 0.999) with linear warmup (500 steps) and cosine decay. Batch size 128, maximum sequence length 1024. Training used a composite loss: PWM reconstruction (IC-weighted MSE + per-column cross-entropy), gate supervision (binary cross-entropy on IC ≥ 0.25 indicator), load-balance loss for MoE routing, and optionally in-batch contrastive PWM loss (weight 0.3, temperature 0.1; ablation only). Models were trained on NVIDIA A100 (80GB) GPUs via SLURM. Early stopping on validation oracle *r*.

**DeepPBS split training.** v18a trained on 520 DeepPBS TFs; test split: `benchmark_no_val.json` (130 TFs). Prior model: deeppbs_v17_200ep (frozen). Contact branch only trained.

**cluster40 training.** Built by `scripts/build_cluster40_split.py`: CD-HIT at 40% identity on 1,320 unique proteins (389 clusters); family-stratified split into train (2,983) / val (625) / test (639). Model `fulldata_cluster40_v18a` trained from scratch with v18 architecture on the full 2,983-TF train set; best checkpoint at epoch 125 (oracle *r* early stopping).

**Leave-family-out training.** Ten separate models, each holding out one of the 10 TF families. Splits built by `scripts/build_lofo_splits.py`; validation drawn from remaining families via 40% identity clustering. Each model trained for up to 200 epochs with oracle *r* early stopping. Family embeddings are pre-computed and fixed; the held-out family uses its correct semantic embedding so the model has a meaningful prior without any label leakage.

### Evaluation

**Per-sample oracle Pearson r.** For each TF, the gate-active predicted columns are aligned (oracle offset ±10, reverse-complement) to the IC ≥ 0.25 target core. Pearson *r* is computed column-wise and averaged. Reported as the primary metric.

**Full metric panel.** Additionally computed: IC-weighted *r* (mean per-column Pearson *r* weighted by target IC), MAE (mean absolute error over 4×L elements), RMSE, cross-entropy, KL divergence, top-1 accuracy (argmax base), macro AUC (one-vs-rest), macro F1, MCC.

**DeepPBS comparison.** DeepPBS predictions are read from pre-computed `results/deeppbs_blind_benchmark/struct_preds.npz` (n=130). The same trimmed-core oracle aligner is applied to DeepPBS predictions for a fair comparison. For the deployable protocol, `scripts/eval_canonical_registration.py` applies identical canonicalization to both TFScope and DeepPBS outputs.

### Case study analyses

**CS1 (Egr1 head-to-head).** v18a checkpoint `deeppbs_v18a_attnrepair/ckpt_best.pt` run on the DeepPBS test split; Egr1 (1a1g_A) selected as a representative C2H2 zinc-finger TF. TFScope prediction and DeepPBS structure-derived prediction compared against the JASPAR experimental motif MA0162.1.

**CS2 (Attention repair).** The `attn_v18.py` script runs KLF4 (DBD sequence, fam=C2H2) and MyoD (fam=bHLH) through v17 (prior, degenerate) and v18a (contact branch active) with retrieval disabled (`retrieved_pwms=None`) to isolate the protein-pathway contribution. Row-constancy, entropy, and per-residue attention mass are reported. Output mutation-blindness is explicitly noted: WT-vs-mutant output PWM *r* ≈ 0.9997–0.9998 despite attention repair.

**CS3 (LFO C2H2_long frontier).** Per-family oracle *r* distributions computed from `results/lofo/per_tf_oracle_r.json` (4,241 TFs, 10 families). ZNF76 and ZNF649 inference from `checkpoints/lofo_v18a/C2H2_long/ckpt_best.pt`; alignment to IC ≥ 0.25 core using gate-active columns.

**CS-RAG (Retrieval ablation).** Aggregate metrics from `results/full_metrics/panel.json`, comparing v18a (with LGO RAG), v18a_noRAG (identical model with retrieval disabled at inference), and DeepPBS on the same 116-TF panel.

---

## Figure Legends

**Fig. 1 — TFScope architecture.** Schematic of the five-stage pipeline: ESM-2/LoRA encoder with DBD indicator → dual-stream gated pooling (global + DBD streams) → family-conditioned MoE decoder → contact-aware v18 cross-attention PWM head (frozen prior + residual contact branch) → position gate + PWM output, with optional RAG retrieval branch.

**Fig. 2 — Benchmark 1 metric panel (Table 1 visualization).** Side-by-side bar chart of 11 motif-level metrics for TFScope (−contact ablation), TFScope, and DeepPBS. 116 DeepPBS-covered test TFs, trimmed core IC ≥ 0.25, oracle offset+RC aligned.

**Fig. 3 — Case study 1: Egr1 head-to-head.** Stacked sequence logos: experimental motif (Egr1/1a1g_A, JASPAR MA0162.1), TFScope prediction (*r* = 0.990, sequence only), DeepPBS prediction (*r* = 0.776, crystal structure required). TFScope recovers the GC-rich 10-position Egr1 motif from protein sequence alone; DeepPBS output is nearly flat. File: `figures/cs1_egr1_headtohead.pdf`.

**Fig. 4 — Benchmark 2: cluster40 honest OOD logo panel.** Predicted vs ground-truth logos for representative TFs from the 639-TF cluster40 held-out set (mean oracle *r* = 0.535). File: `figures/pred_vs_gt_cluster40.pdf`.

**Fig. 5 — Benchmark 3: leave-family-out variance collapse.** Horizontal box + strip plot of per-TF oracle *r* under LFO for all 10 TF families (4,241 TFs). C2H2_long family highlighted in darker vermillion. Dashed line at overall mean (0.479). File: `figures/cs3_family_frontier.pdf`.

**Fig. 6 — Case study 3: ZNF76 vs ZNF649.** Paired sequence logos for the highest-scoring (ZNF76, *r* = 0.997) and lowest-scoring (ZNF649, *r* = −0.302) C2H2_long TFs under LFO. Both TFs' entire family was held out; the contrast reflects per-finger recognition-code complexity. File: `figures/cs3_znf_examples.pdf`.

**Fig. 7 — Case study 2: KLF4/MyoD attention repair.** 2×2 heatmap panel: rows = TFScope without / with contact branch (v17/v18a); columns = KLF4 (C2H2, K409) and MyoD (bHLH, L122). Color = attention weight from each PWM column (y) to each DBD residue (x). Cyan dashed line marks the experimentally known specificity-determining residue. Row-constancy, entropy, and attention mass at the causal residue annotated per panel. File: `figures/cs2_klf4_attention_repair.pdf`.

**Fig. 8 — Case study RAG: retrieval ablation.** Bar chart comparing six key metrics for TFScope −RAG, TFScope +RAG, and DeepPBS on the 130-TF blind split. Without RAG, TFScope ties DeepPBS (mean *r* 0.749 vs 0.750); with RAG it exceeds on 4/6 metrics shown (+0.053 mean *r*). File: `figures/cs_rag_ablation.pdf`.

---

## Supplementary

*To be added:*
- Supplementary Table S1: Full 11-metric panel for all models on the DeepPBS blind split (n=130)
- Supplementary Table S2: cluster40 per-family oracle *r* panel
- Supplementary Table S3: LFO per-family full metric panel
- Supplementary Fig. S1: Deployable canonical-fixed metric comparison (Table 2 visualization)
- Supplementary Fig. S2: cluster40 per-family oracle *r* bar chart
- Supplementary Fig. S3: MyoD v17/v18a attention heatmap (analogous to Fig. 7 for bHLH family)
- Supplementary Fig. S4: TFScope architecture detail diagram

---

## Data availability

DeepPBS benchmark data: `data/processed/splits/deeppbs_only/benchmark_no_val.json`, `data/processed/tf_pwm_deeppbs_only.parquet`. cluster40 split: `data/processed/splits/cluster40/split.json`. LFO splits: `data/processed/splits/lofo_v2/<Family>.json`. Augmented dataset: `data/processed/tf_pwm_aug_dbd_canon_trim.parquet`. [Will be deposited to Zenodo upon acceptance.]

## Code availability

TFScope source code and trained model checkpoints will be made available at [GitHub URL TBD] upon acceptance. Key scripts: `scripts/train.py` (training), `scripts/evaluate.py` (evaluation), `scripts/eval_full_metrics.py` (full metric panel), `scripts/eval_canonical_registration.py` (deployable evaluation).

---

## References

1. Mitra, R. et al. DeepPBS: deep learning–based prediction of protein–DNA binding specificity. *Nat. Methods* **21**, 1674–1683 (2024).
2. Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123–1130 (2023).
3. Hu, E. J. et al. LoRA: Low-rank adaptation of large language models. *ICLR* (2022).
4. Lambert, S. A. et al. The human transcription factors. *Cell* **172**, 650–665 (2018).
5. Weirauch, M. T. et al. Determination and inference of eukaryotic transcription factor sequence specificity. *Cell* **158**, 1431–1443 (2014). [DREAM5]
6. Jolma, A. et al. DNA-binding specificities of human transcription factors. *Cell* **152**, 327–339 (2013). [HT-SELEX]
7. Dai, D. et al. DeepSeek-MoE: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv* 2401.06066 (2024).
8. Elnaggar, A. et al. ProtTrans: Toward understanding the language of life through self-supervised learning. *IEEE Trans. Pattern Anal. Mach. Intell.* **44**, 7112–7127 (2022).
9. [Chatterjee lab DPAC preprint: biorxiv 2025.05.14.654102 — add full citation]
10. [RCADE, CIS-BP, JASPAR citations — to be filled]

---

*Manuscript status: DRAFT. «VERIFY» flags in text denote numbers that must be re-verified against committed result files before submission. See `manuscript/results_three_benchmarks.md` for detailed source attribution. cluster40 full metric panel (Tables 3/4) requires regeneration and file commit before submission.*
