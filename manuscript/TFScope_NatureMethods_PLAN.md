# TFScope — Nature Methods Manuscript Plan

*Planning doc. Status tags: **[DONE]** · **[PARTIAL]** · **[LACKING]** (protocol given inline). Results headers are written in Nature Methods style — short declarative sentences stating the finding; Methods use noun-phrase headers.*

**Working title:** *Structure-free prediction of transcription-factor DNA-binding specificity by distilling structural recognition into a sequence model.*

**One-line contribution:** TFScope predicts TF binding specificity (PWM) from protein sequence alone — competitive with structure-based methods — by distilling structural recognition-residue contacts into a sequence model, enabling specificity for the majority of TFs that lack structures.

**Figures (4 multi-panel, Nature Methods style):**
- **Fig 1** — Structure-free prediction matches structure *(method + benchmark + baselines + per-family + AF3 consensus-folding plausibility).*
- **Fig 2** — Contact distillation drives accuracy and yields a learned recognition code *(contact ablation + nulls + mutagenesis + Wetzel code + AF3+Rosetta contact discovery + prototypes).*
- **Fig 3** — Proteome-scale specificity for orphan TFs *(held-out recovery + orphan logos + CRE enrichment + evolution/design).*
- **Fig 4** — Mutation effects via a sequence→structure hybrid *(MyoD1 + KLF4 + hybrid schematic).*

*The family-conditioning / motif-incoherence exploration is **not** a main-text result (unresolved); it is reduced to one Discussion sentence + an optional supplementary note.*

---

# INTRODUCTION

*(Nature Methods Introductions run as continuous prose — no subsections. Paragraph plan:)*

- **¶1** TF specificity, PWMs, and why most TFs lack experimental motifs. **[DONE]**
- **¶2** Structure-based prediction (DeepPBS) is accurate but needs a protein–DNA structure most TFs don't have; the coverage problem. **[DONE]**
- **¶3** Prior sequence methods (DeepBind, ProBound) learn per-TF from binding data; open gap = a sequence model that *generalizes specificity across TFs and implicitly uses structural recognition without a structure at inference*. **[DONE]**
- **¶4** We present TFScope; preview the four claims (competitive benchmark, contact distillation as the lever + recovery of the recognition code, proteome-scale orphan application, a sequence→structure hybrid for mutations). **[DONE on results-final]**

---

# RESULTS

## **TFScope predicts DNA-binding specificity from protein sequence**  → Fig 1a

Architecture/overview: ESM2/LoRA encoder → gated DBD pooling → family-conditioned MoE → cross-attention PWM head + gate head; trained with recognition-residue **contact supervision** (structure-distilled, train-only; inference is sequence-only). **[DONE]** (model). Figure schematic **[LACKING]** — build with `scientific-schematics`.

## **Sequence-only prediction matches structure-based methods on a contamination-controlled benchmark**  → Fig 1b–c

The cluster40 (40%-DBD-identity) split and the contamination finding (structure-level split inflates both ~0.25); `combined` **oracle-r 0.643 vs DeepPBS 0.631**, plus panel-r/MAE(0–2)/top-1 — DeepPBS reported on every metric. **[DONE].**

### **A baseline ladder isolates the architecture's contribution**  → Fig 1d

**[DONE].** cluster40 test, same oracle-aligned metric. Ladder (oracle-r / top-1 / MAE): **random** 0.417 / 0.552 / 0.994 (floor, à la DeepPBS); **NN-PWM** 0.518 / 0.612 / 0.824 (leave-gene-out nearest DBD by ESM2-cosine, copy donor PWM); **ESM2-linear** 0.579 / 0.644 / 0.832 (frozen ESM2 mean→MLP→PWM — the "is the architecture worth it?" test); **DeepPBS** 0.634 / 0.704 / 0.657; **TFScope** 0.643 / 0.714 / 0.651. TFScope−DeepPBS p=0.74 (matches); TFScope−ESM2-linear +0.064 (architecture justified). Paired bootstrap + Wilcoxon. `results/baseline_ladder/ladder.json`.

### **Accuracy varies systematically by protein family**  → Fig 1e

**[DONE].** Per-family oracle-r, TFScope vs DeepPBS (`results/per_family/`): TFScope wins Other +0.071 / NR +0.083 / bHLH +0.074; DeepPBS wins bZIP −0.357 / C2H2_long −0.182 — sequence-only loses exactly on the most motif-incoherent / quaternary-recognition families. Honest and mechanistically self-consistent.

### **Predicted consensus oligos fold into high-confidence protein–DNA complexes**  → Fig 1f

**[PARTIAL — folds running on other machines].** For all 84 cluster40 test TFs, fold (AF3) the TF with its **TFScope-predicted** highest-IC consensus oligo and, separately, with its **DeepPBS-predicted** consensus (length-matched, correct oligomer state; inputs in `manuscript/AF3_consensus_folding_inputs.md`). Compare **ipTM** and interface-**pLDDT** distributions TFScope vs DeepPBS (paired, per-TF). Read-out: do sequence-only predictions yield protein–DNA complexes as physically confident as the structure-based method's? — an orthogonal, structure-based validation of the benchmark that does not depend on the oracle-aligned PWM metric. *No GT fold needed (experimental structures exist for the test set).* Analyze ipTM/pLDDT when folds return.

### **Predictions generalize to held-out protein families**  → Fig 1 (added when done)

**[LACKING — deferred ~12–20 h].** Leave-family-out (train all-but-X, test X) for 5 families (Homeodomain, merged-C2H2, bZIP, NR, bHLH); reuse `tf_nn_index_lofo_`* + `eval_lofo.py`; per-family + mean; paired-bootstrap significance on the combined-vs-DeepPBS margin.

## **Distilling structural contacts drives accuracy and recovers the protein–DNA recognition code**  → Fig 2  *(core contribution)*

### **Recognition-residue contact supervision is the principal accuracy lever**  → Fig 2a

**[DONE]** — figure `results/baseline_ladder/fig2a_ablation.{png,pdf}` (single panel, oracle-aligned r). Built up from the **same training records DeepPBS uses**, two synergistic levers carry the sequence-only model past structure: (i) a model trained on the **DeepPBS training set alone** (467 records, sequence-only) reaches oracle-r **0.548** — below structure-based DeepPBS (0.631); (ii) **corpus augmentation** to the full 4,250-record corpus lifts it to **0.596 (+0.05)**; (iii) **distilling recognition-residue contacts** (1D, w=0.3, train-only) lifts it to **0.657 (+0.06)**, now *above* DeepPBS. Neither lever alone suffices — augmentation closes most of the data gap, contact distillation supplies the structural-recognition signal that PWM labels lack, and together they synergise (each ≈ +0.05–0.06).

*Mechanism (composition check, `results/baseline_ladder/extended_corpus_composition.json`):* augmentation adds breadth (4,250 vs 467 records) but no new families on top of the already-broad corpus, so once the corpus is large the marginal PWM record saturates; the residual gap to structure is **recognition geometry**, which PWM labels do not contain and contact distillation injects — consistent with the contamination sweep (0.627@90% → 0.548@40%), where accuracy is gap-limited not count-limited. *(w=0.3 1D-recognition optimal vs w=1.0 / full-2D; retrieval and temperature scaling did not help — stated in text, not plotted.)*

### **In silico mutagenesis identifies specificity-determining residues**  → Fig 2b

**[DONE]** — figure `figures/figure2b_mutagenesis/`; data `results/per_family/alascan_population.json`. Single-residue Ala scan over each DBD → Δ(predicted PWM, L1 over the 4×20 motif) → per-residue importance track, run across the **whole cluster40 test set**. **Ground truth = DNA contacts extracted from each test crystal structure** (`scripts/extract_pdb_contacts.py`: protein side-chain+CA atoms within 4.5 Å of a DNA base atom — a *geometric* (distance) definition that captures **all interaction types**: H-bonds, hydrophobic, salt bridges, π-stacking; not restricted to H-bonds; 78/84 structures, exact sequence match). **Structure-derived and training-independent** — the model never sees these labels at train (contact supervision used a separate rule-based prior, train-only) or inference, so the agreement is genuine generalization. (DSSR nt:aa H-bonds cross-validate this set: H-bonds ⊆ geometric.)

**Population result:** per-TF AUROC (contacts vs *other DBD* residues, ±15 hard-negative window) **median 0.65, 75% of TFs above chance** (n=78); pooled, contacting residues are **2.3× more important** than other DBD positions (Mann–Whitney p≈2×10⁻⁴⁶). Monomer-only robustness (excl. obligate dimers bHLH/bZIP/NR) holds (median ≈0.59) — not propped up by easy basic regions.

Two illustrative tracks overlay **three channels** — DNA contacts (red bars, all forces), **TFScope's top-20 important residues (blue ▼)**, other DBD residues (grey) — using **monomeric DBDs** (single chain → single site, matching the single-sequence input): **C2H2 zinc finger** (ZBTB7A, AUROC 0.94, **top-20 recovers 5/6 contacts**) and **homeodomain** (DUX4, AUROC 0.76, **top-20 recovers 6/12**). *(We deliberately avoid bHLH/bZIP examples: they read as dimers, each monomer contacting a half-site, so a single-sequence example misrepresents the binding — see Discussion limitation.)* The sub-0.5 tail (panel c) is dominated by zinc fingers whose structural Zn-coordinating residues get flagged — motivates the recognition-code validation (Fig 2c). MyoD1 is *not* in the benchmark test set — its named vignette lives in Fig 4.

**Structural panel** (`figures/figure2b_structure/`): TFScope's per-residue importance mapped onto the 3D complex (Cα coloured by importance) with the crystal contacts ringed — the model's hottest residues sit on the DNA interface and coincide with the ringed contacts; visual proof the sequence-only model's important residues are the DNA-reading residues.

### **Residue importance agrees with experimentally derived recognition codes**  → Fig 2c

**[LACKING].** Obtain Wetzel–Zhang–Singh (Genome Res 2022) probabilistic recognition-code tables; per family, correlate TFScope residue-importance + contact-attention with the code's recognition positions (AUROC / precision at recognition residues); compare to DeepPBS RI scores on shared TFs. *Analog of DeepPBS's alanine-scanning validation; resolves the C2H2 Zn-confound by scoring against true recognition positions.*

### **Folding and energetic scanning recover atomic contacts for structure-less TFs**  → Fig 2d

**[PARTIAL]** (MyoD1 proof-of-concept done). **This is the capability emphasis: TFScope predicts recognition contacts for TFs that have no experimental structure, and AF3 folding + Rosetta verify they sit at the protein–DNA interface.** Protocol: ~10 high-confidence in-distribution structure-less TFs → AF3-fold (TF + predicted consensus, correct oligomer) → `pwm_hybrid -relax` (`PWM_INTERFACE_MODE=prot_dna`) interface-ΔΔG scan → contacts; cross-validate against the recognition code (Fig 2c). Caveat: needs a roughly-correct PWM (wrong PWM → non-specific fold).

### ~~**Learned prototypes expose interpretable binding concepts**  → Fig 2e~~  **[DROPPED]**

**[DROPPED 2026-06-22].** The MoE interpretability modules are collapsed to uniform in the
trained model — the 32-prototype dictionary activates ~equally per TF (usage 0.030–0.032 vs
uniform 0.031; family-share 0.113–0.125 vs uniform 0.125) and the 12-expert gating is fully
collapsed (gate weight 1/12 each, per-TF top-expert ≈0.084). Decoding prototypes / showing
top-activating-TF logos is therefore not interpretable (it would just display real TF motifs
under near-random attention). Accuracy comes from the cross-attention PWM head + contact
distillation, not MoE specialization. Interpretability is carried instead by Fig 2b
(mutagenesis) + Fig 2c (recognition code). Diagnostic:
`results/per_family/fig2e_prototype_collapse_diagnostic.json`, `scripts/build_fig2e_prototypes.py`.
Figure 2 is now **2a → 2b → 2c → 2d**.

**Why it collapsed (mechanism, for the rebuild):** over-determined.
(1) Two regularizers reward uniformity — `family_diversity_loss` (w=0.01) does `loss = −entropy`
of each family's mean gate distribution, i.e. *maximizes* routing entropy → forces every family
to spread across all experts (mislabeled: it penalizes specialization); and `load_balance_loss`
(Switch-style, w=0.05) evens the marginal expert usage.
(2) No task pressure pushes back: the MoE is a residual (`out = x + shared + routed + proto`),
a shared expert always fires, and the cross-attention PWM head reads ESM directly, so `moe_out`
is not an information bottleneck — uniform gating costs nothing on the main loss.
(3) Family identity already has redundant pathways (FiLM γ/β, gating semantic bias), so experts
never *need* to specialize.
(4) Prototype-specific: `softmax(q·protoᵀ·d^−0.5)` with d^−0.5≈0.044 is a high-temperature
softmax → flat by construction; `proto_out ≈ mean(prototypes)` ≈ const → ~no gradient to sharpen,
and no usage/sparsity loss rewards peaked prototype attention.

**Planned fix (AFTER current training job finishes):** to revive Fig 2e —
(i) flip `family_diversity_loss` to *minimize* within-family routing entropy (reward
specialization) or remove it; (ii) shrink the shared-expert/residual dominance so routing
matters (or add a small router-confidence/usage-entropy penalty per-token); (iii) lower the
prototype softmax temperature (drop or learn the d^−0.5 scale); (iv) add a prototype-usage
sparsity/diversity loss so prototypes are peaked and distinct. Then re-run
`scripts/build_fig2e_prototypes.py` to check specialization (share ≫ 1/n_fam) before deciding
to reinstate the panel. See [[moe-collapse-fig2e]].

## **TFScope nominates binding motifs for orphan transcription factors at proteome scale**  → Fig 3  *(the positive, sequence-only-unique story)*

### **Known motifs are recovered from sequence alone for held-out factors**  → Fig 3a

**[PARTIAL].** Systematic held-out recovery (predicted vs curated JASPAR/HOCOMOCO motif r) across families + masked same-family controls — the scalable advantage over structure-based methods.

### **Nominated motifs are enriched in cis-regulatory elements**  → Fig 3b–c

Orphan nominations (SOHLH1, ZGLP1, ADNP, ADNP2, ZHX2/3); genome-wide CRE scan with dinucleotide-shuffle composition control → 5/6 motifs cCRE-enriched; homeodomains→promoters (ADNP2 17.5×), SOHLH1 E-box→enhancers (1.4×). The composition control flips the naive GC-confounded read. **[DONE]** (`results/genome_cre_scan/`). Frame as candidate-level functional plausibility, not occupancy.

### **Model-guided optimization recovers and redesigns specificity**  → Fig 3d

**[PARTIAL]** (have `evolve_pwm_`*). DNA-side — evolve DNA to maximize predicted binding, recover consensus; protein-side — directed mutation trajectory. *Confirm DeepPBS Fig 3e to mirror.*

## **A sequence-to-structure pipeline predicts mutational effects on specificity**  → Fig 4  *(MyoD1/KLF4 capstone)*

### **Sequence localizes but does not resolve mutation-induced specificity switches**  → Fig 4a

MyoD1 L112R (→CACGTG), KLF4 K409Q (→GGGTGGGTG): TFScope localizes the affected position but cannot name the novel base — an honest limit of sequence-only on de-novo variants. **[DONE].**

### **Structure refolding and energetic scanning recover the altered specificity**  → Fig 4b–c

Refold each candidate base as the biological oligomer → rank by interface energy → recovers CACGTG (one-hot G; ΔΔ≈24 kcal/mol); in-place scan fails, refold succeeds; DeepPBS-on-dimer also recovers G. The hybrid: sequence localizes → structure resolves. **[DONE/MyoD1].** Add KLF4 2×2 (WT/mut protein × WT/mut DNA). **[LACKING]** (needs KLF4 AF3 folds). Hybrid-pipeline schematic → Fig 4d.

---

# DISCUSSION

*(Continuous prose; thematic plan:)*

- Sequence-only specificity enables coverage/scale; contact distillation transfers structural knowledge without inference-time structures.
- The sequence↔structure division of labor (sequence localizes & scales; structure resolves atomic contacts & novel-variant bases) — complementary, not competitive; supported by the AF3 consensus-folding plausibility and the hybrid.
- One sentence on conditioning: structural family labels are **motif-incoherent** (a family maps to many motif grammars), which is why TFScope conditions on learned features rather than family text — a transferable lesson; details in Supplementary. *(was a Results section; demoted as unresolved/exploratory.)*
- Limitations: de-novo/mutant base identity; orphan results candidate-level; modest benchmark margin (mitigated by baselines + significance; LOFO when complete).
- Outlook: oligomer-aware inputs; richer contact distillation; coupling to occupancy data.

---

# METHODS  *(noun-phrase headers, Nature Methods style)*

### Data curation and family assignment

Aug + DeepPBS corpus (4250 records); DBD-canonical trim; PWM canonicalization; family schemes (learned-10, rebin-34, famv2-46, coarse-12). **[DONE].**

### Dataset splits and contamination control

cluster40 (40%-DBD-identity); deeppbs cluster40 test (84); leave-family-out indices; leakage audit. **[DONE]** (LOFO eval [LACKING]).

### Model architecture

ESM2-650M + LoRA; gated DBD pooling; family-aware MoE (SwiGLU experts, FiLM, semantic gate); cross-attention PWM head; gate head; hyperparameters (global batch 36, lr 4.5e-4, 225 ep). **[DONE].**

### Contact supervision

Recognition-residue prior (cluster40-train-only); 1D contact loss (w=0.3); train-only distillation. **[DONE].**

### Evaluation metrics and motif alignment

oracle-aligned r (gate-active core + IC trim + ±10/RC), panel-r, per-pos r, MAE (per-column Σ|pred−gt| ∈ [0,2], as DeepPBS), KL, top-1; alignment applied symmetrically to all methods including DeepPBS. **[DONE].**

### Baselines

Random (shuffled-column + dinuc-matched), NN-PWM, ESM2-linear, DeepPBS ensemble. **[DONE].**

### Structural plausibility of predictions (AF3 consensus folding)

For all test TFs, AF3-fold TF + predicted consensus oligo (TFScope and DeepPBS separately; correct oligomer); compare ipTM and interface-pLDDT, paired per-TF. Inputs `manuscript/AF3_consensus_folding_inputs.md`. **[PARTIAL — folds in progress].**

### Interpretation and in silico mutagenesis

Single-residue scan + gradient saliency; recognition-code comparison (Wetzel data); prototype/attention decoding. **[PARTIAL].**

### Structure folding and energetic scanning

AF3 folding (oligomer state); `pwm_hybrid` PyRosetta `-relax`, `PWM_INTERFACE_MODE=prot_dna`; interface-ΔΔG → Boltzmann PWM (tau=1.5). **[DONE]** (pipeline); contact-discovery generalization [PARTIAL].

### Genome-wide motif scanning

MOODS, hg38, p<1e-4, both strands; ENCODE cCREs; dinucleotide-shuffle composition control. **[DONE].**

### Code and data availability

Repo, checkpoints, splits, indices. **[DONE].**

---

# CRITICAL-PATH EXPERIMENT QUEUE (ordered)

1. **[DONE]** Baseline ladder (random, NN-PWM, ESM2-linear) + significance — *headline = matches DeepPBS.*
2. **[DONE]** Per-family breakdown.
3. **[PARTIAL]** AF3 consensus-folding ipTM/pLDDT, TFScope vs DeepPBS, all 84 — *analyze when folds return (Fig 1f).*
4. **[PARTIAL]** In silico mutational scan → specificity residues (have MyoD1/KLF4; generalize to population, Fig 2b).
5. **[LACKING]** Wetzel recognition-code validation (needs their data; Fig 2c).
6. **[PARTIAL→DO]** AF3+Rosetta contact discovery on ~10 structure-less in-distribution TFs (Fig 2d).
7. **[LACKING]** KLF4 2×2 refold-energy (needs AF3 folds; Fig 4c).
8. **[PARTIAL]** Held-out motif recovery at scale (Fig 3a).
9. **[PARTIAL]** In silico evolution / design demo (confirm DeepPBS Fig 3e; Fig 3d).
10. **[LACKING]** Fig 1a schematic + Fig 4d hybrid schematic.
11. **[LACKING — deferred]** LOFO benchmark + significance vs DeepPBS.

# ALREADY WRITABLE

Benchmark headline + baseline ladder + per-family; contact ablation + nulls; orphan CRE scan; MyoD1 mutation + hybrid.

# CUT FROM MAIN TEXT (→ Supplementary / Discussion sentence)

Family-conditioning negative + motif-incoherence (learned-10 > semantic-34 > -46 > coarse-12; ~372 motif sub-families). Unresolved/exploratory; in-flight RAG & per-protein-text variants are supplementary, not main.
