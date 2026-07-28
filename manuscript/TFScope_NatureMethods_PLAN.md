# TFScope — Nature Methods Manuscript Plan

*Planning doc. Status tags: **[DONE]** · **[PARTIAL]** · **[LACKING]** (protocol given inline). Results headers are written in Nature Methods style — short declarative sentences stating the finding; Methods use noun-phrase headers.*

**Working title:** *Structure-free prediction of transcription-factor DNA-binding specificity by distilling structural recognition into a sequence model.*

**One-line contribution:** TFScope predicts TF binding specificity (PWM) from protein sequence alone — competitive with structure-based methods — by distilling structural recognition-residue contacts into a sequence model, enabling specificity for the majority of TFs that lack structures.

**Figures (4 multi-panel, Nature Methods style):**
- **Fig 1** — Structure-free prediction matches structure *(method + benchmark + baselines + per-family + AF3 consensus-folding plausibility).*
- **Fig 2** — Contact distillation drives accuracy and yields a learned recognition code *(contact ablation + nulls + mutagenesis + Wetzel code + AF3+Rosetta contact discovery + prototypes).*
- **Fig 3** — Proteome-scale specificity for orphan TFs *(held-out recovery + orphan logos + CRE enrichment + evolution/design + de novo designed-protein recovery).*
- **Fig 4** — Mutation effects via a sequence→structure hybrid *(MyoD1 L122R switch score + ER↔GR P-box titration + AF3+Rosetta structure-calibration to G + hybrid schematic).*

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

Architecture/overview: ESM2/LoRA encoder → gated DBD pooling → family-conditioned MoE → cross-attention PWM head + gate head; trained with recognition-residue **contact supervision** (structure-distilled, train-only; inference is sequence-only). **[DONE]** (model). Figure schematic **[LACKING — schematic only; ready prompt below]** — build with `scientific-schematics`.

> **Fig 1a prompt** (`scientific-schematics`; reshaped to *Nature Methods* Fig 1 house style — flat 2D editorial vector schematic, single clean left→right workflow, muted professional palette, concrete biological input/output, no 3D/photoreal):
>
> *"A clean, publication-quality method-overview schematic in the visual style of a Nature Methods Figure 1a: a single horizontal left-to-right pipeline on a white background, flat 2D vector / line-art (NOT 3D-rendered, NOT photorealistic, NOT neon or dark-mode), a calm muted palette of desaturated blue-grey and teal with one warm accent, generous white space, thin clean connector arrows, small sans-serif labels, and consistent visual language throughout. The pipeline shows TFScope, which predicts a transcription factor's DNA-binding motif from protein sequence alone. Left to right: (1) INPUT — a small grey ribbon cartoon of a transcription factor with its DNA-binding domain region tinted, and beneath it a short amino-acid letter string; label 'TF sequence (DNA-binding domain)'. (2) a stylized stacked-block slab labelled 'ESM-2 (650M) protein language model' with a small tab 'LoRA', emitting a single row of per-residue embedding tiles. (3) a block 'Gated DBD pooling' showing a small gate-weight bar track over the residue row that highlights the DBD residues and collapses them into one pooled vector. (4) a block 'Family-conditioned mixture-of-experts' drawn as a small router node fanning into four or five compact expert blocks with one expert highlighted. (5) two stacked OUTPUT heads on the right: 'cross-attention PWM head' producing a DNA sequence-logo (the predicted position weight matrix, drawn as stacked A/C/G/T letters of varying height) shown next to a short, simple DNA double-helix to ground it, and below it a thin 'gate head' bar marking the motif core. A separate DASHED, greyed-out box labelled 'recognition-residue contact supervision (distilled from protein-DNA structures)' connects upward into the PWM head with a small tag 'training only'. A slim banner along the bottom reads 'Inference: protein sequence in -> binding motif out — no structure required'. Keep it minimal and uncluttered, colorblind-friendly, all text legible, no fake equations or random characters, vector-clean lines."* → `figures/figure1a_architecture.png`.
>
> *Reference style: method-overview Fig 1 schematics in recent Nature Methods protein-ML papers (e.g. CARP / "Protein language models using convolutions" 2024; METL / "Biophysics-based protein language models" 2025) — flat workflow diagrams with concrete sequence/structure glyphs, muted color-coded modules, minimal text. NOTE: a draft `figures/tfscope_architecture.pdf` already exists; decide whether to regenerate to this NM style or refine the existing one in vector software (Illustrator) for final submission, since AI-generated diagrams should be hand-verified before a Nature Methods submission.*

## **Sequence-only prediction matches structure-based methods on a contamination-controlled benchmark**  → Fig 1b–c

The cluster40 (40%-DBD-identity) split and the contamination finding (structure-level split inflates both ~0.25); `combined` **mean oracle-r 0.657 vs DeepPBS 0.626** (mean gate-oracle-r; all methods scored by one unified protocol on identical GT cores, `results/baseline_ladder/ladder_mean.json`; Δ +0.031, paired-bootstrap p=0.30, "matches"). **[DONE].**

### **A baseline ladder isolates the architecture's contribution**  → Fig 1d

**[DONE].** cluster40 test (n=84), one unified mean gate-oracle-r protocol for every method (each method's IC core oracle-aligned ±10/RC to the GT core; `results/baseline_ladder/ladder_mean.json`). Mean-r ladder: **random-uniform** 0.464 (floor) → **random-train-PWM** 0.518 → **NN-PWM** 0.534 (leave-gene-out nearest DBD by ESM2-cosine, copy donor PWM) → **ESM2-linear** 0.555 (frozen ESM2 mean→MLP→PWM, trained on the leakage-free 467-record clean split — the "is the architecture worth it?" test) → **DeepPBS** 0.626 → **TFScope** 0.657. TFScope−DeepPBS Δ+0.031, paired-bootstrap p=0.30, 95% CI [−0.028, +0.089] → **matches**; TFScope−ESM2-linear **+0.102** → architecture justified. (NB: training ESM2-linear on the full augmented corpus instead leaks — median jumps to 0.74 > TFScope — because the corpus holds family-matched near-duplicates of the test TFs; the clean 467-split value is the fair control.)

### **Accuracy varies systematically by protein family**  → Fig 1e

**[DONE].** Per-family oracle-r, TFScope vs DeepPBS (`results/per_family/`): TFScope wins Other +0.071 / NR +0.083 / bHLH +0.074; DeepPBS wins bZIP −0.357 / C2H2_long −0.182 — sequence-only loses exactly on the most motif-incoherent / quaternary-recognition families. Honest and mechanistically self-consistent.

### **Predicted consensus oligos fold into high-confidence protein–DNA complexes**  → Fig 1f

**[PARTIAL — folds running on other machines].** For all 84 cluster40 test TFs, fold (AF3) the TF with its **TFScope-predicted** highest-IC consensus oligo and, separately, with its **DeepPBS-predicted** consensus (length-matched, correct oligomer state; inputs in `manuscript/AF3_consensus_folding_inputs.md`). Compare **ipTM** and interface-**pLDDT** distributions TFScope vs DeepPBS (paired, per-TF). Read-out: do sequence-only predictions yield protein–DNA complexes as physically confident as the structure-based method's? — an orthogonal, structure-based validation of the benchmark that does not depend on the oracle-aligned PWM metric. *No GT fold needed (experimental structures exist for the test set).* Analyze ipTM/pLDDT when folds return.

### **Predictions generalize to held-out protein families**  → Fig 1 (added when done)

**[LACKING — deferred ~12–20 h].** Leave-family-out (train all-but-X, test X) for 5 families (Homeodomain, merged-C2H2, bZIP, NR, bHLH); reuse `tf_nn_index_lofo_`* + `eval_lofo.py`; per-family + mean; paired-bootstrap significance on the combined-vs-DeepPBS margin.

## **Distilling structural contacts drives accuracy and recovers the protein–DNA recognition code**  → Fig 2  *(core contribution)*

### **Recognition-residue contact supervision is the principal accuracy lever**  → Fig 2a

**[DONE]** — figure `results/baseline_ladder/fig2a_ablation.{png,pdf}` (single panel, oracle-aligned r). Built up from the **same training records DeepPBS uses**, two synergistic levers carry the sequence-only model past structure: (i) a model trained on the **DeepPBS training set alone** (467 records, sequence-only) reaches oracle-r **0.548** — below structure-based DeepPBS (0.626, mean gate-oracle-r, same unified protocol as Fig 1d); (ii) **corpus augmentation** to the full 4,250-record corpus lifts it to **0.596 (+0.05)**; (iii) **distilling recognition-residue contacts** (1D, w=0.3, train-only) lifts it to **0.657 (+0.06)**, now *above* DeepPBS. Neither lever alone suffices — augmentation closes most of the data gap, contact distillation supplies the structural-recognition signal that PWM labels lack, and together they synergise (each ≈ +0.05–0.06).

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

**Refined design — experts = recognition MODES, not families (key insight, 2026-06-22):**
Do NOT set `num_experts` = #families (12). Distinct families often share the same DNA-recognition
chemistry, so the true latent basis is ~3–6 *generalized recognition modes*. Forcing 12 experts to
split by family fights this structure — which is *why* the entropy/diversity loss could push to
uniform at zero task cost. Proposed coarse mapping of the 10 families → ~5 modes:
  1. **Helical major-groove readout (monomeric HTH / winged-helix):** Homeodomain, ETS, Forkhead
     — a recognition α-helix inserted into the major groove.
  2. **Zinc-coordinated readout:** C2H2_short/medium/long (ββα, Zn-stabilized recognition helix).
  3. **Dimeric basic-helix (leucine-zipper / HLH):** bZIP, bHLH — each monomer reads a half-site.
  4. **Nuclear-receptor-like:** Nuclear_Receptor (C4 Zn modules, dimeric direct/inverted repeats).
  5. **β-sheet / loop & other special:** Other (T-box Ig-fold, STAT, p53 β-sandwich, Rel, MADS…).
Concretely: set `num_experts ≈ 5–6` (5 modes + optional 1 shared general expert); replace the
anti-specialization `family_diversity_loss` with either (a) coarse *mode-label* routing supervision
(map family_id → mode_id, light CE on the gate), or (b) nothing extra — with only ~5 experts and the
shared/residual dominance reduced, specialization should emerge. Expected payoff: a far cleaner Fig 2e
— "the model discovers a small basis of DNA-recognition mechanisms, and families that share a binding
chemistry share an expert" — which is a stronger interpretability claim than per-family prototypes.

## **TFScope nominates binding motifs for orphan transcription factors at proteome scale**  → Fig 3  *(the positive, sequence-only-unique story)*

### **Known motifs are recovered from sequence alone for held-out factors**  → Fig 3a

**[PARTIAL].** Systematic held-out recovery (predicted vs curated JASPAR/HOCOMOCO motif r) across families + masked same-family controls — the scalable advantage over structure-based methods.

### **Nominated motifs are enriched in cis-regulatory elements**  → Fig 3b–c

Orphan nominations (SOHLH1, ZGLP1, ADNP, ADNP2, ZHX2/3); genome-wide CRE scan with dinucleotide-shuffle composition control → 5/6 motifs cCRE-enriched; homeodomains→promoters (ADNP2 17.5×), SOHLH1 E-box→enhancers (1.4×). The composition control flips the naive GC-confounded read. **[DONE]** (`results/genome_cre_scan/`). Frame as candidate-level functional plausibility, not occupancy.

### **Model-guided optimization recovers and redesigns specificity**  → Fig 3d

**[PARTIAL]** (have `evolve_pwm_`*). DNA-side — evolve DNA to maximize predicted binding, recover consensus; protein-side — directed mutation trajectory. *Confirm DeepPBS Fig 3e to mirror.*

### **Sequence-only prediction recovers the specificity of de novo designed DNA-binding proteins**  → Fig 3e

**[DONE].** Applied TFScope (combined, sequence-only) to four de novo designed helix–turn–helix DNA-binding proteins — **DBP5, DBP6, DBP9, DBP35**, all engineered against the same target **GCAGATCTGCACATC** — from Glasscock et al. (*Nat. Struct. Mol. Biol.* 2025; DOI 10.1038/s41594-025-01669-4). This is a strict **out-of-distribution** test: designed proteins share no evolutionary history with the natural TFs in training. TFScope-predicted PWMs (logos) are compared against the study's **experimental single-base-pair mutation binding heatmaps** (yeast-display flow-cytometry competition; lower normalized PE/FITC = stronger binding). Mirrors **DeepPBS Fig 5**.

**Finding — graded, not binary.** TFScope recovers the central **CAC** recognition core from sequence alone, **with confidence that tracks the experimental signal**: **DBP6** is confident and correct (tall high-IC CAC over precisely the specificity-determining columns; **83% per-position agreement, Pearson r=0.65**); **DBP5/9/35** are partially recovered (**r=0.39–0.59**; mean across four designs **r=0.53**). The claim is therefore *calibrated graded prediction* — strong where the designed interface makes a natural-like CAC contact, appropriately tentative otherwise — not all-or-nothing recovery.

**Honest caveats.** Flanks carry little predicted information (sensible for unbound DNA; the structure-driven A-tract / narrow-minor-groove flank readout that DeepPBS infers is **out of scope** for a sequence-only model — state, don't claim). DBP5/DBP9 mis-register the C-core (combined-best predicts A>C at the key position) — a genuine model limitation, faithfully shown (DBP6 proves the pipeline is correct). Metric was **sign-corrected** (earlier `designs_all_models.json` core_r was inverted; the `cac`-substring boolean rewarded a GTG bias); DBP6 WT reference = 0.1202 (an xls row first missed). Once scored correctly, **combined ≈ semfam34_fixed** (combined-best mean r=+0.531 vs semfam34 +0.515) — the earlier "only semfam34 recovers CAC" was a metric artifact.

**Assets.** Figure `figures/figure_dbp_heatmap/dbp_tfscope_heatmap.{pdf,png,svg}` (TFScope predicted logo over experimental mutation-effect heatmap, DeepPBS palette `["#727DB7","#D9DBEC","#FFFFFF","#F7D9D7","#E96B68"]`, log2-vs-WT, WT base boxed). Builder `scripts/build_dbp_tfscope_heatmap.py`; corrected per-design metric via the match-best-binding analysis; narrative `manuscript/results_designed_dbp_narrative.md`; combined epoch sweep `results/design_case_study/combined_epoch_sweep_dbp.json`. Citation `Glasscock2025` in `TFScope_refs.bib`. (Also drafted into the manuscript .tex as a Results subsection + Methods "Designed-protein evaluation".)

## **A sequence-to-structure pipeline predicts mutational effects on specificity**  → Fig 4  *(MyoD1 capstone; KLF4 dropped)*

### **Sequence reproduces the direction of mutation-induced specificity switches**  → Fig 4a

**[DONE].** Measured with a directional difference-in-differences **switch score** (NOT consensus argmax, which is unstable to single-column noise): for predicted PWMs, $\Delta_{\text{switch}} = [S_{\text{mut}}(\text{CACGTG}) - S_{\text{mut}}(\text{CACCTG})] - [S_{\text{WT}}(\text{CACGTG}) - S_{\text{WT}}(\text{CACCTG})]$, $S$ = best PWM log-odds. **MyoD1 L122R** (basic-region L→R): WT strongly prefers the myogenic CACCTG ($S_G-S_C=-11.8$); L122R lifts CACGTG from $-0.4$ to $+5.8$ and collapses the gap to $-2.6$ → **$\Delta_{\text{switch}}=+9.17>0$**: from sequence alone TFScope **reproduces the documented shift toward the MYC-like CACGTG** (a strong directional shift, not a full flip — the mutant still marginally favors CACCTG; a consensus-only readout would have missed it). Figure `figures/figure4a_switch/`. **Determinant-size titration** (ER↔GR P-box / recognition-module swap; `figures/figure4a_titration/`): sequence-only resolution **scales with the number of specificity-determining residues swapped** — single residues / the 3-residue P-box perturb but do not redirect the motif (corr-to-target ~0.34), whereas swapping the bulk of the recognition module crosses "resolved" (corr ≥0.7) at 50–75% — i.e. TFScope captures specificity at the level of the recognition module, not of individual de-novo substitutions. **KLF4 K409Q dropped** (single C2H2 residue, insensitive). Builders `scripts/build_fig4a_switch.py`, `scripts/build_fig4a_titration.py`; narrative `manuscript/fig4a_narrative.md`.

### **AF3 refolding + Rosetta interface energy calibrate the mutant base to the true switch**  → Fig 4b–c

**[DONE].** Sequence localizes the affected E-box position but assigns the wrong de-novo base (L122R: TFScope predicts CAC**A**TG; truth CAC**G**TG). A targeted **structure-calibration** step corrects it: for each candidate base, AF3-refold the L122R bHLH **homodimer** on that E-box (5 models) and score the protein–DNA interface with Rosetta `pwm_hybrid -relax` (RM8B charges, default `ddg_filter`, `metal_free`). **By ensemble median the true switch CACGTG (G) ranks best ($-25.9$ REU)**, WT myogenic CACCTG second ($-24.9$), and **TFScope's CACATG (A) decisively worst ($-13.3$, ~11–13 REU above)** — structure prefers **G** and **calibrates TFScope's sequence-only error (A→G)**. DeepPBS-on-the-dimer also recovers G (orthogonal). The hybrid: **sequence localizes & scales → structure calibrates the atomic base**.

**Honest method notes (load-bearing):** (i) the **in-place** single-base ΔΔG scan FAILS — dominated by backbone-strain clash that swings ±100s of REU across AF3 models and re-elects the seeded base (G scored *worst* inside the CACATG-built backbone); only **refold-per-candidate** (each base gets its own relaxed backbone) recovers G. (ii) Use the **ensemble median**, not mean/single-model: per-model relax variance is large (CACATG models span $-26$→$+21$), and AF3 confidence is **uniform across all four bases** (ranking_score 0.91–0.94, no clash) — AF3 cannot tell the correct base, so the **Rosetta interface energy is the calibrator, not AF3**. (iii) An iterative in-place "scan→refold→re-scan" evolution loop **diverges** (chaotic / GC-drift, non-convergent) — documented negative control; refold-per-candidate is the sound procedure. **KLF4 2×2 dropped** (K409Q insensitive). Data `results/myod1_mut/refold_ensemble_ddg.json`, `results/pwm_rosetta/myod1_*_model0_scan/`; pwm_rosetta settings = beta_nov16 + RM8B charges (default `ddg_filter`, `-relax`). Hybrid-pipeline schematic → Fig 4d **[LACKING — schematic only; ready prompt below]**.

> **Fig 4d prompt** (`scientific-schematics`): *"Clean horizontal two-stage flowchart for a hybrid sequence-to-structure pipeline that predicts the effect of a mutation on transcription-factor DNA-binding specificity. STAGE 1, labelled 'Sequence — localize' (blue): a box 'Mutant TF sequence (MyoD1 L122R)' → arrow → box 'TFScope (sequence-only PWM head)' → a small DNA sequence-logo output reading CACATG with the central position highlighted, captioned 'localizes the affected E-box position but assigns the wrong base (A)'. STAGE 2, labelled 'Structure — calibrate' (orange): from that highlighted position, branch to the four candidate bases A/C/G/T → box 'AF3 refold the bHLH homodimer on each candidate E-box (5 models)' → box 'Rosetta interface ΔΔG, ensemble median' → a sequence-logo output reading CACGTG with G highlighted, captioned 'calibrates the base to the true switch (G)'. A wide ribbon arrow spanning both stages along the bottom reads 'sequence localizes & scales → structure calibrates the atomic base'. Minimal, publication-quality, colorblind-friendly, white background, large readable sans-serif labels, no photoreal rendering."* → `figures/figure4d_hybrid_schematic.png`.

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
7. **[DONE]** MyoD1 L122R structure-calibration: AF3 refold-per-candidate (5-model ensemble) + Rosetta interface ΔΔG → ensemble-median ranks CACGTG (G) best, calibrates TFScope's A→G (Fig 4b–c). KLF4 2×2 dropped (K409Q insensitive).
8. **[PARTIAL]** Held-out motif recovery at scale (Fig 3a).
9. **[PARTIAL]** In silico evolution / design demo (confirm DeepPBS Fig 3e; Fig 3d).
10. **[LACKING — schematic only, prompts ready]** Fig 1a architecture schematic + Fig 4d hybrid-pipeline schematic (ready `scientific-schematics` prompts in the Fig 1a / Fig 4b–c plan entries; Fig 1a may reuse `figures/tfscope_architecture.pdf`).
11. **[LACKING — deferred]** LOFO benchmark + significance vs DeepPBS.

# ALREADY WRITABLE

Benchmark headline + baseline ladder + per-family; contact ablation + nulls; orphan CRE scan; MyoD1 mutation + hybrid; **de novo designed-protein recovery (Fig 3e, DBP5/6/9/35 vs Glasscock 2025 experimental heatmaps)**.

# CUT FROM MAIN TEXT (→ Supplementary / Discussion sentence)

Family-conditioning negative + motif-incoherence (learned-10 > semantic-34 > -46 > coarse-12; ~372 motif sub-families). Unresolved/exploratory; in-flight RAG & per-protein-text variants are supplementary, not main.
