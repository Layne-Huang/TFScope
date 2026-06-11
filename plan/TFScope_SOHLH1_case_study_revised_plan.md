# Revised SOHLH1 Case Study Plan for TFScope

**Purpose:** Implement a Nature Methods-style application case study showing that TFScope can generate a **confidence-calibrated, sequence-only motif hypothesis** for an orphan human transcription factor, using SOHLH1 as the focused example.

**Main claim to support:**

> TFScope prioritizes SOHLH1 as a medium-confidence, retrieval-supported E-box candidate from amino-acid sequence alone, without requiring a protein-DNA complex structure.

**Important wording constraint:** Do **not** claim that the true SOHLH1 binding motif is experimentally validated. The correct claim is that TFScope nominates a **testable motif hypothesis**.

---

## 0. Manuscript context and constraints

The current TFScope manuscript already contains:

1. TFScope architecture: amino-acid sequence + DBD mask -> PWM, no structural input.
2. DeepPBS comparison: TFScope reaches structure-based performance without protein-DNA complexes.
3. cluster40 and leave-family-out benchmarks: leakage-controlled generalization and family-level transfer limits.
4. Attention analysis: contact-aware attention can focus on known DNA-recognition residues.

Therefore this SOHLH1 case study should **not repeat cluster40 or LFO benchmark results**. Instead, it should use them only to calibrate confidence and motivate the application.

The case study should answer:

> Given a real orphan human TF with no curated motif, can TFScope produce a confidence-ranked, biologically plausible, experimentally testable motif hypothesis?

---

## 1. Target figure redesign

The old version had repeated PWM logo panels. Replace it with the following structure.

### Figure 5a. Confidence-calibrated orphan TF prioritization

**Goal:** Replace the weak funnel plot with a result-driven confidence distribution.

Show:

- Distribution of TFScope confidence scores for held-out known TFs.
- Distribution of TFScope confidence scores for orphan TFs.
- Mark SOHLH1 with a red arrow or vertical marker: `SOHLH1 confidence = 0.61`.
- Optional inset: calibration curve on held-out known TFs.

Recommended panel variants:

1. **Main histogram/density:**
   - x-axis: calibrated confidence score.
   - y-axis: density or count.
   - groups:
     - held-out known TFs, successful predictions (`oracle_r >= 0.6`)
     - held-out known TFs, unsuccessful predictions (`oracle_r < 0.6`)
     - orphan TF predictions
   - SOHLH1 marker at 0.61.

2. **Inset calibration curve:**
   - x-axis: confidence bin.
   - y-axis: observed fraction with `oracle_r >= 0.6`.
   - Use cluster40/held-out known TFs only.

**Interpretation:** SOHLH1 is not cherry-picked; it falls into a calibrated medium-confidence region of the orphan TF prediction distribution.

---

### Figure 5b. SOHLH1 target card and leakage audit

**Goal:** Show why SOHLH1 is a clean sequence-only orphan target.

Show:

- SOHLH1 protein schematic:
  - length: 328 aa
  - bHLH DBD: currently estimated as 53-104; verify from UniProt/InterPro/HumanTFs before finalizing.
- Labels:
  - `SOHLH1 (UniProt Q5JUK2)`
  - `germ-cell bHLH TF`
  - `no curated motif in JASPAR/HOCOMOCO/CIS-BP` (verify programmatically)
  - `no protein-DNA complex structure` (verify by PDB/AlphaFold or RCSB mapping if possible)
  - `absent from TFScope training/retrieval/benchmark tables` (must be verified locally)
- Input statement:
  - `Input to TFScope: amino-acid sequence + bHLH DBD mask only`

**Required output table:**

`results/sohlh1_case/leakage/SOHLH1_leakage_audit.tsv`

Columns:

```text
gene_symbol
uniprot_id
in_training_gene
in_training_motif
in_retrieval_index
in_benchmark_tables
max_train_dbd_identity
max_retrieval_dbd_identity
same_gene_retrieval_allowed
same_gene_retrieval_found
same_paralog_retrieval_found
notes
```

Acceptance criteria:

- `SOHLH1` must not appear as same-gene in training, retrieval, or benchmark tables.
- Same-gene retrieval must be disabled.
- SOHLH2 should not be allowed as a retrieved neighbor if used as post hoc paralog reference.

---

### Figure 5c. SOHLH1 motif prediction and reference comparison

**Goal:** Merge the previous Panels c and d into one compact PWM comparison panel.

Show four logos in one row or 2x2 grid:

1. SOHLH1 noRAG prediction
   - label: `retrieval-free: weak GC-rich prior`
2. SOHLH1 LGO-RAG prediction
   - label: `leave-gene-out RAG: E-box hypothesis`
3. SOHLH2 reference motif
   - label: `SOHLH2 paralog reference, JASPAR MA1560.1`
4. Canonical E-box motif
   - label: `canonical E-box CACGTG`

Add a small metrics table below or beside the logos:

```text
comparison                              similarity
SOHLH1 noRAG vs SOHLH1 RAG              r = <fill>
SOHLH1 noRAG vs SOHLH2                  r = <fill>
SOHLH1 RAG vs SOHLH2                    r = 0.76 currently
SOHLH1 RAG vs canonical E-box            r = 0.75 currently
SOHLH1 RAG mean IC                       1.36 bits currently
SOHLH1 calibrated confidence             0.61 medium currently
```

**Important wording:**

Do not say “TFScope recovered the SOHLH1 motif.” Say:

> The retrieval-augmented SOHLH1 prediction is substantially more similar to the SOHLH2 paralog motif and canonical E-box than the retrieval-free prediction, providing a paralog-level plausibility check rather than direct validation.

Required outputs:

```text
results/sohlh1_case/predictions/SOHLH1_noRAG.pwm.tsv
results/sohlh1_case/predictions/SOHLH1_RAG_LGO.pwm.tsv
results/sohlh1_case/predictions/SOHLH1_prediction_metrics.tsv
figures/figure5/panel_c_sohlh1_logo_comparison.pdf
figures/figure5/panel_c_sohlh1_logo_comparison.png
```

---

### Figure 5d. SOHLH1-like held-out validation / masked SOHLH2 control

This is the key new validation panel. Implement both if possible; otherwise implement at least one.

#### Option 1: Held-out bHLH confidence calibration

**Goal:** Show what confidence = 0.61 means for known bHLH TFs.

Data:

- Held-out known TFs from cluster40/test or another leakage-clean validation table.
- Restrict to bHLH if enough samples exist; otherwise show all TFs with bHLH highlighted.

Plot:

- x-axis: calibrated confidence.
- y-axis: actual motif accuracy, e.g. oracle Pearson r.
- points: held-out known TFs.
- highlight bHLH points.
- vertical line: SOHLH1 confidence = 0.61.
- optional horizontal line: success threshold `oracle_r = 0.6`.

Output:

```text
figures/figure5/panel_d_confidence_vs_accuracy.pdf
results/sohlh1_case/validation/heldout_confidence_accuracy.tsv
```

Text to support:

> Held-out TFs in the same confidence range as SOHLH1 provide an empirical calibration of how often medium-confidence predictions recover experimental motifs.

#### Option 2: Masked SOHLH2 positive control

**Goal:** Test whether the same workflow can recover a known paralog motif when SOHLH2 is artificially treated as orphan.

Procedure:

1. Remove SOHLH2 from retrieval candidates.
2. If feasible, use a checkpoint/index where SOHLH2 is not a training target. If not feasible, at minimum remove SOHLH2 from retrieval and clearly label this as a retrieval-masked control, not full train-masked validation.
3. Run TFScope on SOHLH2 noRAG and LGO-RAG.
4. Compare predicted SOHLH2 motif to JASPAR MA1560.1.
5. Report whether the workflow recovers the known E-box-like SOHLH2 motif.

Output:

```text
results/sohlh1_case/validation/SOHLH2_masked_control_metrics.tsv
figures/figure5/panel_d_sohlh2_masked_positive_control.pdf
```

Important label:

- If SOHLH2 was only removed from retrieval, call it `retrieval-masked SOHLH2 control`.
- If SOHLH2 was removed from both training and retrieval, call it `fully masked SOHLH2 positive control`.

Do **not** overstate if training masking was not possible.

---

### Figure 5e. Biological nomination: germ-cell regulatory elements or target candidates

**Goal:** Translate the motif hypothesis into testable biological hypotheses.

Preferred implementation: promoter/enhancer motif scoring.

Data sources, in order of preference:

1. Germ-cell / spermatogenesis / oogenesis / folliculogenesis gene sets from GO/MSigDB or curated local gene list.
2. Promoters of SOHLH1-related genes, e.g. TSS +/- 2 kb.
3. Optional: public germ-cell ATAC-seq/enhancers if already available; do not make ATAC essential.

Procedure:

1. Build target promoter set:
   - genes annotated with spermatogenesis, oogenesis, folliculogenesis, germ cell development.
2. Build background promoter set:
   - GC-matched random promoters.
   - match promoter length and chromosome distribution if possible.
3. Scan with motifs:
   - SOHLH1 RAG PWM.
   - SOHLH1 noRAG PWM.
   - SOHLH2 JASPAR MA1560.1.
   - canonical E-box CACGTG.
   - shuffled SOHLH1 RAG PWM.
4. Compute:
   - best motif score per promoter.
   - hit rate above FIMO-like threshold.
   - enrichment vs background.
   - AUROC/AUPRC if treating target set vs background as labels.
5. Produce either:
   - enrichment boxplot/violin; or
   - ranked candidate target table with top 10 genes.

Outputs:

```text
results/sohlh1_case/targets/germ_cell_gene_set.tsv
results/sohlh1_case/targets/promoter_scan_scores.tsv
results/sohlh1_case/targets/top_candidate_targets.tsv
figures/figure5/panel_e_target_enrichment.pdf
```

Recommended wording:

> This promoter analysis does not establish direct SOHLH1 occupancy, but nominates candidate regulatory elements for future PBM, HT-SELEX, EMSA, or reporter validation.

If promoter enrichment is weak:

- Do not force it into the main figure.
- Replace panel e with a concise candidate target table or move target scan to supplement.
- Use attention map as Extended Data only.

---

## 2. Extended Data panels

### Extended Data 5a. SOHLH1 attention map

Use the existing attention heatmap idea, but improve labeling.

Required improvements:

- Mark the bHLH basic DNA-recognition region as a shaded band.
- Mark helix-loop-helix region separately if coordinates are available.
- Add a column-summed attention bar plot on top or right.
- Avoid claiming causal residues unless mutational sensitivity supports it.

Output:

```text
figures/extended/extended_sohlh1_attention_map.pdf
results/sohlh1_case/attention/SOHLH1_attention.tsv
```

Caption wording:

> The attention pattern provides an interpretable link between the predicted motif and the bHLH DNA-recognition region, but does not by itself validate SOHLH1 binding specificity.

### Extended Data 5b. Full orphan TF confidence table

Output:

```text
results/sohlh1_case/orphan_tf_confidence_table.tsv
```

Columns:

```text
gene_symbol
uniprot_id
family
dbd_start
dbd_end
known_motif_status
in_training_gene
in_retrieval_index
confidence
confidence_class
noRAG_vs_RAG_similarity
mean_IC_RAG
gate_confidence
retrieval_top1_gene
retrieval_top1_similarity
retrieval_top3_genes
notes
```

---

## 3. Implementation tasks for Codex / Claude Code

Implement the tasks below as modular scripts. Use existing TFScope inference utilities where possible instead of rewriting model code.

### Task 1. Create configuration file

Create:

```text
configs/case_study_sohlh1.yaml
```

Suggested fields:

```yaml
project_name: sohlh1_case_study
output_dir: results/sohlh1_case
figure_dir: figures/figure5
extended_figure_dir: figures/extended

# Inputs to update for local repo
human_tfs_database: data/external/humantfs/DatabaseExtract_v_1.01.csv
human_tfs_motif_list: data/external/humantfs/Human_TF_MotifList_v_1.01.csv
jaspar_sohlh2_motif: data/external/jaspar/MA1560.1.jaspar
uniprot_fasta: data/external/uniprot/human_tf_uniprot.fasta
training_gene_table: data/tfscope/training_genes.tsv
retrieval_gene_table: data/tfscope/retrieval_genes.tsv
benchmark_gene_table: data/tfscope/benchmark_genes.tsv
cluster40_predictions: results/cluster40/predictions_with_confidence.tsv
cluster40_truth_motifs: data/tfscope/cluster40_truth_motifs/

# Model checkpoints
checkpoint_production: checkpoints/tfscope_production.pt
checkpoint_cluster40: checkpoints/tfscope_cluster40.pt
retrieval_index_lgo: data/tfscope/retrieval/lgo_index.faiss

# Target
case_gene: SOHLH1
case_uniprot: Q5JUK2
case_family: bHLH
case_dbd_start: 53
case_dbd_end: 104
paralog_reference_gene: SOHLH2
paralog_reference_motif: MA1560.1
canonical_ebox: CACGTG

# Confidence
success_threshold_r: 0.6
confidence_bins: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
medium_confidence_min: 0.4
high_confidence_min: 0.7

# Motif processing
ic_threshold: 0.25
active_gate_threshold: 0.5
max_alignment_offset: 10
allow_reverse_complement: true
```

---

### Task 2. Build orphan TF and confidence input tables

Create script:

```text
scripts/case_study/build_orphan_tf_tables.py
```

Inputs:

- HumanTFs database.
- HumanTFs motif list.
- training/retrieval/benchmark gene tables.
- model outputs for orphan TFs if already computed.

Outputs:

```text
results/sohlh1_case/orphans/orphan_tf_master.tsv
results/sohlh1_case/orphans/SOHLH1_target_card.tsv
results/sohlh1_case/leakage/SOHLH1_leakage_audit.tsv
```

Required checks:

- Identify `Motif status == No motif` entries.
- Identify no direct experimental motif entries if motif evidence is available.
- Confirm SOHLH1 status.
- Confirm SOHLH1 absence from training/retrieval/benchmark tables.
- Confirm SOHLH2 handling status.

Acceptance criteria:

- Script exits with nonzero status if SOHLH1 is found in same-gene training or retrieval without explicit override.
- Script writes a readable audit table.

---

### Task 3. Run SOHLH1 noRAG and LGO-RAG inference

Create wrapper:

```text
scripts/case_study/run_sohlh1_inference.sh
```

It should call existing inference code twice:

```bash
# noRAG
python <existing_tfscope_infer_script>.py \
  --checkpoint checkpoints/tfscope_production.pt \
  --input results/sohlh1_case/orphans/SOHLH1_target_card.tsv \
  --retrieval off \
  --output results/sohlh1_case/predictions/noRAG

# LGO-RAG
python <existing_tfscope_infer_script>.py \
  --checkpoint checkpoints/tfscope_production.pt \
  --input results/sohlh1_case/orphans/SOHLH1_target_card.tsv \
  --retrieval on \
  --retrieval_index data/tfscope/retrieval/lgo_index.faiss \
  --exclude_same_gene true \
  --exclude_genes SOHLH1,SOHLH2 \
  --output results/sohlh1_case/predictions/RAG_LGO
```

Expected extracted outputs:

```text
results/sohlh1_case/predictions/SOHLH1_noRAG.pwm.tsv
results/sohlh1_case/predictions/SOHLH1_RAG_LGO.pwm.tsv
results/sohlh1_case/predictions/SOHLH1_retrieved_neighbors.tsv
results/sohlh1_case/predictions/SOHLH1_gate_probs.tsv
```

Acceptance criteria:

- SOHLH1 and SOHLH2 do not appear in retrieved neighbors.
- Top-3 retrieved neighbors are reported with gene name, family, motif ID, sequence identity/cosine similarity, and PWM source.

---

### Task 4. Compute motif metrics and confidence features

Create script:

```text
scripts/case_study/compute_sohlh1_metrics.py
```

Compute:

- noRAG vs RAG motif similarity.
- noRAG vs SOHLH2 motif similarity.
- RAG vs SOHLH2 motif similarity.
- noRAG vs canonical E-box similarity.
- RAG vs canonical E-box similarity.
- mean IC for noRAG and RAG.
- active motif length.
- mean gate probability over active columns.
- retrieval support score.
- calibrated confidence score and confidence class.

Output:

```text
results/sohlh1_case/predictions/SOHLH1_prediction_metrics.tsv
```

Metric implementation notes:

- Use the same PWM alignment protocol as the manuscript when possible:
  - trim to IC >= 0.25 core;
  - allow offset +/-10;
  - allow reverse complement;
  - report Pearson r and IC-weighted r.
- Also report a deployable fixed-orientation score if implemented.

---

### Task 5. Build confidence distribution and calibration curve

Create script:

```text
scripts/case_study/plot_confidence_distribution.py
```

Inputs:

- Held-out known TF prediction table with actual `oracle_r` and confidence features.
- Orphan TF prediction table with confidence features.
- SOHLH1 confidence metrics.

Outputs:

```text
figures/figure5/panel_a_confidence_distribution.pdf
figures/figure5/panel_a_confidence_distribution.png
results/sohlh1_case/confidence/confidence_calibration_bins.tsv
```

Plot requirements:

- Show held-out known TFs and orphan TFs in the same confidence space.
- Mark SOHLH1 at confidence = 0.61.
- If enough data exist, show accurate vs inaccurate held-out known TFs.
- Include calibration inset or separate calibration panel:
  - confidence bin;
  - observed success fraction (`oracle_r >= 0.6`);
  - number of TFs per bin.

Acceptance criteria:

- Figure makes clear that SOHLH1 is medium confidence.
- Figure does not imply direct validation of SOHLH1.

---

### Task 6. Implement held-out bHLH validation

Create script:

```text
scripts/case_study/heldout_bhlh_validation.py
```

Inputs:

- cluster40 or other held-out prediction table.
- family annotations.
- confidence scores.

Outputs:

```text
results/sohlh1_case/validation/heldout_bhlh_confidence_accuracy.tsv
figures/figure5/panel_d_heldout_bhlh_validation.pdf
```

Plot:

- x-axis: confidence.
- y-axis: actual oracle Pearson r.
- points: held-out TFs.
- bHLH points highlighted.
- vertical line at SOHLH1 confidence.
- horizontal line at success threshold r = 0.6.

Compute summary:

```text
n_bHLH_in_same_confidence_bin
median_r_same_bin
fraction_success_same_bin
n_all_TFs_same_confidence_bin
median_r_all_same_bin
fraction_success_all_same_bin
```

Acceptance criteria:

- If bHLH sample size is too small, fallback to all TFs with bHLH highlighted and report sample size explicitly.

---

### Task 7. Implement masked SOHLH2 positive control

Create script:

```text
scripts/case_study/run_sohlh2_masked_control.sh
```

Procedure:

1. Prepare SOHLH2 input card.
2. Run noRAG and LGO-RAG inference with SOHLH2 excluded from retrieval.
3. Preferably also exclude SOHLH1.
4. Compare prediction against JASPAR MA1560.1.

Outputs:

```text
results/sohlh1_case/validation/SOHLH2_masked_noRAG.pwm.tsv
results/sohlh1_case/validation/SOHLH2_masked_RAG_LGO.pwm.tsv
results/sohlh1_case/validation/SOHLH2_masked_control_metrics.tsv
figures/figure5/panel_d_sohlh2_masked_control.pdf
```

Acceptance criteria:

- Output clearly states whether SOHLH2 was only retrieval-masked or fully train-and-retrieval masked.
- If full train masking is not feasible, do not call it independent validation.

---

### Task 8. Build motif logo comparison panel

Create script:

```text
scripts/case_study/plot_sohlh1_logo_comparison.py
```

Inputs:

- SOHLH1 noRAG PWM.
- SOHLH1 RAG PWM.
- SOHLH2 JASPAR PWM.
- canonical E-box PWM generated from CACGTG.
- metrics table.

Outputs:

```text
figures/figure5/panel_c_sohlh1_logo_comparison.pdf
figures/figure5/panel_c_sohlh1_logo_comparison.png
```

Plot requirements:

- Use same logo style across all motifs.
- Avoid oversized repeated panels.
- Add compact metric annotations.
- Use labels:
  - `SOHLH1 noRAG: weak prior`
  - `SOHLH1 LGO-RAG: E-box hypothesis`
  - `SOHLH2 reference`
  - `canonical E-box`

---

### Task 9. Optional promoter / target-element scanning

Create script:

```text
scripts/case_study/scan_germ_cell_promoters.py
```

Outputs:

```text
results/sohlh1_case/targets/germ_cell_gene_set.tsv
results/sohlh1_case/targets/background_gene_set.tsv
results/sohlh1_case/targets/promoter_scan_scores.tsv
results/sohlh1_case/targets/top_candidate_targets.tsv
figures/figure5/panel_e_target_enrichment.pdf
```

Acceptance criteria:

- Use GC-matched background.
- Report no claim of direct SOHLH1 occupancy.
- If enrichment is weak, still save candidate table but do not use as main panel.

---

### Task 10. Compose final Figure 5

Create script:

```text
scripts/case_study/assemble_figure5.py
```

Inputs:

- Panel a confidence plot.
- Panel b target card.
- Panel c logo comparison.
- Panel d validation/control.
- Panel e target nomination or alternative panel.

Outputs:

```text
figures/figure5/Figure5_SOHLH1_case_study.pdf
figures/figure5/Figure5_SOHLH1_case_study.png
```

Suggested title:

> Confidence-calibrated TFScope annotation nominates an E-box motif for SOHLH1

Alternative title:

> TFScope prioritizes SOHLH1 as a medium-confidence E-box candidate

---

## 4. Manuscript text to generate after analysis

Create:

```text
results/sohlh1_case/manuscript/result5_draft.md
```

Use this wording template, filling in final numeric results:

```markdown
### TFScope nominates a retrieval-supported E-box motif hypothesis for SOHLH1

Having established the generalization behavior of TFScope under leakage-controlled benchmarks, we next asked whether the model could be used as a sequence-only annotation engine for human TFs lacking curated binding motifs. We first calibrated TFScope confidence scores on held-out TFs with known motifs and then applied the calibrated model to orphan human TFs. SOHLH1, a germ-cell bHLH transcription factor, was selected as a focused case because it lacks a curated PWM and has no available protein-DNA complex structure, while its biological role in germ-cell development makes its DNA-binding specificity of interest.

We verified that SOHLH1 itself was absent from the TFScope training, retrieval and benchmark tables. Retrieval-free TFScope produced a weak, low-information GC-rich motif, suggesting that the sequence-intrinsic pathway identified bHLH-compatible specificity but did not resolve a sharp motif. Enabling leave-gene-out retrieval, while excluding SOHLH1 and SOHLH2 from retrieved neighbors, sharpened this weak prior into a medium-confidence E-box motif hypothesis. The retrieval-augmented SOHLH1 prediction had mean information content of <IC> bits and calibrated confidence of <confidence>.

The retrieval-augmented SOHLH1 motif was more similar to the curated SOHLH2 paralog motif and to the canonical CACGTG E-box than the retrieval-free prediction, providing a paralog-level plausibility check rather than direct validation of SOHLH1 binding. To calibrate this interpretation, we examined held-out bHLH TFs and/or a masked SOHLH2 positive control, which showed that predictions in the SOHLH1 confidence range <summary of validation>. Finally, scanning germ-cell-related regulatory regions with the SOHLH1 motif nominated candidate target elements for future experimental testing.

Together, this case illustrates how TFScope can convert the amino-acid sequence of an orphan human TF into a confidence-ranked, experimentally testable motif hypothesis without requiring a protein-DNA structure.
```

---

## 5. Files expected at completion

```text
configs/case_study_sohlh1.yaml

scripts/case_study/build_orphan_tf_tables.py
scripts/case_study/run_sohlh1_inference.sh
scripts/case_study/compute_sohlh1_metrics.py
scripts/case_study/plot_confidence_distribution.py
scripts/case_study/heldout_bhlh_validation.py
scripts/case_study/run_sohlh2_masked_control.sh
scripts/case_study/plot_sohlh1_logo_comparison.py
scripts/case_study/scan_germ_cell_promoters.py
scripts/case_study/assemble_figure5.py

results/sohlh1_case/leakage/SOHLH1_leakage_audit.tsv
results/sohlh1_case/predictions/SOHLH1_noRAG.pwm.tsv
results/sohlh1_case/predictions/SOHLH1_RAG_LGO.pwm.tsv
results/sohlh1_case/predictions/SOHLH1_prediction_metrics.tsv
results/sohlh1_case/predictions/SOHLH1_retrieved_neighbors.tsv
results/sohlh1_case/confidence/confidence_calibration_bins.tsv
results/sohlh1_case/validation/heldout_bhlh_confidence_accuracy.tsv
results/sohlh1_case/validation/SOHLH2_masked_control_metrics.tsv
results/sohlh1_case/targets/top_candidate_targets.tsv
results/sohlh1_case/manuscript/result5_draft.md

figures/figure5/panel_a_confidence_distribution.pdf
figures/figure5/panel_b_sohlh1_target_card.pdf
figures/figure5/panel_c_sohlh1_logo_comparison.pdf
figures/figure5/panel_d_heldout_bhlh_validation.pdf
figures/figure5/panel_d_sohlh2_masked_control.pdf
figures/figure5/panel_e_target_enrichment.pdf
figures/figure5/Figure5_SOHLH1_case_study.pdf
figures/extended/extended_sohlh1_attention_map.pdf
```

---

## 6. Final interpretation rules

Use the following claim levels.

### Allowed strong claims

- TFScope generates a motif hypothesis for SOHLH1 from sequence alone.
- No protein-DNA complex structure is required.
- Leave-gene-out retrieval sharpens a weak sequence-derived prior into an E-box-like motif.
- SOHLH1 is a medium-confidence, retrieval-supported E-box candidate.
- The predicted motif is biologically plausible because it resembles bHLH/E-box grammar and SOHLH2 paralog specificity.
- The case nominates candidate regulatory elements for future experimental validation.

### Claims to avoid

- TFScope experimentally validates the SOHLH1 motif.
- SOHLH1 definitively binds the predicted motif.
- SOHLH2 similarity is independent validation unless SOHLH2 is fully removed from training and retrieval.
- The noRAG pathway alone discovers the SOHLH1 E-box motif, unless future results show this.
- Attention proves causal recognition residues.

---

## 7. Quality-control checklist

Before using this case in the manuscript, verify:

- [ ] SOHLH1 same-gene leakage audit is clean.
- [ ] SOHLH2 is not used as a retrieved neighbor for SOHLH1.
- [ ] RAG/noRAG predictions are both saved and shown transparently.
- [ ] Confidence score is calibrated on held-out known TFs, not arbitrarily assigned.
- [ ] Figure 5a shows SOHLH1 relative to held-out known TFs and orphan TFs.
- [ ] Panel c does not duplicate separate PWM logo panels.
- [ ] Panel d provides actual computational validation/calibration.
- [ ] Any promoter/enhancer analysis is worded as nomination, not proof of binding.
- [ ] Final text calls the result a medium-confidence hypothesis if confidence remains ~0.61.
- [ ] All output tables are saved for supplement/reproducibility.

