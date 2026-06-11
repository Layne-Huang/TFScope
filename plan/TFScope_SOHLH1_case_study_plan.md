# TFScope SOHLH1 Case Study Plan for Codex / Claude Code

## Goal

Build a **Nature Methods-style biological application case study** showing that TFScope can nominate a DNA-binding motif for an **uncharacterized human transcription factor from sequence alone**, without requiring a protein-DNA complex structure.

Primary candidate:

- **SOHLH1**: spermatogenesis and oogenesis specific basic helix-loop-helix 1.
- Proposed biological framing: an orphan / poorly characterized germ-cell bHLH transcription factor with disease and fertility relevance.
- Proposed comparison: its paralog **SOHLH2** or nearest related bHLH motifs, if available in HumanTFs / CIS-BP / JASPAR.

The result should support a manuscript section like:

> **TFScope nominates the binding specificity of an orphan germ-cell transcription factor from sequence alone.**

The case study should be implemented as a reproducible analysis pipeline that can be run from the repository and produce tables, motif logos, QC plots, and one Figure-5-style panel set.

---

## Important conceptual boundary

Do **not** treat this case study as another benchmark.

Existing benchmark sections already evaluate:

1. DeepPBS split performance.
2. cluster40 out-of-distribution performance.
3. leave-family-out generalization.
4. recognition-code / attention behavior.

This case study should instead answer:

> Given a real human TF with no curated motif or no protein-DNA complex, what motif hypothesis does TFScope generate, and how can we prioritize it for experimental testing?

Use benchmark results only to support confidence calibration and reviewer trust. Do not re-report cluster40 or leave-family-out as the main result here.

---

## Expected final deliverables

Create the following outputs:

```text
results/case_study_sohlh1/
├── README.md
├── logs/
├── metadata/
│   ├── humantfs_full_database.tsv
│   ├── humantfs_motif_list.tsv
│   ├── candidate_tf_metadata.tsv
│   ├── sohlh1_leakage_audit.tsv
│   ├── sohlh1_sequence_and_dbd.tsv
│   └── sohlh1_retrieval_neighbors.tsv
├── predictions/
│   ├── SOHLH1_noRAG.pwm.tsv
│   ├── SOHLH1_RAG_LGO.pwm.tsv
│   ├── SOHLH1_noRAG.meme
│   ├── SOHLH1_RAG_LGO.meme
│   ├── SOHLH1_attention.npy
│   └── SOHLH1_prediction_summary.tsv
├── validation/
│   ├── SOHLH2_known_motif_or_nearest_bHLH_motif.meme
│   ├── sohlh1_vs_sohlh2_similarity.tsv
│   ├── bhlh_reference_motif_similarity.tsv
│   └── optional_promoter_enrichment.tsv
├── figures/
│   ├── fig5a_atlas_screening.pdf
│   ├── fig5b_sohlh1_metadata.pdf
│   ├── fig5c_sohlh1_predicted_logos.pdf
│   ├── fig5d_sohlh1_paralog_or_reference_comparison.pdf
│   ├── fig5e_attention_or_promoter_enrichment.pdf
│   └── fig5_combined.pdf
└── manuscript/
    ├── result5_draft.md
    ├── methods_case_study_draft.md
    └── figure5_caption.md
```

---

## Repository assumptions

Before coding, inspect the repository and adapt names to existing conventions.

Expected existing components may include:

- TFScope model checkpoint loading.
- TFScope inference script.
- PWM evaluation utilities.
- Motif logo plotting utilities.
- Retrieval index utilities.
- Attention extraction utilities.
- Existing cluster40 / DeepPBS evaluation scripts.

Do not rewrite core model code unless necessary. Prefer writing thin wrappers around existing scripts.

If exact script names differ, add compatibility wrappers under:

```text
analysis/case_studies/sohlh1/
```

---

## Recommended directory layout for new code

```text
analysis/case_studies/sohlh1/
├── README.md
├── config.yaml
├── 00_download_humantfs.py
├── 01_select_candidate_tf.py
├── 02_fetch_sequences_and_domains.py
├── 03_leakage_audit.py
├── 04_run_tfscope_inference.py
├── 05_compute_prediction_confidence.py
├── 06_compare_reference_motifs.py
├── 07_optional_promoter_enrichment.py
├── 08_plot_figure5_panels.py
├── 09_write_manuscript_snippets.py
└── utils/
    ├── motif_io.py
    ├── motif_similarity.py
    ├── logo_plotting.py
    ├── humantfs_utils.py
    └── tfscope_wrappers.py
```

---

## Config file

Create `analysis/case_studies/sohlh1/config.yaml`.

Use this template:

```yaml
case_study:
  primary_tf: "SOHLH1"
  paralog_tf: "SOHLH2"
  tf_family_expected: "bHLH"
  organism: "Homo sapiens"
  output_dir: "results/case_study_sohlh1"

models:
  # Use cluster40 for confidence calibration / leakage-aware validation.
  cluster40_checkpoint: "checkpoints/tfscope_cluster40.pt"

  # Use production checkpoint for final atlas / case-study motif prediction.
  # If absent, use cluster40 checkpoint and mark results as conservative.
  production_checkpoint: "checkpoints/tfscope_production.pt"

retrieval:
  index_path: "data/retrieval/lgo_index.faiss"
  metadata_path: "data/retrieval/retrieval_metadata.tsv"
  exclude_same_gene: true
  top_k: 3

training_data:
  train_gene_symbols: "data/tfscope_training/train_gene_symbols.txt"
  retrieval_gene_symbols: "data/tfscope_training/retrieval_gene_symbols.txt"
  train_motif_metadata: "data/tfscope_training/train_motif_metadata.tsv"

human_tfs:
  download_dir: "data/external/humantfs_v1_01"
  full_database_csv: "data/external/humantfs_v1_01/human_tfs_full.csv"
  motif_list_csv: "data/external/humantfs_v1_01/human_tf_motif_list.csv"
  motif_pwm_zip: "data/external/humantfs_v1_01/human_tf_motif_pwms.zip"

sequence_sources:
  # Prefer existing local sequence database if available.
  uniprot_fasta: "data/external/uniprot/human_reviewed.fasta"
  allow_web_download: false

motif_processing:
  active_gate_threshold: 0.5
  ic_threshold: 0.25
  min_motif_length: 4
  max_motif_length: 20

confidence:
  success_threshold_r: 0.60
  high_confidence_threshold: 0.70
  medium_confidence_threshold: 0.40

optional_promoter_enrichment:
  enabled: false
  genome_fasta: "data/genomes/hg38.fa"
  gene_annotation_gtf: "data/genomes/gencode.vXX.annotation.gtf"
  target_gene_list: "data/case_study_sohlh1/germ_cell_or_sohlh1_related_genes.txt"
  promoter_window_bp: 2000
  n_background_sets: 100
```

---

## Step 0 — Preflight checks

Implement `00_download_humantfs.py` and a small preflight function.

### Tasks

1. Create output directories.
2. Check availability of model checkpoints.
3. Check availability of retrieval index.
4. Download or load HumanTFs full database and motif list.
5. Save metadata snapshots under `results/case_study_sohlh1/metadata/`.

### HumanTFs files to use

The HumanTFs download page provides:

- Full Database.
- Full Database README.
- TF gene names.
- Motif list.
- Motif list README.
- Motif PWM files.

Use official HumanTFs files if local copies are missing.

### Acceptance criteria

- `humantfs_full_database.tsv` exists.
- `humantfs_motif_list.tsv` exists.
- Script prints number of human TFs and number with `Motif status == "No motif"`.
- Script exits with a clear error if required files are missing.

---

## Step 1 — Candidate selection and metadata verification

Implement `01_select_candidate_tf.py`.

### Goals

Verify whether SOHLH1 is a good case-study TF, not just assume it.

For SOHLH1, extract:

```text
gene_symbol
ensembl_id
entrez_gene_id
dbd
is_tf
tf_assessment
binding_mode
motif_status
interpro_ids
pdb_id
tested_by_ht_selex
tested_by_pbm
conditional_binding_requirements
final_notes
final_comments
```

For SOHLH2, extract the same metadata and any motif IDs from HumanTFs motif list.

### Logic

1. Load HumanTFs full database.
2. Select rows where `HGNCsymbol == "SOHLH1"`.
3. Select rows where `HGNCsymbol == "SOHLH2"`.
4. Confirm:
   - SOHLH1 exists.
   - SOHLH1 is classified as a TF or likely sequence-specific TF.
   - SOHLH1 has a recognizable bHLH DBD.
   - SOHLH1 lacks a high-quality motif or is marked `No motif`.
   - SOHLH1 has no PDB protein-DNA complex, if possible.
5. Confirm SOHLH2 status:
   - If SOHLH2 has a motif, use it as paralog reference.
   - If SOHLH2 has no motif, fall back to nearest bHLH motif references.

### Fallback candidates

If SOHLH1 fails preflight, programmatically search HumanTFs for candidates satisfying:

```text
Is TF? == Yes
AND Motif status == No motif
AND DBD contains bHLH or GATA or Forkhead
AND Binding mode is not "Low specificity DNA binding"
AND no PDB protein-DNA complex
```

Rank candidates by:

1. Clear DBD annotation.
2. Has paralog with known motif.
3. Biological relevance in Entrez/NCBI summary.
4. Not present in TFScope training/retrieval same-gene data.
5. Short / compact DBD family preferred over long C2H2 for primary Result 5.

Save fallback list to:

```text
results/case_study_sohlh1/metadata/fallback_candidate_tfs.tsv
```

### Acceptance criteria

- `candidate_tf_metadata.tsv` exists.
- It contains SOHLH1 and SOHLH2 metadata.
- It includes a `case_status` column with one of:
  - `primary_candidate_pass`
  - `primary_candidate_warning`
  - `primary_candidate_fail_use_fallback`
- It logs reasons for pass/warning/fail.

---

## Step 2 — Sequence and DBD extraction

Implement `02_fetch_sequences_and_domains.py`.

### Goals

Create the exact input needed for TFScope inference.

TFScope requires:

- protein sequence,
- DBD start/end coordinates,
- family label,
- optional DBD mask.

### Tasks

1. Obtain canonical SOHLH1 protein sequence.
2. Obtain canonical SOHLH2 protein sequence.
3. Obtain bHLH DBD coordinates:
   - Prefer local UniProt / InterPro / Pfam annotations.
   - If unavailable, use HumanTFs InterPro/Pfam fields.
   - If still unavailable, use sequence-domain scan from existing tools, if present.
4. Build TFScope input TSV.

### Output

```text
results/case_study_sohlh1/metadata/sohlh1_sequence_and_dbd.tsv
```

Expected columns:

```text
gene_symbol
uniprot_id
ensembl_id
entrez_gene_id
family
dbd_type
dbd_start
dbd_end
sequence
dbd_sequence
source_sequence
source_dbd
dbd_annotation_confidence
```

### Acceptance criteria

- SOHLH1 sequence is non-empty.
- DBD coordinates are within sequence length.
- bHLH DBD length is plausible.
- DBD mask can be generated.
- If DBD cannot be confidently identified, mark case as warning and do not silently continue.

---

## Step 3 — Leakage audit

Implement `03_leakage_audit.py`.

### Goals

Make the case study reviewer-safe.

Because TFScope has a retrieval branch and training data include multiple motif databases, confirm that SOHLH1 is not trivially represented in training or retrieval.

### Tasks

For SOHLH1 and SOHLH2:

1. Check if same gene appears in:
   - training gene symbols,
   - retrieval index metadata,
   - motif metadata,
   - benchmark test sets if available.
2. Check if same motif ID appears in training.
3. Check nearest training sequence identity if a sequence database exists.
4. Check whether any retrieval neighbor is the same gene and verify leave-gene-out exclusion.
5. Save top-k retrieval neighbors after LGO filtering.

### Output

```text
results/case_study_sohlh1/metadata/sohlh1_leakage_audit.tsv
results/case_study_sohlh1/metadata/sohlh1_retrieval_neighbors.tsv
```

### Acceptance criteria

- Same-gene SOHLH1 retrieval is excluded.
- If SOHLH1 appears in training, mark the case as leakage-risk and make the main result rely on no-RAG or choose fallback.
- If SOHLH2 is retrieved, report it transparently as paralog support, not as independent proof.

---

## Step 4 — TFScope inference

Implement `04_run_tfscope_inference.py`.

### Run two inference modes

1. **No retrieval**:
   - main sequence-intrinsic prediction.
   - best for claim: no protein-DNA structure and no homology motif transfer needed.

2. **Leave-gene-out retrieval**:
   - deployable mode.
   - use retrieval index, exclude same gene.
   - report retrieved neighbors.

### Commands to support

```bash
python analysis/case_studies/sohlh1/04_run_tfscope_inference.py \
  --config analysis/case_studies/sohlh1/config.yaml \
  --mode noRAG

python analysis/case_studies/sohlh1/04_run_tfscope_inference.py \
  --config analysis/case_studies/sohlh1/config.yaml \
  --mode RAG_LGO
```

### Outputs

```text
results/case_study_sohlh1/predictions/SOHLH1_noRAG.pwm.tsv
results/case_study_sohlh1/predictions/SOHLH1_RAG_LGO.pwm.tsv
results/case_study_sohlh1/predictions/SOHLH1_noRAG.meme
results/case_study_sohlh1/predictions/SOHLH1_RAG_LGO.meme
results/case_study_sohlh1/predictions/SOHLH1_attention.npy
results/case_study_sohlh1/predictions/SOHLH1_prediction_summary.tsv
```

### Prediction summary columns

```text
gene_symbol
model_checkpoint
inference_mode
retrieval_enabled
same_gene_excluded
active_motif_length
mean_gate_probability
mean_information_content
max_information_content
motif_entropy
top_base_sequence
nearest_retrieved_gene_1
nearest_retrieved_similarity_1
nearest_retrieved_gene_2
nearest_retrieved_similarity_2
nearest_retrieved_gene_3
nearest_retrieved_similarity_3
```

### Acceptance criteria

- noRAG and RAG_LGO predictions both complete.
- PWMs are valid probability matrices; each column sums to 1.
- Active motif length is within configured bounds.
- MEME files can be read by downstream motif tools.

---

## Step 5 — Compute prediction confidence

Implement `05_compute_prediction_confidence.py`.

### Required confidence components

Compute:

1. `rag_noRAG_similarity`
   - motif similarity between SOHLH1 noRAG and RAG_LGO predictions.
2. `gate_confidence`
   - mean gate probability over active motif columns.
3. `motif_information_content`
   - mean IC over active motif columns.
4. `retrieval_support`
   - top LGO retrieval similarity.
5. `family_prior`
   - bHLH expected reliability from cluster40 / leave-family-out summary if available.
6. `attention_coherence`
   - optional; use only if attention output is accessible.
7. `ensemble_similarity`
   - optional; use if multiple checkpoints or seeds exist.

### Confidence score

If an existing calibrated confidence model exists, use it.

Otherwise compute a transparent rule-based score:

```text
confidence_raw =
    0.40 * rag_noRAG_similarity
  + 0.20 * gate_confidence
  + 0.15 * motif_information_content_normalized
  + 0.15 * retrieval_support
  + 0.10 * family_prior
```

Normalize each component to 0-1.

Assign:

```text
High:   confidence >= 0.70
Medium: 0.40 <= confidence < 0.70
Low:    confidence < 0.40
```

If bHLH family-specific calibration from cluster40 exists, use that instead.

### Output

```text
results/case_study_sohlh1/predictions/SOHLH1_confidence.tsv
```

### Acceptance criteria

- Confidence score is present.
- Each component is present.
- Confidence class is present.
- Any missing component is explicitly marked `NA` and described in `README.md`.

---

## Step 6 — Reference motif comparison

Implement `06_compare_reference_motifs.py`.

### Goals

Evaluate whether SOHLH1 predicted motif is biologically plausible relative to:

1. SOHLH2 motif, if available.
2. bHLH family reference motifs.
3. Generic E-box motifs.
4. Nearest retrieved motif(s).

### Tasks

1. Extract SOHLH2 motif from HumanTFs motif list and PWM files if available.
2. If SOHLH2 motif is unavailable:
   - select nearest bHLH references from HumanTFs / CIS-BP / JASPAR.
3. Compute motif similarity:
   - oracle-aligned Pearson r,
   - IC-weighted r,
   - top-base agreement,
   - Tomtom-style q-value if available,
   - reverse-complement-aware similarity.
4. Compare SOHLH1 predicted top-base consensus to:
   - SOHLH2 motif,
   - canonical E-box `CANNTG`,
   - high-confidence bHLH motifs.

### Outputs

```text
results/case_study_sohlh1/validation/sohlh1_vs_sohlh2_similarity.tsv
results/case_study_sohlh1/validation/bhlh_reference_motif_similarity.tsv
```

### Acceptance criteria

- Similarity table includes noRAG and RAG_LGO.
- If SOHLH2 motif unavailable, output states that nearest bHLH reference motifs were used.
- The script does not hard-code SOHLH2 motif; it must parse or import it from source files.

---

## Step 7 — Optional promoter / regulatory enrichment

Implement `07_optional_promoter_enrichment.py`.

This step is optional and should be disabled by default.

### Goal

Provide supporting evidence that the predicted SOHLH1 motif nominates plausible regulatory targets, without claiming direct in vivo occupancy.

### Inputs

- Gene list related to germ-cell development, spermatogenesis, oogenesis, folliculogenesis, or SOHLH1 literature.
- Promoter sequences from hg38.
- GC-matched background promoters.
- SOHLH1 predicted motif.
- Reference motifs:
  - SOHLH2 if available,
  - generic bHLH / E-box,
  - shuffled motif.

### Analysis

1. Extract promoter regions, e.g. TSS ± 2 kb.
2. Scan with FIMO or internal PWM scanner.
3. Compare motif score distributions:
   - target gene promoters vs matched background.
4. Report:
   - enrichment fold,
   - AUROC,
   - Mann-Whitney or permutation p-value,
   - top candidate genes.

### Output

```text
results/case_study_sohlh1/validation/optional_promoter_enrichment.tsv
```

### Caution

Do not write that promoter enrichment proves SOHLH1 binding. It only nominates candidate regulatory elements for future testing.

---

## Step 8 — Plot Figure 5 panels

Implement `08_plot_figure5_panels.py`.

### Figure 5 panel design

#### Fig. 5a — HumanTFs / atlas screening

Purpose: show how SOHLH1 was selected.

Possible visual:

```text
Human TFs
→ sequence-specific TFs
→ no curated motif / orphan TFs
→ bHLH orphan candidates
→ SOHLH1 selected
```

This can be a flowchart or stacked bar.

#### Fig. 5b — SOHLH1 metadata panel

Show compact schematic:

```text
SOHLH1
- bHLH DBD
- germ-cell / fertility-related TF
- motif status: no motif or no high-quality curated motif
- no protein-DNA complex structure found
```

Use a domain schematic with DBD location.

#### Fig. 5c — TFScope predicted motifs

Show logos:

```text
SOHLH1 noRAG
SOHLH1 RAG_LGO
```

Add:

```text
confidence score
RAG/noRAG similarity
mean IC
gate confidence
```

#### Fig. 5d — Paralog/reference comparison

Show:

```text
SOHLH1 TFScope noRAG
SOHLH1 TFScope RAG_LGO
SOHLH2 known motif or nearest bHLH motif
generic E-box reference
```

Add motif similarity scores.

#### Fig. 5e — Mechanistic or optional biological support

Choose one depending on available outputs:

1. Attention map over SOHLH1 bHLH DBD.
2. Promoter enrichment of predicted motif in germ-cell target genes.
3. Top candidate regulatory elements / genes.

### Style requirements

- Nature-style clean plots.
- Avoid dense labels.
- Use vector PDF and high-resolution PNG.
- Keep panel text minimal.
- Save source data for every panel.

### Outputs

```text
results/case_study_sohlh1/figures/fig5a_atlas_screening.pdf
results/case_study_sohlh1/figures/fig5b_sohlh1_metadata.pdf
results/case_study_sohlh1/figures/fig5c_sohlh1_predicted_logos.pdf
results/case_study_sohlh1/figures/fig5d_sohlh1_paralog_or_reference_comparison.pdf
results/case_study_sohlh1/figures/fig5e_attention_or_promoter_enrichment.pdf
results/case_study_sohlh1/figures/fig5_combined.pdf
```

---

## Step 9 — Manuscript snippets

Implement `09_write_manuscript_snippets.py`.

Generate draft text files:

```text
results/case_study_sohlh1/manuscript/result5_draft.md
results/case_study_sohlh1/manuscript/methods_case_study_draft.md
results/case_study_sohlh1/manuscript/figure5_caption.md
```

### Result draft template

Use values from generated tables. Do not hard-code numeric results.

Suggested structure:

```markdown
### TFScope nominates the binding specificity of the orphan germ-cell transcription factor SOHLH1

Having established the generalization behavior of TFScope under leakage-controlled benchmarks, we next asked whether the model could be used as a sequence-only annotation engine for human TFs lacking curated motifs. Screening the HumanTFs catalogue identified SOHLH1, a bHLH transcription factor implicated in germ-cell development and infertility, as a candidate case study because [motif status], [DBD status], and [structure status].

TFScope predicted a [motif_length]-bp motif for SOHLH1 using amino-acid sequence and DBD annotation alone. The retrieval-free and leave-gene-out retrieval-augmented predictions showed [RAG/noRAG similarity], indicating [interpretation]. The predicted motif had mean information content [mean_IC] and gate confidence [gate_confidence], yielding a calibrated confidence score of [confidence_score] ([confidence_class]).

To assess biological plausibility, we compared the SOHLH1 prediction with [SOHLH2 / nearest bHLH reference motifs]. The predicted motif [matched/diverged from] the reference bHLH grammar, with [similarity metric]. These results nominate a testable DNA-binding specificity for SOHLH1 without requiring a protein-DNA complex structure.
```

### Methods draft template

Include:

- HumanTFs download and filtering.
- SOHLH1 selection criteria.
- sequence / DBD extraction.
- leakage audit.
- noRAG and RAG_LGO inference.
- motif similarity.
- confidence calculation.
- optional promoter enrichment.

### Figure caption template

Include each panel:

```markdown
Fig. 5 | Sequence-only motif nomination for SOHLH1.
a, HumanTFs screening workflow...
b, SOHLH1 domain and annotation summary...
c, TFScope predicted SOHLH1 motifs...
d, Comparison to SOHLH2 or bHLH reference motifs...
e, Attention map or promoter enrichment...
```

---

## Step 10 — README and reproducibility

Create `results/case_study_sohlh1/README.md`.

Must include:

1. How to run the pipeline.
2. Input data versions.
3. Model checkpoints used.
4. Whether production checkpoint or cluster40 checkpoint was used.
5. Whether SOHLH1 passed leakage audit.
6. Whether SOHLH2 motif was available.
7. How confidence was computed.
8. Which outputs support each Figure 5 panel.
9. Known limitations.

### Minimum run command

Create a master runner if appropriate:

```bash
python analysis/case_studies/sohlh1/run_all.py \
  --config analysis/case_studies/sohlh1/config.yaml
```

or document sequential commands.

---

## Key acceptance criteria for the whole case study

The case study is complete only if all of the following are true:

1. SOHLH1 metadata is verified from HumanTFs or another authoritative database.
2. SOHLH1 sequence and DBD coordinates are reproducibly defined.
3. Same-gene SOHLH1 leakage is audited.
4. TFScope noRAG prediction is generated.
5. TFScope RAG_LGO prediction is generated or explicitly skipped with reason.
6. RAG/noRAG motif similarity is computed.
7. Prediction confidence is computed.
8. SOHLH2 or nearest bHLH reference motif comparison is computed.
9. At least four Figure 5 panels are generated.
10. Manuscript draft text, methods, and caption are generated.
11. All numeric claims in drafts are pulled from output tables, not hard-coded.

---

## Reviewer-risk checklist

Before using this in the manuscript, answer these questions in `README.md`:

### Leakage

- Is SOHLH1 in training?
- Is SOHLH1 in retrieval index?
- Does HumanTFs / CIS-BP / JASPAR contain a same-gene SOHLH1 motif?
- Are same-gene retrieval hits excluded?

### Biological suitability

- Is SOHLH1 truly sequence-specific or only likely sequence-specific?
- Does it bind as a monomer/homodimer or require an obligate heteromer?
- Is a first-order PWM appropriate for this TF?
- Does the bHLH DBD annotation look reliable?

### Prediction quality

- Is the PWM sharp or flat?
- Do noRAG and RAG_LGO agree?
- Is confidence high enough for a main-text case?
- If confidence is medium or low, should this become a cautionary / prioritization example rather than a success case?

### Reference comparison

- Is SOHLH2 motif available?
- If SOHLH2 is used as reference, is it a direct experimental motif or inferred?
- If only a generic bHLH motif is used, avoid overclaiming paralog-specific validation.

### Interpretation

- Do not claim direct in vivo SOHLH1 occupancy.
- Do not claim experimental validation unless actual experimental data exist.
- Phrase output as a **testable motif hypothesis**.

---

## Suggested final manuscript claim

Use a conservative claim unless direct validation is added:

> TFScope nominated a high-confidence E-box-like binding motif for SOHLH1 from protein sequence alone, providing a testable specificity hypothesis for an orphan germ-cell bHLH transcription factor without requiring a protein-DNA complex structure.

If RAG/noRAG disagree:

> TFScope nominated a retrieval-sensitive motif hypothesis for SOHLH1, illustrating how the atlas can flag orphan TFs whose specificity requires experimental profiling.

If SOHLH1 fails QC:

> TFScope screening identified [fallback TF] as a cleaner orphan-TF case study, while SOHLH1 was retained as a lower-confidence candidate in the supplementary atlas.

---

## Notes on source context

The manuscript currently establishes that TFScope:

- predicts PWMs directly from TF amino-acid sequence without structural input;
- uses ESM-2, DBD/global pooling, family-conditioned MoE, contact-aware attention, position gates, and optional leave-gene-out retrieval;
- matches or exceeds DeepPBS on the DeepPBS blind split;
- uses cluster40 and leave-family-out settings for leakage-controlled generalization;
- identifies long C2H2 arrays as a frontier;
- uses cross-attention to recover known recognition residues in KLF4 and MyoD.

This case study should connect those existing results to a concrete biological application rather than duplicate the benchmark sections.
