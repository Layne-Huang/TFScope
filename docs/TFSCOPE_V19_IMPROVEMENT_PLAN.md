# TFScope V19 Improvement Plan

## Objective

Improve TFScope's sequence-to-PWM accuracy, canonical motif registration,
retrieval robustness, mechanistic grounding, and confidence calibration without
increasing model capacity prematurely.

The current evidence supports four conclusions:

1. Retrieval is useful, but donor construction and evaluation require stricter
   leakage controls.
2. Motif registration is the largest unresolved model error.
3. More experts, finer family taxonomies, and generic pretraining have not
   improved the primary benchmark.
4. Structure-derived residue-to-base supervision and assay-aware labels remain
   underused.

This plan incorporates the recommendations in
`docs/deep-research-report.md`, the current architecture and results, the
repository implementation, and a direct audit of the cluster40 split and
retrieval index.

## Implementation Status

As of June 13, 2026, E0 through E3 are implemented:

- `scripts/build_cluster40_split.py` creates deterministic connected groups
  across genes, UniProt accessions, 40%-identity clusters, exact sequences,
  and specific motif source IDs.
- Source labels reused across many genes, such as `H13CORE.0`, `RCADE`, and
  `MEME`, are not treated as motif identities. A source ID is considered
  specific when it maps to at most two genes.
- `scripts/build_nn_index.py` builds a K=16 train-only donor bank and excludes
  same-record, same-gene, same-UniProt, same-specific-source, and exact-sequence
  matches.
- `scripts/audit_split_hygiene.py` independently audits both artifacts and can
  fail automated runs on leakage.
- `scripts/train_v19_e1_baselines.sbatch` launches matched no-RAG and K=16 RAG
  baselines for seeds 42, 43, and 44 on the clean benchmark.
- `--gene-balanced-sampling` enables the deterministic E2 sampler while
  retaining the original number of training examples per epoch.
- E2 seed-42 no-RAG and K=16 RAG models were trained with three-GPU DDP,
  BF16, and TF32. RAG improved gene-macro aligned r from 0.4963 to 0.5092 and
  reduced aligned DeepPBS-scale MAE from 0.8358 to 0.8201.
- `scripts/audit_pwm_registration.py` implements E3 and emits pair-level
  offset/RC alignments, gene classifications, and train-only consensus-relative
  registration labels.
- E3 found median aligned r 0.9706 versus fixed-frame r 0.2665 across 6,005
  valid same-gene pairs. The train-only label artifact contains 1,746 labels
  across 565 genes.

Generated benchmark artifacts:

| Artifact | Path |
|---|---|
| Clean split | `data/processed/splits/cluster40_clean/split.json` |
| Connected-group manifest | `data/processed/splits/cluster40_clean/group_manifest.json` |
| Full hygiene audit | `data/processed/splits/cluster40_clean/full_hygiene_report.json` |
| Train-only K=16 index | `data/processed/tf_nn_index_cluster40_clean.json` |
| Index exclusion manifest | `data/processed/tf_nn_index_cluster40_clean.json.manifest.json` |
| E3 audit summary | `results/v19_e3_registration/audit_summary.json` |
| E3 train-only relative labels | `results/v19_e3_registration/relative_registration_anchors_train.tsv` |

The resulting split contains 2,947 training rows, 686 validation rows, and 614
test rows. The audit reports zero cross-split gene, UniProt, specific-source,
or exact-sequence overlaps and zero invalid retrieval edges. All 4,247 queries
have eligible donors.

---

## 1. Benchmark Repair

Benchmark reconstruction must precede new architecture comparisons.

### 1.1 Grouped split construction

Build indivisible groups using:

- gene symbol;
- UniProt accession;
- sequence-identity cluster;
- sequence-augmentation lineage;
- duplicate motif `source_id`;
- optionally, motif-similarity cluster for a stricter benchmark.

Assign each complete group to exactly one of train, validation, or test.

The current cluster40 artifacts contain cross-split genes and same-gene
retrieval. In particular, most validation rows have a same-gene donor among
their top three candidates. This makes validation-based checkpoint selection
optimistic and likely contributes to the large validation-to-test gap.

### 1.2 Retrieval donor policy

Use the following closed-book rules:

| Query split | Allowed donors |
|---|---|
| Train | Other training genes |
| Validation | Training genes only |
| Test | Training genes only |

Always exclude:

- the query record;
- the query gene;
- matching UniProt accessions;
- records from the same augmentation lineage;
- forbidden duplicate motif sources.

Every index must include a machine-readable exclusion manifest.

After model selection, a final train+validation model may use a train+validation
bank only when evaluated against a new untouched test set.

### 1.3 Evaluation units

Report:

- row-level metrics for compatibility with existing results;
- gene-macro metrics as the primary measure;
- per-family macro metrics;
- paired bootstrap confidence intervals;
- paired permutation tests for model comparisons.

Use three seeds during development and five seeds for final selected models.

### 1.4 Acceptance criteria

- Zero gene, UniProt, source-ID, or augmentation-lineage leakage.
- Zero same-gene retrieval.
- Deterministic split and index generation.
- Clean no-RAG and RAG baselines reproduced before V19 experiments.

The clean RAG gain may be smaller than the current reported gain. A smaller
leakage-controlled gain is scientifically preferable.

---

## 2. Target Representation And Sampling

The dataset contains multiple assay-derived PWM records for many genes. These
records should not be treated as independent, equally reliable targets.

### 2.1 Gene-balanced training

Sample genes uniformly, then sample one motif record for each selected gene per
epoch. Preserve source, assay, quality, and subtype metadata.

This prevents heavily studied TFs with many records from dominating gradient
updates.

### 2.2 Motif consistency audit

For every gene with multiple records, calculate:

- pairwise aligned PWM similarity;
- offset and reverse-complement disagreement;
- motif-length and information-content variation;
- assay and source disagreement.

Classify each gene as:

1. consistent single motif;
2. registration-discordant;
3. genuine subtype or multimodal;
4. noisy or contradictory.

### 2.3 Assay-aware targets

Start with a gene-level consensus PWM plus assay/source embeddings. Later,
model the observed record as:

```text
P(observed PWM | latent gene motif, assay, source)
```

A later probabilistic head may predict a Dirichlet mean and concentration for
each PWM column.

### 2.4 Acceptance criteria

- Gene-macro performance does not regress.
- Source-specific CE/KL and calibration improve.
- No single high-record gene dominates training batches.

---

## 3. Biological Motif Registration

Canonical registration cannot be solved reliably by a standalone offset
classifier because most records lack trustworthy offset/orientation labels and
the PWM representation has a reverse-complement symmetry.

Use semi-supervised latent registration anchored by structural and
family-specific evidence.

### 3.1 Registration states

Enumerate:

```text
orientation in {forward, reverse-complement}
offset in [-10, 10]
```

For unanchored samples, marginalize over the 42 possible states:

\[
L_{\mathrm{register}}
=
-\log \sum_r p_\theta(r \mid x)
\exp[-L_{\mathrm{PWM}}(T_r(\hat P), P)]
\]

### 3.2 Anchored registration examples

Derive reliable states from:

- PDB protein-DNA contact maps;
- family canonical contact maps;
- Pfam/HMM-aligned recognition slots;
- rCLAMPS-style mappings for homeodomains and C2H2 zinc fingers;
- highly consistent same-gene motif collections.

These anchors break the otherwise unresolved orientation symmetry.

### 3.3 Contact-frame decoder

Predict motif positions in a family-aligned DNA-contact frame. A register head
then maps this internal frame to exported PWM coordinates.

The register head should predict:

- offset distribution;
- orientation probability;
- register entropy;
- optional motif boundary distribution.

### 3.4 Required ablations

| Run | Description |
|---|---|
| R0 | Existing fixed canonicalization |
| R1 | Latent alignment without anchors |
| R2 | Latent alignment with structural anchors |
| R3 | Contact-frame decoder plus register head |

R1 is an important control: it may improve motif content but is not expected to
resolve canonical orientation by itself.

### 3.5 Acceptance criteria

- `canon_fixed_r` improves by at least 0.03.
- Gate-r regression is no worse than 0.005.
- Register accuracy improves on structurally anchored examples.
- Evaluation no longer depends on oracle alignment as the headline result.

---

## 4. Motif-Transfer Retrieval

Current retrieval ranks proteins by pooled ESM cosine similarity. V19 should
rank candidates by expected PWM transfer quality.

### 4.1 Candidate generation

Retrieve 16 candidates using frozen ESM residue representations under the
closed-book donor policy.

### 4.2 Transfer-specific reranking

Train a reranker using:

- query and donor recognition-residue embeddings;
- token-level late interaction;
- family compatibility;
- differences at aligned recognition slots;
- donor motif quality, assay, and source;
- pooled embedding similarity.

The target is aligned PWM transfer quality, not protein similarity.

Use cross-fitting so the query gene or its target motif cannot supervise its
own candidate score.

### 4.3 Hard negatives

Prioritize donors with:

- high ESM similarity;
- the same family;
- low aligned PWM similarity.

These examples directly teach the reranker not to overtrust paralog-like
proteins with different specificity.

### 4.4 Retrieval fusion

Predict:

- per-donor trust;
- per-position donor trust;
- donor offset/orientation distribution;
- all-donors-bad probability.

Replace the single motif-wide beta gate with position-wise retrieval gates.
The model must be able to transfer only part of a donor motif.

### 4.5 Robustness tests

Evaluate:

- removal of the top-ranked donor;
- masking of same-family donors;
- injected high-similarity bad donors;
- shuffled donor PWMs;
- empty donor bank;
- donor-quality performance curves.

### 4.6 Acceptance criteria

Across at least three seeds:

- Gene-macro gate-r improves by at least 0.015.
- Donor transfer-quality AUC improves.
- Canon-r does not decrease by more than 0.01.
- Performance degrades gracefully when the top donor is removed.

---

## 5. Structure As Privileged Training Information

Structure should supervise the sequence model during training without becoming
an inference requirement.

Existing repository assets include:

- 1,087 protein-chain contact maps;
- 455 PDB structures;
- 140 mapped TF identities;
- 542 PWM-aligned contact targets.

All validation/test genes must be excluded from structural pretraining inputs.

### 5.1 Pretraining tasks

1. Predict DNA-contacting residues.
2. Predict residue-to-DNA-position contact maps.
3. Predict family-aligned recognition slots.
4. Predict base preference from residue identity at each slot.
5. Distill interface-confidence or structural features where available.

Run this as a dedicated pretraining stage before PWM fine-tuning, rather than
only as a low-weight auxiliary loss on sparse examples.

### 5.2 Recognition-code residual

Use a family-aware recognition code:

\[
\Delta z_{j,b}
=
\sum_i A_{j,i}G(f,h_i,a_i,b)
\]

where:

- `f` is the DBD family;
- `h_i` is the aligned recognition slot;
- `a_i` is the amino-acid identity;
- `G` is a low-rank amino-acid/base interaction table.

Center `G` across bases for identifiability, then add a small neural residual
for non-additive interactions.

### 5.3 Sparse attention

Test sparse attention only after contact pretraining:

- softmax baseline;
- entmax15;
- learnable-alpha entmax.

Include shuffled-contact and random-contact controls. Sparse attention is not
evidence of biological correctness unless it is enriched on held-out contacts.

### 5.4 Mutation evaluation

For contact mutations, compare:

\[
\Delta z
=
G(f,h,a_{\mathrm{mut}},b)
-
G(f,h,a_{\mathrm{WT}},b)
\]

Evaluate:

- contact versus non-contact mutation sensitivity;
- localization of PWM changes;
- direction of known specificity changes;
- stability under non-contact mutations.

### 5.5 Acceptance criteria

- Contact enrichment exceeds random and shuffled controls.
- Mutation sensitivity improves.
- Canon-r improves by at least 0.02, or held-out contact-map performance
  improves substantially without PWM regression.

---

## 6. Confidence And Abstention

Separate three distinct confidence quantities:

1. donor transfer confidence;
2. registration confidence;
3. final PWM accuracy confidence.

Useful confidence features include:

- donor trust distribution;
- RAG/no-RAG disagreement;
- register entropy;
- ensemble variance;
- predicted Dirichlet concentration;
- distance from known recognition-code combinations;
- structural/contact confidence.

Use cross-fitted global calibration with hierarchical family adjustment.
Family-specific calibration should only be used when the calibration sample
size is adequate.

Report:

- ECE and adaptive ECE;
- Brier score;
- calibration slope and intercept;
- 90% conformal coverage;
- accuracy-versus-abstention curves.

Confidence calibration must use strictly held-out data and must not reuse the
optimistic current validation retrieval bank.

---

## 7. Simplification And Later Extensions

These experiments should not block the core V19 work.

### 7.1 Dense alternative to MoE

Compare the current routed MoE against:

- a shared SwiGLU block;
- family-conditioned low-rank adapters;
- family-conditioned recognition-code parameters.

If the dense model performs within 0.01 gate-r of the MoE, prefer the simpler
model for deployment.

### 7.2 Retrieval-free student

Distill the clean RAG model into a no-RAG student by matching:

- PWM distribution;
- contact-frame representation;
- uncertainty outputs.

Evaluate this model specifically on donor-poor and true orphan TFs.

### 7.3 Multi-hypothesis motifs

Use a multi-motif head only for genes classified as genuinely multimodal.
Train it with permutation-invariant matching between predicted and observed
subtypes.

### 7.4 Partner-aware specificity

For bHLH, bZIP, and nuclear-receptor families, add:

- partner sequence or homodimer indicator;
- half-site motif decoder;
- spacing and orientation distribution.

Partner-dependent families should not be forced indefinitely into a
single-protein, single-PWM representation.

---

## 8. Experiment Sequence

| Experiment | Change | Main decision metric |
|---|---|---|
| E0 | Clean grouped split and train-only retrieval | Leakage audit |
| E1 | Clean no-RAG and RAG baselines | Gene-macro gate-r |
| E2 | Gene-balanced sampler | Gene-macro stability |
| E3 | Registration-consistency audit | Anchored-label coverage |
| E4 | Latent register loss | Content versus canon-r |
| E5 | Structurally anchored register head | Canon-r |
| E6 | K=16 transfer reranker | Gate-r and donor AUC |
| E7 | Per-position retrieval trust | Gate-r and robustness |
| E8 | Contact-map pretraining | Contact enrichment |
| E9 | Recognition-code residual | Canon-r and mutations |
| E10 | Entmax ablation | Contact precision |
| E11 | Assay-aware uncertainty | CE, KL, calibration |
| E12 | Confidence and conformal abstention | ECE and coverage |
| E13 | Dense replacement for MoE | Accuracy and complexity |
| E14 | RAG-to-noRAG distillation | Donor-poor subset |

Each experiment must use the same clean split, donor policy, evaluation code,
and gene-level reporting.

### Execution status, June 15, 2026

- Corrected E2 and E5b checkpoints now include trained LoRA tensors.
- E6 improves aligned donor reranking but not final PWM fusion.
- E7 position-wise retrieval gating was rejected on validation
  (`0.5683` best oracle-r versus E6 `0.5703`).
- A de-novo-frame donor-alignment variant was also rejected
  (`0.5656` best through epoch 6).
- The current candidate is a validation-locked, family-specific composition of
  the corrected E2 frame and E5b motif content. Held-out test gene-macro
  panel-r improves from `0.4945` to `0.5454`; paired gene bootstrap 95% CI for
  the delta is `[+0.0305, +0.0724]`.
- This candidate is an ensemble and must be reported alongside its two-model
  inference cost and the corrected single-model E2 baseline.
- The user fixed the final experiment scope to seed 42. The general multi-seed
  recommendation below remains scientifically preferable, but no additional
  seeds should be launched without a new user decision; the manuscript must
  state the single-seed limitation.

---

## 9. V19 Scope

The core V19 release should contain:

1. leakage-controlled gene-grouped benchmark;
2. gene-balanced training;
3. structurally anchored latent registration;
4. motif-transfer retrieval reranker;
5. per-position retrieval trust;
6. separate confidence outputs and abstention support.

Contact-code pretraining may be released as V19.1 if it is not ready for the
first V19 benchmark.

The following should not block V19:

- a larger backbone;
- more MoE experts;
- LoRA+;
- structure-informed backbone adapters;
- multi-hypothesis decoding;
- partner-aware modeling;
- assisted external motif banks.

---

## 10. Reporting Policy

Maintain separate protocols:

- **Closed-book:** donor bank contains only permitted training records.
- **Assisted:** external JASPAR, HOCOMOCO, CIS-BP, or other motif banks are
  available.

Assisted results must never be mixed into the closed-book leaderboard.

Treat the reproducible unit as:

```text
model + split + donor bank + exclusion manifest + calibration artifact
```

Version and checksum:

- processed datasets;
- split manifests;
- retrieval embeddings and indices;
- contact targets;
- model configuration and checkpoint;
- evaluator version;
- calibration data and parameters.

The primary model metrics should be:

- gene-macro gate-r;
- gene-macro canon-r;
- MAE and top-1 accuracy;
- calibration and abstention metrics.

Oracle-aligned scores remain useful diagnostics, but should not be the sole
headline result.
