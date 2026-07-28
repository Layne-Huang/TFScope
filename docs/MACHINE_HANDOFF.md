# TFScope Machine Handoff

This file is the durable project memory for continuing TFScope development on
another machine. It records the current architecture assessment, implemented
benchmark repairs, generated artifacts, verification commands, and next
experiments.

## Current Objective

Develop TFScope V19 with improvements to:

1. leakage-controlled evaluation;
2. gene-balanced learning;
3. canonical motif registration;
4. retrieval based on motif-transfer quality;
5. structure-supervised recognition residues;
6. calibrated confidence and abstention.

The full rationale and experiment sequence are in
`docs/TFSCOPE_V19_IMPROVEMENT_PLAN.md`. The previous architecture and measured
results are in `docs/ARCHITECTURE_AND_RESULTS.md`. The external research advice
is preserved in `docs/deep-research-report.md`.

## Implemented Work

### E0: benchmark repair

Implemented:

- `src/tfscope/data/split_hygiene.py`
- `scripts/build_cluster40_split.py`
- `scripts/build_nn_index.py`
- `scripts/audit_split_hygiene.py`
- `tests/test_split_hygiene.py`

The clean split groups records transitively by:

- gene symbol;
- UniProt accession;
- CD-HIT 40% sequence cluster;
- exact sequence;
- specific motif source ID.

A source ID is considered a motif identity only when it maps to at most two
genes. Broad method labels such as `H13CORE.0`, `RCADE`, and `MEME` are not used
as identities because they otherwise connect most of the dataset.

The retrieval bank contains training records only. For every query it excludes:

- the query record;
- the same gene;
- the same UniProt accession;
- the same specific source ID;
- the same exact sequence.

Current clean split:

| Split | Rows | Genes | UniProt IDs |
|---|---:|---:|---:|
| Train | 2,947 | 914 | 899 |
| Validation | 686 | 234 | 231 |
| Test | 614 | 197 | 192 |

The independent audit reports zero split overlaps and zero invalid retrieval
edges. All 4,247 queries have eligible K=16 donors.

### E1/E2 baselines

`scripts/train_v19_e1_baselines.sbatch` launches six matched baseline runs:

- no-RAG seeds 42, 43, and 44;
- train-only K=16 RAG seeds 42, 43, and 44.

The selected E2 seed-42 models were trained with gene-balanced sampling,
three-GPU DDP, BF16, TF32, and global batch size 96. On the clean held-out test
split:

| Model | Best epoch | Gene-macro aligned r | Gene-macro aligned MAE |
|---|---:|---:|---:|
| E2 no-RAG | 90 | 0.4963 | 0.8358 |
| E2 RAG K=16 | 140 | **0.5092** | **0.8201** |

The selected checkpoint is:

```text
/data1/leihuang/project/TFScope/checkpoints/v19_e2_gene_balanced_bf16_ddp3/rag_seed42/ckpt_best.pt
```

### E2 support

`scripts/train.py --gene-balanced-sampling` enables deterministic uniform gene
sampling while preserving the original number of examples per epoch. For each
sampled gene, one of its motif records is selected.

### E3 registration-consistency audit

Implemented:

- `src/tfscope/data/registration_audit.py`
- `scripts/audit_pwm_registration.py`
- `tests/test_registration_audit.py`

The audit collapses exact repeated motif rows by gene, source ID, and PWM
fingerprint, then searches offset and reverse-complement states for every
same-gene motif pair. It produces:

```text
results/v19_e3_registration/audit_summary.json
results/v19_e3_registration/pairwise_alignments.tsv
results/v19_e3_registration/gene_consistency.tsv
results/v19_e3_registration/relative_registration_anchors.tsv
results/v19_e3_registration/relative_registration_anchors_train.tsv
```

Key results:

- 4,247 rows reduce to 3,935 unique motif records after collapsing 312 repeats.
- 940 genes have multiple unique motif records.
- Across 6,005 valid pairs, median aligned r is 0.9706 but median fixed-frame r
  is only 0.2665.
- 37.98% of valid pairs require reverse complementation and 50.17% require a
  nonzero shift.
- 649 genes are registration-discordant, 167 are fixed-frame consistent, 42
  are candidate multimodal, 82 are noisy/contradictory, and 405 have one
  unique record.
- The train-only artifact contains 1,746 consensus-relative labels across 565
  training genes.

These are relative pseudo-anchors. Same-gene agreement does not resolve
absolute strand symmetry, so E5 still requires structural or family-frame
anchors.

## Required Portable Data

The repository includes `data/` because the complete tree is approximately
117 MB and no individual file exceeds GitHub's 100 MB file limit.

Core files for the clean V19 benchmark:

```text
data/processed/tf_pwm_aug_dbd_canon_trim.parquet
data/processed/tf_dbd_embeddings_aug.npz
data/processed/splits/cluster40_clean/split.json
data/processed/splits/cluster40_clean/group_manifest.json
data/processed/splits/cluster40_clean/cdhit_clusters.clstr
data/processed/splits/cluster40_clean/full_hygiene_report.json
data/processed/tf_nn_index_cluster40_clean.json
data/processed/tf_nn_index_cluster40_clean.json.manifest.json
```

SHA-256 checksums:

```text
f2f9b5f8d7382c92e5020c736830376c252ebddd395bffd94913a1f73f1bb375  data/processed/tf_pwm_aug_dbd_canon_trim.parquet
40f1fc0584721936356e6b2dbf81c0ea27a962b67371e98a6d848bcbf9b9a940  data/processed/tf_dbd_embeddings_aug.npz
5d8e94bf7b548278b813bb48b2df954342ff08bd1377b93ab0d89cf17a827da2  data/processed/splits/cluster40_clean/split.json
2932e938e6344bacf45ac052f8b814855ba8931157b29405b5d56c1eca3bda1e  data/processed/splits/cluster40_clean/cdhit_clusters.clstr
30c171545579f6bda5b8dce7740a58a023538c6dda65a5d318803f6cec2dae9f  data/processed/tf_nn_index_cluster40_clean.json
```

Downloaded papers, licensed FoldX binaries, generated results, logs, and
checkpoints are intentionally excluded. Existing checkpoints live outside the
repository under machine-specific `/n/holylabs/...` paths and must be copied
separately if old trained weights are needed.

Redistribution of raw third-party datasets remains subject to their upstream
licenses. Use a private repository if those terms do not permit public
redistribution.

## Environment Setup

```bash
mamba env create -f environment.yml
mamba activate tfscope
```

The environment installs Python 3.10, CD-HIT, and the project dependencies.
PyTorch is intentionally excluded. Install the CUDA or CPU build appropriate
for the target machine using https://pytorch.org/get-started/locally/. Install
`entmax` afterward only when running sparse-attention experiments.

The original environment name was `tfscope`. Training uses ESM-2
`esm2_t33_650M_UR50D`; model weights are downloaded or loaded through
`TORCH_HOME` and are not committed.

## Verification

Run from the repository root:

```bash
PYTHONPATH=src:scripts python -m unittest discover -s tests -v

python scripts/audit_split_hygiene.py \
  --data data/processed/tf_pwm_aug_dbd_canon_trim.parquet \
  --split data/processed/splits/cluster40_clean/split.json \
  --index data/processed/tf_nn_index_cluster40_clean.json \
  --fail-on-leakage
```

Expected result:

- 28 tests pass;
- split audit is clean;
- retrieval audit has zero violations.

Rebuild artifacts with:

```bash
python scripts/build_cluster40_split.py
python scripts/build_nn_index.py
```

The split builder reuses
`data/processed/splits/cluster40/cdhit_clusters.clstr` by default because
CD-HIT was not installed in the original runtime.

## Current V19 Result And Next Step

The old E2/E4/E5 checkpoints were invalid as trained-model baselines because
the checkpoint filter omitted all LoRA tensors under `backbone._esm_model`.
Saving and evaluation now retain and require those tensors.

Corrected single-seed results on the clean split:

- E2 fixed-frame RAG: test gene-macro panel-r `0.4945`, canon-r `0.1590`;
- E5b family-register model: panel-r `0.6252`, canon-r `0.1023`;
- E6 aligned donor reranker improves donor AUC and top-1 transfer quality but
  does not improve the final fused PWM;
- E7 position-wise gating and E8 de-novo-frame donor alignment were rejected
  on validation.

The current publication candidate is a validation-locked E2-frame/E5b-content
family composition. The frozen policy is stored in
`results/v19_e9_model_composition/validation_composition_grid.json`. On test it
achieves gene-macro:

```text
panel-r    0.5454  (E2: 0.4945)
canon-r    0.1527  (E2: 0.1590)
MAE        0.8361  (E2: 0.8386)
fixed MAE  1.1168  (E2: 1.1427)
RMSE       0.3070  (E2: 0.3186)
CE         1.4309  (E2: 1.5592)
KL         0.9729  (E2: 1.1005)
```

The paired 10,000-replicate gene bootstrap gives panel-r delta `+0.0509`,
95% CI `[+0.0305, +0.0724]`. Fixed MAE, RMSE, CE, and KL also have
improvement intervals excluding zero. Canon-r and aligned MAE are statistically
unchanged; top1/F1/MCC have small non-significant declines.

Next:

1. compare against external published baselines under the same split where
   possible;
2. report the two-model inference cost and retain E2 as the single-model
   baseline;
3. use `results/v19_e9_publication/` for final family-stratified tables,
   confidence intervals, and motif-logo examples.

Do not interpret improvements against the historical cluster40 benchmark as
V19 gains. All V19 comparisons must use the clean split and train-only donor
policy.

## Important Design Decisions

- Keep the historical split and index untouched for result reproducibility.
- Use explicit clean artifact paths rather than silently changing global
  defaults in old experiment scripts.
- Treat source IDs carefully: broad assay/method labels are not identities.
- Use whole connected components for splitting; never split individual motif
  rows independently.
- Validation and test retrieval must remain train-only.
- Prefer gene-macro metrics because row-level averages overweight TFs with
  many motif records.
- Registration work should precede larger backbones or additional MoE experts.

## Known Limitations

- Eight rows have UniProt IDs absent from the reused CD-HIT cluster file. They
  remain grouped by gene, UniProt ID, specific source, and exact sequence.
- The clean split guarantees explicit identity and 40%-cluster separation but
  does not yet include PWM-similarity clustering.
- The user explicitly fixed the experiment scope to seed 42. Report this as a
  limitation; do not launch seeds 43/44 unless the user changes that decision.
- E3 labels are consensus-relative and cannot establish an absolute biological
  strand without structural or family-frame evidence.
- Calibration, conformal abstention, and external benchmark comparisons remain
  incomplete.
