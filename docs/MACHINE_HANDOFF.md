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

### E1 launcher

`scripts/train_v19_e1_baselines.sbatch` launches six matched baseline runs:

- no-RAG seeds 42, 43, and 44;
- train-only K=16 RAG seeds 42, 43, and 44.

These jobs were prepared but were not submitted as part of the implementation.

### E2 support

`scripts/train.py --gene-balanced-sampling` enables deterministic uniform gene
sampling while preserving the original number of examples per epoch. For each
sampled gene, one of its motif records is selected.

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

- six tests pass;
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

## Next Step

Run E1 before making architectural changes:

```bash
sbatch scripts/train_v19_e1_baselines.sbatch
```

Evaluate the six runs using gene-macro gate-r, gene-macro canon-r, MAE,
top-base accuracy, and per-family metrics. This establishes the genuine RAG
benefit after leakage removal.

Then:

1. repeat the baseline with `--gene-balanced-sampling`;
2. audit same-gene PWM registration consistency;
3. build anchored orientation/offset labels;
4. implement latent registration loss;
5. train a motif-transfer retrieval reranker;
6. add per-position donor trust;
7. pretrain recognition/contact supervision.

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
- The E1 jobs have not produced clean baseline metrics yet.
- Registration, transfer reranking, structure pretraining, and calibration are
  planned but not implemented.
