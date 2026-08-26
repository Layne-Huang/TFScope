# v26 Data Design (Phase 1) — COMPLETE

Companion to `docs/v26_audit.md` (findings) and `docs/v26_implementation_plan.md` (file plan).
Everything here is reproducible from the frozen snapshot with no network access.

## 1. What Phase 1 delivers

A canonical, UniProt-coordinate data model in which **`seq_` and `str_` rows share one
sequence-derived DBD definition**. No DBD boundary anywhere is derived from DNA contacts — that
closes audit Finding A, where 291/291 test rows had structure-defined input boundaries.

| artifact | contents |
|---|---|
| `data/annotations_v26/{uniprot,interpro,sifts,gene_resolution}.jsonl.gz` | raw API responses, verbatim, `{key,status,payload}` per line |
| `data/annotations_v26/release.json`, `SNAPSHOT.md` | UniProt release **2026_02 (10-June-2026)**, access date, per-source ok/fail counts |
| `data/annotations_v26/dbd_pfam_whitelist.json` | 2-tier curated DBD family whitelist + provenance |
| `data/processed/v26/accessions.parquet` | 1,400 accessions: canonical sequence, `sequence_hash`, gene, organism, reviewed flag |
| `data/processed/v26/domains.parquet` | 61,255 InterPro fragments over 13 member databases, 1-based UniProt coords |
| `data/processed/v26/sifts_mappings.parquet` | 2,987 PDB-chain → UniProt mappings over 819 PDB ids |
| `data/processed/v26/row_resolution.parquet` | every one of the 6,012 v23 rows → accession + resolution method |
| `data/processed/v26/dbd_candidates.parquet` | **1,598 sequence-derived DBD spans over 1,385 accessions** |
| `results/v26/missing_dbd_triage.csv` | every accession without a DBD, with an explicit decision |
| `results/v26/accession_ambiguity.csv` | every fallback / multi-candidate resolution |

## 2. Reproducible commands

All jobs run detached from the `/data1` mirror (see `docs/v26_implementation_plan.md`):

```bash
scripts/v26/sync_to_data1.sh                          # mirror code + read-only inputs to /data1
scripts/v26/run_detached.sh --mirror snap  scripts/v26/fetch_annotation_snapshot.py
scripts/v26/run_detached.sh --mirror chain scripts/v26/run_phase1_chain.sh      # steps 2-3
scripts/v26/run_detached.sh --mirror r2    scripts/v26/run_phase1_round2.sh     # round-2 accessions
scripts/v26/run_detached.sh --mirror f     scripts/v26/run_phase1_step3b.sh     # apply triage
python tests/v26/test_dbd_spans.py && python tests/v26/test_legacy_untouched.py
```

## 3. Accession resolution

`str_` rows resolve through **SIFTS `(pdb_id, chain_id)` → UniProt**, which is independent of DNA
contacts. `seq_` rows resolve through the `tf_pwm.parquet` gene→UniProt seed, verified against the
snapshot. Gene-symbol search is a logged fallback.

| method | rows |
|---|---|
| `gene_seed_uniprot_id` | 4,578 |
| `sifts_pdb_chain_gene_matched` | 1,265 |
| `sifts_pdb_chain` (no gene match available) | 97 |
| `gene_symbol_search_FALLBACK` (logged ambiguous) | 62 |
| `UNRESOLVED` | 10 |

Two problems found and fixed here:

**Round-1 seed was incomplete.** The initial fetch seeded accessions only from `tf_pwm.parquet`'s
gene map, but SIFTS resolves structure chains to orthologs and isoforms outside that map — 163
accessions / 496 rows had no sequence or annotation. A resumable round-2 fetch closed it
(`--round2`).

**SIFTS returns crystallisation fusion partners.** For some chains the highest-coverage accession
was maltose-binding protein (`MALE`, 14 rows) or GFP, not the TF — so the row had no DBD. Resolution
now prefers the accession whose UniProt gene matches the row's gene symbol, falling back to
coverage. Gene symbol is preprocessing metadata only, never a model input.

The 10 `UNRESOLVED` rows are 6 **composite dimer names** — `FOS::JUN`, `MAX::MYC`, `NR1H2::RXRA`,
`POU2F1::SOX2`, `PPARG::RXRA`, `RXRA::VDR`. A single primary accession is the wrong model for these;
they must be built as two-entity assemblies in `partner_entities`. **Open item for Phase 2.**

## 4. Unified DBD definition

Boundaries come only from the frozen InterPro snapshot:

1. Keep fragments whose Pfam or InterPro accession is on the curated whitelist.
2. Merge overlapping or abutting fragments.
3. Cluster merged fragments separated by ≤ **40 residues** into one candidate DBD — the
   `cluster_crop_v2.py` biological argument (C2H2 TGEKP linkers are ~5–7 aa, loose arrays rarely
   exceed 20–30), so tandem arrays stay intact while separate domains split.
4. Emit **every** candidate. No selection happens here.

Result: 1,598 spans over 1,385 accessions; span length median 93, p10 57, p90 391;
253 tandem-array spans; **147 accessions have >1 candidate** and therefore require an explicit
selection mode downstream. Coverage: **5,966 / 6,012 v23 rows (99.2%)** have a sequence-derived DBD.

### Two-tier whitelist

**Tier 1** (57 families) defines spans. **Tier 2** (59 entries) is *rescue-only*: applied solely to
accessions with zero tier-1 fragments.

This split exists because of a regression caught during triage. Adding `IPR013087`
("Zinc finger C2H2-type") globally lengthened **534 / 1,357 existing spans, up to +429 aa on
ZNF142**, because it annotates more fingers than `PF00096` — i.e. rescuing 40 unannotated
accessions would have silently redefined the DBD for the largest family in the dataset. Tier 2 makes
that impossible, and `tests/v26/test_dbd_spans.py` asserts it.

Round-3 additions were promoted to tier 1 because they are domain-level (not superfamily-inflating)
and cover organisms that previously had no DBD at all: Zn(2)Cys(6) fungal, WRKY, B3 (plant),
Brinker, BES1/BZR1, AFT, Brf1-TBP. These produced exactly **one** boundary change on a pre-existing
accession — POGK (Q9P215), 250→195 — which is correct (POGK carries both a CENP-B HTH and a Brinker
DBD) and is recorded with justification in `tests/v26/accepted_span_changes.json`. Any unreviewed
change now fails the test suite.

### Residual: 15 accessions / 36 rows without a DBD, all explicit

| decision | accessions | rows |
|---|---|---|
| `not_a_TF` (chemokine receptor, protein kinase, MBP/GFP fusion tags, cGAS, rRNA processing) | 10 | 21 |
| `excluded_by_policy` (PHD/bromodomain/MBD chromatin readers — same policy as `reextract_dbd_round3.py`) | 2 | 10 |
| `unclassified` (needs review) | 3 | 5 |

The 3 unclassified: `C0HLU2` ZNF689 (46-aa entry, no domains — likely a wrong accession),
`Q13952` NFYC (histone-fold NF-Y subunit; genuine TF but "histone-fold" is too generic to whitelist
globally — needs an accession-level exception), `Q7FAD5` ZEP1 (no InterPro annotation at all).

## 5. Multi-domain proteins — 147 accessions

No selection is made in Phase 1, and the motif database's family label is **never** used to choose a
domain (audit Finding I: 103/208 genes were previously cropped that way). Phase 2+ must declare one
mode per experiment:

- **Target-DBD mode** — the task supplies the annotated DBD; `dbd_selection_mode` records the
  provenance of that choice.
- **Candidate-enumeration mode** — predict a PWM for every candidate, select nothing using target
  PWM metadata.

## 6. Invariants under test

`tests/v26/test_dbd_spans.py` (6/6 passing):
spans within protein bounds and 1-based · no contact-derived column · no oracle column
(`motif_source`, `family_id`, `pwm`, `gene_symbol`) · tier-2 never co-occurs with tier-1 ·
zero unreviewed span drift · residual triage fully explicit.

`tests/v26/test_legacy_untouched.py`: 26 v24/v25 artifacts fingerprinted and unchanged. Paths are
absolute AFS on purpose — a relative version produced a false "DELETED" alarm when the suite ran
from the `/data1` mirror.

## 7. Not yet built

`v26_core` / `v26_flank20` / `v26_flank32` datasets are **not** emitted yet. `build_v26_datasets.py`
needs the candidate-selection mode decided, and the 6 composite-dimer targets modelled as
assemblies. Both are Phase 2 entry conditions.
