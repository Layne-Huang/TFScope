# v26 Implementation Plan (file-level)

Companion to `docs/v26_audit.md`. Every path below is **new**; nothing in the v24/v25 namespace is
touched. Phase gates are hard — a phase does not start until the previous phase's tests pass.

## Namespace rules

| kind | location |
|---|---|
| build scripts | `scripts/v26/` |
| model code | `src/tfscope/v26/` (new package; v24 modules imported read-only, never edited) |
| configs | `configs/v26/*.yaml` |
| datasets | `data/processed/v26/` |
| coordinates / contacts | `data/contacts_v26/` |
| annotation snapshot | `data/annotations_v26/` |
| splits | `data/processed/splits/v26/` |
| checkpoints | `/data1/leihuang/TFScope_store/checkpoints/v26/` (symlinked as `checkpoints/v26/`) |
| results | `results/v26/` |
| tests | `tests/v26/` |
| docs | `docs/v26_*.md` |

**Untouched, by assertion:** `tf_pwm_training_v2{3,5flank,5xtal}.parquet`,
`data/contact_maps/*_v2{3,5*}.json`, `checkpoints/v24_contact/**`, `checkpoints/iclr_phase1/**`,
`results/iclr_phase1_apples_to_apples/**`. `tests/v26/test_legacy_untouched.py` hashes these and
fails the suite if any changes.

---

## Execution convention — everything runs detached

**Standing rule for all v26 work: no job runs in the foreground of an interactive session.**
Every build, fetch, training run and evaluation goes through `scripts/v26/run_detached.sh`, which
wraps the payload in `setsid nohup … </dev/null &` + `disown`. Verified behaviour: launched jobs
reparent to `PPID=1` with their own session ID, so they survive terminal close, SSH drop and agent
session termination.

```bash
scripts/v26/run_detached.sh [--gpu N] <job_name> <command...>
scripts/v26/job_status.sh                 # table of all jobs
scripts/v26/job_status.sh <job_name> 60   # last 60 log lines
kill -- -$(cat /data1/leihuang/TFScope_store/v26_logs/<job>/pid)   # kill whole process group
```

State per job in `/data1/leihuang/TFScope_store/v26_logs/<job_name>/`:
`cmd.txt`, `pid`, `log.txt`, `started_at`, `finished_at`, `STATUS` ∈ `RUNNING | DONE | FAILED:<rc>`.
`run_detached.sh` refuses to start a job that is already `RUNNING`.

Two hard requirements, both learned from the v25 post-mortem (`docs/v26_audit.md` §2.3, where a
conda `PermissionError` still printed "done" and the result was lost):

1. **Every chain script uses `set -euo pipefail`, checks the upstream `STATUS` marker rather than
   file existence, and returns a distinct exit code per failure mode.** Never pipe a payload into
   `grep` — it masks the exit status.
2. **Every backgrounded loop over more than ~20 items prints a per-item progress counter**
   (index, ok/fail, rate, elapsed, ETA) so progress is legible without guessing from CPU time.

Long jobs are also resumable where the work is expensive: the annotation fetch skips keys already
present in its `.jsonl.gz` outputs, so a killed run loses at most one request.

---

## Phase 0 — audit ✅ COMPLETE

- `scripts/v26/audit_phase0.py` → `results/v26_audit/` (8 tables + `audit_phase0.json`)
- `docs/v26_audit.md`

---

## Phase 1 — annotation snapshot + canonical data model

**Gate:** every example resolves to a UniProt accession + a sequence-derived DBD, or is explicitly
reported as unresolvable. No live API call in any downstream script.

| file | purpose |
|---|---|
| `scripts/v26/fetch_annotation_snapshot.py` | one-time InterPro/Pfam + UniProt fetch for all 1,335 genes; stores **raw JSON responses**, release string, access date → `data/annotations_v26/interpro_snapshot_<release>.json.gz`, `uniprot_snapshot_<date>.json.gz`, `SNAPSHOT.md` |
| `scripts/v26/build_canonical_targets.py` | gene/PDB-chain → UniProt accession → canonical sequence; emits `data/processed/v26/targets.parquet` with the brief's schema (`example_id, target_unit_id, primary_accession, primary_full_sequence, primary_sequence_hash, primary_dbd_spans, …`); **gene-symbol fallback logged to `results/v26/accession_ambiguity.csv`** |
| `scripts/v26/build_dbd_spans.py` | DBD spans from the snapshot **only** — never from DNA contacts; multi-domain handling in two explicit modes (`--mode target_dbd` / `--mode enumerate`), recorded per row in `dbd_selection_mode` |
| `scripts/v26/build_v26_datasets.py` | emits `v26_core`, `v26_flank20`, `v26_flank32` from one span table, so the three differ **only** in flank width |
| `tests/v26/test_dbd_spans.py` | asserts: no span derived from contacts; spans are UniProt-coordinate; `core ⊂ flank20 ⊂ flank32`; every `primary_sequence_hash` matches its sequence |

`example_id` is a **content hash**, not a positional index — this retires Finding J.
A `legacy_filename` column preserves the `seq_<i>`/`str_<i>` mapping for backward joins.

**Deliverable:** `docs/v26_data_design.md`.

---

## Phase 2 — canonical contact coordinates

**Gate:** `clipped == 0` everywhere; every dropped contact appears in a report with a reason.

| file | purpose |
|---|---|
| `scripts/v26/build_contact_coordinates.py` | re-derives contacts from mmCIF into the brief's full record (`pdb_auth_resid → chain_local_idx → uniprot_idx → crop_idx → tensor_idx`, `chain_entity_id`, `chain_role`, `duplex_id`, `bp_index`, `pwm_column`, `min_distance`) → `data/contacts_v26/contacts_canonical.parquet` |
| `scripts/v26/project_contacts_to_crop.py` | projects canonical → any crop; **masks** out-of-crop contacts, never moves or drops them; emits `contact_targets_<dataset>.npz` + `mapping_report.csv` |
| `scripts/v26/build_recognition_prior_v26.py` | rule-based prior, kept **strictly separate** from empirical contacts, in UniProt coordinates |
| `scripts/v26/rebuild_v24_compatible_contacts.py` | regenerates v24-shaped targets from the same canonical source so the v24-vs-v26 ablation is fair (writes to `data/contacts_v26/`, does **not** overwrite `contact_maps/`) |
| `src/tfscope/v26/contacts.py` | loader that **raises** on an unmappable index instead of `if 0 <= i < L` |
| `tests/v26/test_contact_coordinates.py` | round-trip auth-resid → tensor idx; chain-identity assertion; zero silent drops; coverage report non-empty |

Test-structure contacts are tagged `eval_only=True` and physically excluded from any training
target file by `project_contacts_to_crop.py`.

**Deliverable:** `results/v26/contact_mapping_audit/` tables.

---

## Phase 3 — leakage-controlled split

**Gate:** all zero-overlap assertions pass, *including* Assembly-OOD (Finding E), with the
application sets locked out first.

| file | purpose |
|---|---|
| `scripts/v26/lock_application_sets.py` | removes whole components for Barrera (20 genes), MyoD1, DBP5/6/9/35, orphan case studies → `data/processed/splits/v26/application_holdout.json` **before** the split is built |
| `scripts/v26/build_v26_split.py` | hypergraph over accession / primary-DBD cluster / sequence hash / partner accession / partner cluster / WT-mutant group / design group / PDB assembly → components → stratified 70-75 / 10-15 / ~15 allocation |
| `scripts/v26/cluster_dbd_cores.py` | mmseqs on **DBD cores only**; records exact command + version (`17.b804f`) + cov-mode semantics |
| `scripts/v26/audit_c2h2_finger_units.py` | tandem-finger-unit audit so array-length differences don't create hidden leakage (targets the 1,189-row / 400-gene mega-component) |
| `scripts/v26/build_eval_subsets.py` | Test-Struct / Test-SeqOnly / Test-Monomer / Test-Multimer / Test-Contact / per-family |
| `scripts/v26/verify_split.py` | emits the zero-overlap assertion table; **non-zero → exit 1** |
| `tests/v26/test_split_manifest.py` | manifest schema + assertion table + application-holdout disjointness |

Outputs `data/processed/splits/v26/manifest.parquet` (frozen, hashed) and both Primary-OOD and
Assembly-OOD variants. Legacy 291-row manifest copied to
`data/processed/splits/v26/legacy_291_backcompat.json`, used for backward comparison only.

**Deliverable:** `docs/v26_split_report.md`.

---

## Phase 4 — model

**Gate:** smoke test trains 20 steps on 200 rows on one GPU; parameter counts and a
`assert_no_metadata_inputs()` guard pass.

| file | purpose |
|---|---|
| `src/tfscope/v26/encoder.py` | **per-chain** ESM-2 forward (retires Finding H); AA-identity skip + relative-DBD-position embedding: `h_i = LN(h_ESM + W_aa·AA(a_i) + W_pos·relpos(i))`; lightweight residue refiner |
| `src/tfscope/v26/moe.py` | sequence-conditioned router `softmax(W[h_i ; z_core])` — **no `family_id`**; 1 shared + 4 routed, top-2, load balance; param-matched dense FFN control |
| `src/tfscope/v26/context.py` | gated flank residual, `alpha = sigmoid(W[z_core;z_flank]+b)`, `b init = −3`; flank dropout 0.3–0.5; shuffled-flank control hook |
| `src/tfscope/v26/complex.py` | permutation-invariant partner-set aggregation (adapts `chain_set_encoder.py`), primary-as-query; partner dropout; stoichiometry |
| `src/tfscope/v26/length_head.py` | `P(L\|z)` over 4–42 → contiguous prefix gate `g_j = P(L>j)` |
| `src/tfscope/v26/pwm_head.py` | `Z_final = Z_prior + λ_contact·Z_contact`, λ learned + small init; flanks excluded from contact keys/values |
| `src/tfscope/v26/model.py` | assembly; `assert_no_metadata_inputs()` raises if any tensor carries family/source/gene/PDB/provenance |
| `configs/v26/{core,core_nomoe,core_moe,context,context_shuffled,complex_primary,complex_partners,prior_only,full}.yaml` | one config per required ablation |
| `tests/v26/test_no_metadata_inputs.py` | forward-hook scan of every input tensor |
| `tests/v26/test_permutation_invariance.py` | partner-order swap → identical output |
| `tests/v26/test_flank_gate_init.py` | at init, flanked forward ≡ DBD-only forward within tolerance |

**Deliverable:** `docs/v26_model_design.md`.

---

## Phase 5 — training

| file | purpose |
|---|---|
| `src/tfscope/v26/sampler.py` | target-unit → construct → motif-source → record hierarchical sampling |
| `scripts/v26/train_v26.py` | Stage A (core) → B (context+partners, lower LR on core path) → C (contact distillation, valid rows only, never zero-filled) |
| `scripts/v26/run_stage_{a,b,c}.sh` | launchers, GPU pinning below |
| `tests/v26/test_sampler.py` | target-unit uniformity; no application-holdout row ever emitted |

**GPU allocation** (verified free at audit time; GPU 0 excluded — known-bad on this node; 3/6/7/8 in use):

```
1 GPU-808f6dc0-74c2-c81b-b926-f15a367f6a4d
2 GPU-a0253811-eb51-fb1d-9951-80e883714728
4 GPU-26df3b25-f077-10ed-57eb-47e5a71c0cef
5 GPU-348cc075-8a3d-0689-3039-395677fd3431
9 GPU-3e792bc3-d868-a8f5-ea8c-806dde5cdd80
```

Two modes, both required by the node's known DDP quirks (UUID pinning + `NCCL_P2P_DISABLE=1`,
A6000 without NVLink):
- **Ablation sweep** — 1 config per GPU, 5 concurrent single-GPU runs (best throughput for the
  10 model × 5 data comparisons × 3 seeds).
- **Final ensemble** — 5-way DDP over the same UUIDs, `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.

Every long-running loop gets a per-item progress counter before launch.

---

## Phase 6 — evaluation

| file | purpose |
|---|---|
| `scripts/v26/eval_v26.py` | unified evaluator; statistical unit = **target unit**, not row |
| `scripts/v26/bootstrap_ci.py` | paired bootstrap over target units; non-inferiority test for the DeepPBS parity claim |
| `scripts/v26/eval_mutation_representation.py` | the 8-way Barrera/MyoD1 diagnostic (predicted vs observed ΔPWM, directional switch accuracy, and the four embedding-distance probes) — separates PLM invariance / pooling dilution / head insensitivity / crop effects |
| `scripts/v26/eval_contacts.py` | AUROC, AUPRC, top-k recovery, 2-D alignment, mapping coverage — aggregate before any named example |
| `scripts/v26/eval_designed_proteins.py` | DBP5/6/9/35, run **only after model freeze** |
| `scripts/v26/make_results_tables.py` | final summary tables + CIs |

**Deliverables:** `docs/v26_experiment_plan.md`, `results/v26/RESULTS.md`.

---

## Sequencing and cost

Phases 1–3 are CPU-only (plus one network fetch) and cheap — a day of wall-clock, no GPU. Phase 4
smoke tests are minutes on one GPU. Only Phase 5 is expensive, and it does not start until
`verify_split.py` exits 0 and the Phase-2 mapping report shows zero silent drops.

Recommended first increment: **Phase 1 + Phase 2 together**, because the contact-coordinate repair
(Finding C/D) is what makes the v24-vs-flank comparison interpretable, and both feed the split.
