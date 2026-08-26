# v26 Phase-0 Audit

**Date:** 2026-08-14 · **Branch:** `iclr` · **Scope:** read-only audit of the v23/v24/v25 pipeline.
**Reproduce:** `python scripts/v26/audit_phase0.py` (tfscope env, repo root) → `results/v26_audit/`

Nothing under `data/`, `checkpoints/` or `results/iclr_phase1_apples_to_apples/` was modified.
Every claim below is either a source-code reference (`path:line`) or a number emitted by
`scripts/v26/audit_phase0.py` into `results/v26_audit/`.

---

## 0. Executive summary — what actually blocks a Nature Methods submission

Ranked by severity. Items marked **CONFIRMED** were measured; **VERIFIED-AS-DESCRIBED** means the
task brief's description matched the code exactly.

| # | Finding | Severity | Evidence |
|---|---|---|---|
| A | **100% of the 291-row primary test set (291/291) has its model-input boundary defined by the held-out co-crystal.** Every test row is a `str_` row, and `str_` DBD spans are the contiguous 4.5 Å DNA-contact span. | **Blocking** | §4, `04_test_rows_with_contact_defined_dbd.csv` |
| B | **All 20 Barrera genes are in the training split.** The mutation-blindness result — the basis for the "frozen ESM-2 is the wall" claim — was measured on TFs the model trained on. | **Blocking** | §7.5 |
| C | **v23/v24 silently clip 423/6,623 (6.4%) recognition indices and 1,034/10,232 (10.1%) contact links; 122 contact columns are emptied entirely** and then masked out, so the loss silently never sees them. | **Blocking** | §9 |
| D | **The v24↔v25flank comparison is confounded.** v23 clips 1,034 contact links to zero; v25flank clips 0 but relocates 680 links into flank residues. The two models were trained on materially different supervision, so "flanks hurt" is not a clean result. | High | §9 |
| E | **Assembly-OOD leakage exists**: 13 partner/primary DBD clusters shared train↔test, 16 train↔val, 5 val↔test. Primary-only clusters are clean (0 shared). | High | §7.4 |
| F | **MyoD1 +9.17 / +0.03 / −1.40 differ in model AND sequence AND crop.** The 52-aa case-study construct is not a substring of either training row — it differs by a leading residue, 15 extra C-term residues, *and* a C24S substitution. | High | §8 |
| G | `family_id` is a live model input threaded into the residue-MoE, the MoE and the PWM head. | High | §2.2 |
| H | All chains are **concatenated into one ESM-2 forward** with `<eos>` separators, not encoded separately. | High | §4 |
| I | 103/208 multi-domain genes had their DBD chosen using the **motif database's family label** — oracle preprocessing. | Medium | §11 |
| J | `filename` keys (`seq_<i>` / `str_<i>`) are **positional row indices**; all contact/recognition supervision is keyed on them. A row-order change silently misaligns supervision (this class of bug already occurred once — see memory `v24-contact-grounding`). | Medium | §1.3 |
| K | `cluster_crop_v2.py` writes its only audit table to `/tmp` — not durable, not reproducible. | Medium | §11 |
| L | InterPro/Pfam annotations are fetched **live from the API** with no release pin or snapshot. | Medium | §1.4 |

**Not a problem (verified, contrary to the brief's concern #8):** the ESM-C embedding cache is keyed
by `hashlib.md5(sequence)`, not by filename — `src/tfscope/data/dataset.py:282-285`. See §5.

---

## 1. Exact current dataset paths and schemas

### 1.1 Lineage (verified by reading each builder's declared inputs/outputs)

```
AllPWMs_JASPARV1_H13Core_CISBPv194w2.tar.gz
  └─ scripts/legacy/parse_pwms.py                    → data/processed/pwm.parquet
                                                       data/processed/tf_pwm.parquet
     ├─ SEQUENCE TRACK
     │    scripts/map_tf_annotations.py (live InterPro API)
     │    scripts/reextract_dbd_round3.py            (Pfam→family table, 3 rounds)
     │    scripts/cluster_crop_v2.py                 (multi-domain crop choice)
     │    scripts/legacy/build_augmented_dbd_dataset.py
     │                                               → tf_pwm_aug_dbd.parquet
     │    scripts/canonicalize_pwms.py               → tf_pwm_aug_dbd_canon_trim_v2.parquet
     └─ STRUCTURE TRACK
          scripts/build_deeppbs_structural_v2.py     → tf_pwm_deeppbs_v2_deduped.parquet
                                                       tf_pwm_deeppbs_v2_partner.parquet
  └─ scripts/legacy/build_training_table.py          → tf_pwm_training_v22.parquet
                                                       splits/train_v22/{split.json,assignments.parquet}
  └─ scripts/legacy/build_nchain_v23.py              → tf_pwm_training_v23.parquet   ← v24 trains on this
  └─ scripts/legacy/build_flank_dataset.py           → tf_pwm_training_v25flank.parquet
  └─ scripts/legacy/build_flank_dataset_xtal.py      → tf_pwm_training_v25xtal.parquet
  └─ scripts/legacy/build_flank_contact_targets.py   → contact_targets_v25flank.json
                                                       recognition_residues_v25flank.json
```

**Discrepancy vs the brief:** the brief lists `scripts/contact_teacher_v2.py`. No such file exists.
The actual 2-D contact builder is **`scripts/contact_teacher/build_contact_targets.py`**; the 1-D
rule-based prior builder is **`scripts/legacy/build_recognition_prior.py`**.

### 1.2 Schema (21 columns, identical across v23/v25flank/v25xtal)

`filename, gene_symbol, sequence, pwm, motif_length, seq_length, dbd_start, dbd_end, family_id,
family_name, family_source, motif_source, partner_sequence, partner_gene, is_dimer, _set, gene_key,
group_id, multichain_eligible, partner_seqs, n_chains`

v25flank/v25xtal add `flank_source`.

### 1.3 Inventory (`results/v26_audit/01_inventory.csv`)

| version | rows | genes | uniq seq | seq\_ | str\_ | med len | DBD frac of input | dbd_start>0 |
|---|---|---|---|---|---|---|---|---|
| v23 (v24 trains here) | 6,012 | 1,335 | 2,262 | 4,641 | 1,371 | 83 | **1.00** | 0% |
| v25flank | 6,012 | 1,335 | — | 4,641 | 1,371 | 117 | 0.72 | 95% |
| v25xtal | 6,012 | 1,335 | — | 4,641 | 1,371 | 111 | 0.73 | 92% |

`flank_source`: v25flank `{flanked: 5752, dbd_only: 260}`; v25xtal `{flanked: 4554, xtal_full: 1371, dbd_only: 87}`.

**Finding J.** `filename` is assigned positionally — `build_training_table.py:68` (`"seq_%d" % i`) and
`:96` (`"str_%d" % i`). Contact and recognition JSONs are keyed on these strings. Any reordering or
row insertion upstream silently remaps supervision onto the wrong protein. This exact failure mode
already occurred once (silent contact-supervision no-op, fixed in v24).

### 1.4 Annotation reproducibility (Finding L)

`scripts/map_tf_annotations.py` / `scripts/reextract_dbd_round3.py` call the live InterPro REST API.
No release version, no access date, no cached response is stored. Re-running today would silently
produce different DBD boundaries. There is **no snapshot to pin**, so v26 must create one.

---

## 2. Exact configs

### 2.1 v24 — `/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/config.json`

Launcher `scripts/run_v24_contact_ddp.sh`. Selected values:

```
esm_model esm2_t33_650M_UR50D · esm_embed_dim 1280 · esm_layers_to_average 4 · freeze_encoder true
lora_rank 16 · lora_alpha 32 · lora_n_layers 6 · learning_rate 4.5e-4 · lora_learning_rate 7.5e-6
moe_granularity residue · num_experts 8 · n_shared_experts 2 · top_k 2 · expert_hidden_dim 512
num_families 10 · family_embed_dim 64 · family_embedding_path ""          <-- FINDING G
gate_mode span · max_motif_length 42 · min_motif_length 4 · latent_registration true
pwm_head_v18 true · v18_contact_supervision true · v18_contact_weight 0.3
v18_contact_bias_scale 1.0 · v18_contact_bias_learnable true · contact_distill_weight 0.2
contact_targets_path data/contact_maps/contact_targets_v23.json
recognition_prior_path data/contact_maps/recognition_residues_v23.json
two_chain_input true · chain_id_embedding true · max_chains 4              <-- FINDING H
batch_size 6 · grad_accum 1 · world_size 6 · effective_batch 36 · epochs 225 · seed 42 · bf16
```

Data `tf_pwm_training_v23.parquet`, split `splits/train_v22/split.json`.

### 2.2 `family_id` is a live model input (Finding G)

`src/tfscope/models/tfscope.py:97` takes `family_id` in `forward`, then passes it to:
- `:150` `self.residue_moe(dbd_emb, family_id, dbd_mask)`
- `:160` `self.moe(combined, family_id)`
- `:198,202` into the PWM head kwargs

Case-study scripts hardcode it, e.g. `scripts/build_fig4a_switch_ens.py:29` `FID = 3  # bHLH`.
This violates the v26 acceptance criterion directly.

### 2.3 v25flank / v25xtal

`scripts/legacy/run_v25flank_single.sh`, `run_v25xtal_single.sh`. Byte-identical to v24 except
`--data`, `--contact-targets-path`, `--recognition-prior-path`, and batch geometry
(single-GPU `--batch-size 12 --grad-accum-steps 3` = the same global batch of 36).

Checkpoints: `checkpoints/iclr_phase1/v25{flank,xtal}/seed42/ckpt_best.pt` — **both deleted**.
`checkpoints/iclr_phase1/` now holds only `B0–B7` and `v24_ens`. Neither v25 result reached
`results/iclr_phase1_apples_to_apples/unified_models.json` (25 keys, no `v25*`); the surviving
numbers live in `/data1/leihuang/TFScope_store/v25flank_prelim.json` (0.602, tagged "prelim" but in
fact scored on the final `ckpt_best.pt` 42 min after training ended) and `v25xtal_bench.json` (0.540).
The scheduled final evals both failed silently — `eval_v25flank_when_done.sh` died on a conda
`PermissionError` and still printed "done" because it has no `set -e` and pipes into `grep`.

---

## 3. How each version constructs its inputs and supervision

| | v23 / v24 | v25flank | v25xtal |
|---|---|---|---|
| **sequence** | DBD crop only | `full[s−20 : e+20]` from UniProt | `str_` rows: whole resolved crystal chain; `seq_` rows: as v25flank |
| **dbd_start/end** | `0 / len(seq)` for **every** row | interior, 95% have `start>0` | interior, 92% |
| **dbd_mask** | `[dbd_start:dbd_end]` → **all-True, degenerate** | informative | informative |
| **partner_seqs** | ordered list, ≤3 partners, ≤150 aa each (`build_nchain_v23.py`) | carried through unchanged | carried through unchanged |
| **recognition prior** | rule-based, DBD-relative == seq-relative | `+dbd_start` shift, filtered to `[0, seq_len)` | same shift, xtal offsets |
| **1-D contacts** | *(no separate 1-D channel exists — see below)* | — | — |
| **2-D col×residue** | `contact_targets_v23.json` | `+dbd_start` shift | `+dbd_start` shift |

**Discrepancy vs the brief:** there is **no separate empirical 1-D DNA-contact label channel** in
the current code. There are only two channels — the *rule-based* recognition prior
(`recog_prior`, 1-D) and the *empirical* 2-D map (`contact_target`). The brief's requirement to
"not merge rule-based priors with empirical contact labels" is currently satisfied by accident
(they are separate tensors) but the rule-based prior is used at weight 0.3 (`v18_contact_weight`),
*higher* than the empirical 2-D distillation at 0.2 (`contact_distill_weight`). v26 must invert this.

Construction sites: `src/tfscope/data/dataset.py:545-577` (sequence/mask/partners),
`:615-635` (recog prior), `:637-667` (2-D contacts).

---

## 4. Does ESM receive concatenated or separate chains? — **CONCATENATED** (Finding H)

`src/tfscope/data/dataset.py:547-566` builds a single token stream:

```
chain1  <eos>  protomer1  <eos>  protomer2  ...
```

using ESM-2's `<eos>` (token 2) as a chain break, capped at `max_seq_len`. `dbd_mask` is True on all
protomer residues and False on separators. `src/tfscope/models/backbone.py:147-151` then prepends a
single `<cls>` and runs **one** ESM-2 forward over the concatenation. A per-token chain-ID embedding
(`chain_id_embedding: true`) makes the representation **order-aware**, so swapping two partners of a
homodimer changes the prediction.

A permutation-equivariant alternative already exists and is unused in v24:
`src/tfscope/models/chain_set_encoder.py` (ICLR Candidate A). It is a good starting point for
v26-complex, but note it still operates *downstream* of a shared residue projection — it does not by
itself make ESM encode chains separately. v26 must move the split up into the ESM call.

**Contact-defined boundaries (Finding A).** `scripts/build_deeppbs_structural_v2.py` (docstring
item 2) defines the `str_` DBD as the contiguous span `[min_contact_resnum, max_contact_resnum]`
over residues within 4.5 Å of DNA. Structure rows by split: **train 936, val 144, test 291**.
Because every test row is a `str_` row, **291/291 = 100%** of the primary benchmark has its input
boundary chosen using the held-out co-crystal. Full list:
`results/v26_audit/04_test_rows_with_contact_defined_dbd.csv`.

---

## 5. Cached embeddings and key generation — **PASSES**

`src/tfscope/data/dataset.py:272-308`. The ESM-C cache resolves each row to
`md5(sequence).hexdigest() + ".pt"`, with a secondary `md5(sequence[:1022])` key for truncated
sequences, and **raises** if a file is missing rather than silently falling back (`:296-306`).
Content-addressed, so a v25/v26 sequence change automatically misses the cache. Acceptance criterion
"all sequence caches are keyed by sequence content hash" is **already satisfied** for this cache.

v24 itself does not use it (`esm_model: esm2_t33_650M_UR50D`, live encode). `esm_dbd_embedding` in
`scripts/case_study/cs_utils.py:73` memoises only the *model object*, not embeddings — no staleness
risk. `tf_nn_index.json` retrieval is off in v24 (`use_retrieval: false`).

The real key-hygiene problem is not the embedding cache — it is the **filename-keyed supervision
JSONs** (Finding J, §1.3).

---

## 6. Split construction and MMseqs2 parameters

`scripts/legacy/build_training_table.py:196-278`.

```
mmseqs easy-cluster <fasta> <prefix> <tmp> --min-seq-id 0.4 -c 0.8 --cov-mode 1 -v 1
```

MMseqs2 version present on this machine: **17.b804f** (`/data1/leihuang/miniconda3/bin/mmseqs`).
`--cov-mode 1` = coverage of the **target** sequence. The clustered sequences are the *stored*
sequences (v22 DBD crops).

Then: bipartite gene↔cluster graph → connected components (`:212-226`); test components chosen
family-diverse, highest `n_str/n` first, target 200 (`:229-241`); structure rows in test components →
`test`, **sequence rows in the same components → `excluded`** (`:245-247`); 12% of remaining
components → `val`, seed 42.

Result `{train: 4794, val: 658, test: 291, excluded: 269}` — 274 components.

**Component-size pathology.** The largest component holds **1,189 rows across 400 genes** (the C2H2
mega-cluster: `ZNF273`, `ZNF274`, `ZNF519`… all collapse to cluster rep `seq_3799`). Top-10 sizes:
`{0:1189, 13:752, 11:251, 77:199, 19:124, 36:114, 96:109, 255:101, 57:99, 30:91}`. Two components
hold 32% of all rows. This is safe for leakage but leaves the split coarse and unbalanceable, and it
is exactly the "shared homologous finger unit" problem the brief flags — array-length differences
create clusters that are biologically near-identical.

---

## 7. Overlap audits (`results/v26_audit/03_split_overlaps.csv`)

Recomputed independently of the build script, on v23 rows under `splits/train_v22`.

| unit | train/test | train/val | val/test |
|---|---|---|---|
| gene symbol | 0 | 0 | 0 |
| exact sequence | 0 | 0 | 0 |
| `group_id` | 0 | 0 | 0 |
| legacy mmseqs cluster `_c` | 0 | 0 | 0 |
| primary DBD cluster (recomputed, mmseqs 17.b804f, 40%/0.8/cov-mode 1) | 0 | 0 | 0 |
| component `_comp` | 0 | 0 | 0 |
| **assembly clusters (primary + partner)** | **13** | **16** | **5** |

Components crossing train/val/test: **none** (the 7 "multi-split" components seen naively are
train+`excluded` pairs, which is the intended design).

**7.4 Assembly-OOD leakage (Finding E).** Clustering primary DBDs *and* all `partner_seqs` into one
space, 13 clusters appear in both train and test. Primary-OOD holds; Assembly-OOD does not. Any
multimer claim from v24 is Primary-OOD at best.

**7.5 Application-set leakage (Finding B).** All 20 Barrera homeodomain genes are in **train**:
`ARX, CRX, ESX1, HESX1, HOXB7, HOXC4, HOXD13, ISX, MSX2, NKX2-5, NKX2-8, PBX4, PHOX2B, PITX2, PROP1,
SIX6, VAX2, VENTX, VSX1, VSX2`. MYOD1 has 2 rows, both **train**.

Consequence: the mutation-sensitivity number that motivated the "frozen ESM-2 is the architectural
wall" conclusion was measured on memorised WT motifs. The conclusion may still be right, but it is
**not currently supported by a held-out measurement**. This must be re-measured on a locked,
never-trained component before it appears in a manuscript.

---

## 8. Origin of MyoD1 +9.17 / +0.03 / −1.40 (Finding F)

The three numbers differ in **model**, **sequence** and **crop** simultaneously.

| value | model / checkpoint | input sequence | length | script |
|---|---|---|---|---|
| **+9.17** | v19 "combined" — `checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt` | 52-aa case-study construct | 52 | `scripts/legacy/build_fig4a_switch.py:24` → `results/myod1_mut/switch_score_tfscope.json` |
| **+1.70** | v24 seed42 | same 52-aa construct | 52 | AUDIT_FINDINGS §14 |
| **+0.03** | v24 seed42 | parquet `str_700` (training-consistent) | 38 | AUDIT_FINDINGS §15 |
| **−1.40** | v25flank seed42 | flanked construct, `dbd=[20:58]` | 78 | `/data1/leihuang/TFScope_store/v25flank_myod1.log` |

The Δ_switch **formula is identical** across all four
(`Δ = [S_mut(CACGTG) − S_mut(CACCTG)] − [S_WT(CACGTG) − S_WT(CACCTG)]`, best log-odds over offsets
and both strands, background 0.25) and target-site orientation is handled the same way
(`rc()` + max over strands). So the formula is **not** the source of the discrepancy.

**New finding — the case-study construct is not the training sequence.** The 52-aa `WT_DBD` is *not*
a substring of either MYOD1 row:

```
52-aa case study : RKAATMRERRRLSKVNEAFETLKR C TSSNPNQRLPKV EILRNAIRYIEGLQA
str_700 (38 aa)  :RRKAATMRERRRLSKVNEAFETLKR S TSSNPNQRLPKV
                  ^ extra R              ^ C24S            ^ 15 extra residues
```

Three simultaneous differences: a leading `R`, a **C24S substitution** (crystal-construct
Cys→Ser), and 15 extra C-terminal helix-2 residues. The mutation under test is at index 11 (`L→R`),
common to both. So "+1.70 → +0.03 is a crop-length artifact" is itself under-determined — it could
be crop length, the C24S difference, or both. v26 must re-run this on a single canonical UniProt
coordinate frame with the substitution applied explicitly.

`FID = 3` (bHLH) is hardcoded in every MyoD1 script — another reason v26 must drop `family_id`.

---

## 9. Invalid / clipped recognition and contact indices (Finding C, D)

`results/v26_audit/02_recognition_index_validity.csv`, `02_contact_index_validity.csv`,
per-index detail in `02_invalid_index_detail.csv`.

Loader semantics being audited — **all three are silent drops**:
- recog: `if 0 <= p < len(recog)` — `dataset.py:630-633`
- contact column: `if not (0 <= c < max_motif_length): continue` — `dataset.py:645-647`
- contact residue: `if 0 <= ridx < ct.shape[1]` — `dataset.py:648-650`
- a column whose links were all dropped gets `contact_base_mask = 0` (`:651-652`) — masked, not
  zero-filled, which is correct behaviour but happens with no record.

### Recognition prior

| version | indices | inside DBD | in seq, outside DBD | **silently clipped** |
|---|---|---|---|---|
| v23 (v24) | 6,623 | 6,200 | 0 | **423 (6.4%)** |
| v25flank | 6,515 | 6,211 | **304** | 0 |
| v25xtal | 6,460 | 6,200 | **260** | 0 |

### 2-D contact targets

| version | residue links | **silently clipped** | outside DBD | cols dropped | **cols emptied** |
|---|---|---|---|---|---|
| v23 (v24) | 10,232 | **1,034 (10.1%)** | 0 | 0 | **122** |
| v25flank | 9,931 | 0 | **680** | 0 | 0 |
| v25xtal | 9,749 | 0 | **551** | 0 | 0 |

**Reading.** In v23 the DBD *is* the whole sequence, so nothing can be "outside the DBD" — indices
either fit or are destroyed. 10.1% of the empirical contact supervision v24 was trained on never
reached the loss, and 122 PWM columns lost their supervision entirely. After the `+dbd_start` shift,
v25flank keeps those same links but places 680 of them **on flank residues** — which the brief
correctly identifies as unacceptable, and which also means **v24 and v25flank optimise different
objectives**. The "flanks hurt" conclusion (Δ 0.629→0.602, n=1 seed, no CI) is confounded by this.

Root cause: the source indices in `recognition_residues_v23.json` are already out of bounds for the
crop — 77/228 entries have `max(primary) >= len(sequence)` in v23 itself, i.e. the defect predates
the flank work and lives in the prior builder, not the shift.

---

## 10. Structure rows whose input DBD came from co-crystal contacts

**All 1,371 `str_` rows**, of which **291 are the entire test set** and 936 are in train.
`results/v26_audit/04_test_rows_with_contact_defined_dbd.csv` lists filename, gene, family, length
for the 291 test rows.

`seq_` rows use InterPro/Pfam boundaries; `str_` rows use 4.5 Å DNA contacts. So the two provenances
do **not** share an input definition — exactly the asymmetry v26 Phase 1 must remove.

---

## 11. Multi-domain proteins cropped using motif-family knowledge (Finding I)

Recovered `/tmp/cluster_crops_v2.parquet` → `results/v26_audit/cluster_crops_v2_recovered.parquet`
(208 genes) and `results/v26_audit/06_multidomain_family_oracle_crops.csv`.

**103 / 208 genes (49.5%)** were cropped with a `reason` beginning `family:` — i.e. the cluster was
chosen because it matched the family the **motif database** assigns to that gene. Breakdown:

```
family:C2H2 71 · family:C2H2+len_tiebreak 11 · family:Nuclear_Receptor 9 · family:bZIP 2
family:Homeodomain 2 · family:C2H2C_type 2 · family:Homeodomain+len_tiebreak 1
family:CCCH+len_tiebreak 1 · family:GATA 1 · family:BED_zf+len_tiebreak 1 · family:FLYWCH 1
family:GTF2I+len_tiebreak 1
--- non-oracle: single 97 · largest 4 · at_hook_span 2 · largest+len_tiebreak 2
```

109 genes have >1 domain cluster; 17 are flagged `crop_ambiguous`. For an uncharacterised TF this
information is unavailable, so the current pipeline's input is not obtainable at inference time.

**Finding K:** `scripts/cluster_crop_v2.py:226` writes this table only to `/tmp`. It survived by luck
(dated 2026-07-21). v26 must persist it under `data/`.

---

## 12. Source-code reference index

| Claim | Reference |
|---|---|
| DBD truncation convention (`dbd_start=0`) | `scripts/legacy/build_augmented_dbd_dataset.py` docstring; verified: `dbd_start.unique()==[0]`, `dbd_end==len(seq)` 100% |
| `str_` DBD = 4.5 Å contiguous contact span | `scripts/build_deeppbs_structural_v2.py` docstring item 2 |
| Multi-domain crop uses motif-DB family | `scripts/cluster_crop_v2.py` docstring item 1; `:186-196,226` |
| PWM canonicalisation (IC 0.25 trim + strand rule) | `scripts/canonicalize_pwms.py:11-21` |
| Split: mmseqs + component graph + `excluded` | `scripts/legacy/build_training_table.py:196-278` |
| `filename` positional | `build_training_table.py:68,96` |
| N-chain partners, ≤3, ≤150 aa | `scripts/legacy/build_nchain_v23.py:27-29` |
| Chain concatenation with `<eos>` | `src/tfscope/data/dataset.py:547-566` |
| Single ESM forward, `<cls>` prepend | `src/tfscope/models/backbone.py:147-151` |
| `family_id` into MoE / head | `src/tfscope/models/tfscope.py:97,150,160,198,202` |
| Silent clipping (recog / contact) | `src/tfscope/data/dataset.py:630-633, 645-652` |
| ESM-C cache keyed by md5(sequence) | `src/tfscope/data/dataset.py:282-285` |
| Flank build + alignment fallback | `scripts/legacy/build_flank_dataset.py:28-45,60-80` |
| Flank contact re-index (`+dbd_start`) | `scripts/legacy/build_flank_contact_targets.py:1-8,28-48` |
| MyoD1 +9.17 checkpoint | `scripts/legacy/build_fig4a_switch.py:24,26` |
| Permutation-equivariant chain encoder (unused) | `src/tfscope/models/chain_set_encoder.py:1-28` |
| 2-D contact builder (real path) | `scripts/contact_teacher/build_contact_targets.py` |

---

## Open questions requiring a decision before Phase 1

1. **Test-set redefinition.** Fixing Finding A means the current 291-row benchmark cannot be the
   primary result — its inputs are structure-defined. The legacy manifest will be preserved for
   backward comparison, but the headline number will change. Confirm this is acceptable.
2. **Barrera relocation.** Locking the 20 Barrera genes out of train removes ~ several hundred
   homeodomain rows from the largest well-populated family. Expect a measurable drop in Homeodomain
   performance that is *correct* but will look like a regression.
3. **InterPro snapshot.** No pinned release exists. v26 must fetch once, store the raw responses,
   and record the release string — this is a one-time network dependency.
