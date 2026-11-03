# v24 ablation study

What each component of the v24 recipe is actually worth on the canonical 291-row
structure benchmark, measured as a paired delta against the full recipe run through
the same harness.

## Why this exists on top of the ICLR B0–B8 ladder

`iclr/variants.py` already answers "is the whole stack necessary" (B2–B4 simple
baselines, B5 no-MoE, B6 no-contacts, B7 chain-only, B8 = v24). It leaves three gaps:

1. **B6 removes all three contact mechanisms at once** — the 1-D attention supervision,
   the 2-D distillation KL, and the learnable head bias — so it cannot say which one
   carries the effect. A2/A3/A4 decompose it.
2. **There is no negative control for the contact block.** B6 shows removing contacts
   costs something; it does not show that what was removed was *structural knowledge*
   rather than a generic regulariser. A5 supplies that control.
3. **There is no data ablation.** Every rung trains on all 4 794 rows, so nothing
   addresses the standing objection that TFScope beats DeepPBS because it also trains
   on 4 641 sequence-only PWMs a structure model cannot consume. A1 addresses it.

## Variants

Everything not named is byte-for-byte `scripts/run_v24_contact_ddp.sh`, re-expressed
single-GPU as micro-batch 12 × accum 3 = the same global batch of 36. Identical to
`scripts/lofo/train_lofo_v24.sh` apart from the delta, so these numbers are comparable
to both the LOFO runs and the B-ladder.

| tag | delta from v24 | question |
|---|---|---|
| `A0_full` | none | same-harness reference for the paired deltas |
| `A1_deeppbsfold` | **byte-identical DeepPBS fold 0** (396 train + 81 valid NPZ) with **DeepPBS's masked-PWM labels** | does the win survive being handed exactly DeepPBS's data? |
| `A1b_structonly` | train/val restricted to structure-backed rows (936/144 instead of 4794/658) | cheaper regime-only version of A1; not part of the headline study |
| `A2_nodistill` | `--contact-distill-weight 0` | what is the 2-D distillation KL worth? |
| `A3_noattnsup` | drop `--v18-contact-supervision` | what is the 1-D recognition-residue attention loss worth? |
| `A4_nobias` | `--v18-contact-bias-scale 0 --v18-contact-bias-learnable 0` | what is the learnable head bias worth? |
| `A5_shufcontact` | all three kept, residue targets permuted | **control**: structure, or regulariser? |
| `A6_nolora` | `--lora-rank 0` | does the ESM-2 encoder need to adapt at all? |
| `A7_singlechain` | drop `--two-chain-input --chain-id-embedding --max-chains 4` | clean chain delta (B7 confounds chain with MoE + head removal) |

### A1 — a byte-identical copy of DeepPBS's folds, with DeepPBS's labels

Built by `scripts/ablation_v24/build_deeppbs_fold_dataset.py`; read its docstring for the
full derivation. In brief:

* **Fold membership verbatim.** `iclr_folds/train0.txt` (396) + `valid0.txt` (81) — the
  gene-disjoint retrain folds behind the published DeepPBS-vs-v24 tie. Each NPZ
  `<pdb>_<chain>_<motifid>.npz` resolves to exactly one row of
  `tf_pwm_deeppbs_v2.parquet` (the pre-dedup structural table TFScope's own `str_*` rows
  were carved from) for **all 477 entries** — 1:1, no ambiguity, no drops. The DBD crop
  the model sees is therefore the same crop TFScope uses everywhere else; only the row
  set changes. (The deduped table, whose row *i* is exactly `str_i`, covers only 350/477 —
  the pre-dedup table is what makes the copy exact.)
* **Labels verbatim.** Target = `Y_pwm[0]` restricted to `pwm_mask[0]`: only the PWM
  columns covered by DNA in the co-crystal. Strand 1 is the exact reverse complement of
  strand 0 (477/477), and the valid region is a contiguous block (477/477), so it slices
  onto TFScope's prefix-mask convention. Mean valid length 9.6 (range 5–19); 819 columns
  dropped in total. Not padding removal: in 303/477 rows the masked-out columns carry
  non-zero probability TFScope's own PWM would have supervised.
* **Test set untouched.** The 291 canonical `str_*` rows are copied verbatim (v23 PWMs,
  v23 partners) into the same parquet, and the split's `test` list is the canonical 291.
* **Contact supervision follows the data.** The v23 contact files are keyed by `str_*`, so
  without a remap the fold rows would get *zero* contact supervision and A1 would secretly
  become "DeepPBS data AND no contacts". They are re-keyed onto the `dpb_*` ids,
  transferring only what is exact: the 1-D recognition prior is column-independent and
  transfers verbatim (164 rows); the 2-D target is column-indexed and transfers only where
  the two label lengths agree 1:1 (54 rows, 173 dropped — no recoverable register between
  DeepPBS's masked window and the v23 PWM). For scale, v24 has 254 / 213 over 4 794 rows.

**Leakage, stated up front.** DeepPBS's folds were made disjoint from the 20 test genes
that have a co-crystal, not from all 51 genes of the 291-row benchmark. 16 test genes
(65 of 291 rows) appear in fold 0, plus 29 shared PDBs. The leakage flatters A1, so always
also run `make_ablation_table.py --leak-free`, which drops those genes from **every** model
and reports the 226-row / 35-gene clean subset. Byte-identical folds were the explicit
requirement; the clean subset is the honest read.

### A5 — the permuted-contact control

Two facts make this the most load-bearing run in the study:

* **The contact block touches ~5% of training rows.** Of 4 794 train rows, 254 have a
  recognition-residue prior and 213 have a 2-D contact target. Whatever B6's −0.074 is,
  it comes from a very thin slice of supervision.
* **v23's contact indices are known to be frame-shifted** for 192/273 structure rows
  ([[v23-contact-index-frame-shift]]), and many indices exceed the chain-1 token span
  outright (e.g. `str_326`: `L`=53, max residue index 18, actual sequence 16 aa), so
  they are silently clipped or land in a partner block. A large share of the "structural"
  supervision already points at the wrong residue.

So the control has to be tight. `build_ablation_assets.py` draws one permutation π of
`[0, len(chain1))` per row and maps every residue index `r → π(r)` if `r < len(chain1)`,
leaving out-of-range indices alone. That conserves, exactly: the number of supervised
residues, the number of supervised columns, every weight value, the total supervision
mass, and the clipping/partner-spill pattern. The **only** thing randomised is which
residue a PWM column is told to attend to. The same π is applied to both the 1-D prior
and the 2-D targets so they stay mutually consistent.

Reading it: if A2/A3/A4 each cost accuracy but A5 does not, the contact terms are a
regulariser, not structural knowledge, and the paper's contact-grounding claim needs
rewording.

### A3 vs A4 — what is and is not isolated

`recog_prior` is both a **loss target** and a **model input**: `TFDataset` builds it
whenever `--pwm-head-v18` is set, and `PWMHeadV18` uses it as an additive attention bias
whenever the bias is enabled. So:

* `A3_noattnsup` removes the supervision **loss** only; the prior is still fed as a bias.
* `A4_nobias` removes the bias **pathway** only; the supervision loss still applies.

Note the training/inference asymmetry this creates in the shipped model: v24 trains with
`--v18-contact-bias-scale 1.0 --v18-contact-bias-learnable 1`, but the canonical evaluator
(`iclr/score_checkpoint_unified.py`) runs sequence-only and passes no `recog_prior`, so
the bias is inert at test time. A4 tests whether removing that mismatch helps.

## Protocol

Scored through the frozen unified evaluator, identical to the B-ladder:

* **Panel A** `gene_content_r` — GT motif length, oracle shift/RC registration. Pure
  motif content.
* **Panel B** `gene_covR` — the model's own predicted gate span; coverage and
  gate-length error reported separately.

Both gene-balanced over the 291-row test set. Inference is sequence-only.

**Seed handling matters more than usual here.** v24's own five-seed spread on Panel A is
0.529–0.665 (sd 0.055), so an *unpaired* single-seed difference under ~0.11 is unreadable.
`make_ablation_table.py` therefore reports both:

* `mean ± sd` over seeds — the conservative, noise-inclusive view;
* a **seed-matched paired delta** with a hierarchical bootstrap 95% CI (resample genes,
  then rows within gene) — the estimator to quote, and the same one
  `iclr/paired_stats.py` uses for v24-vs-B0.

The paired lens is not cosmetic: at seed 42 the existing B6 sits at −0.074
[−0.115, −0.033] paired (significant), while the unpaired three-seed means differ by only
−0.028.

Target is 3 seeds (42, 1, 7) per variant, matching the B-ladder.

## Running it

```bash
# 1. derived inputs (structonly split + permuted contact files) — already built
scripts/ablation_v24/build_ablation_assets.py

# 2. mirror to /data1 (AFS tokens drop every ~10 min; long jobs must not read AFS)
scripts/ablation_v24/sync_ablation_to_data1.sh

# 3. drain a queue onto free GPUs, unattended, detached
scripts/v26/run_detached.sh abl_v24_queue \
  bash scripts/ablation_v24/run_ablation_queue.sh A1_structonly:42 A2_nodistill:42 ...

# 4. progress
scripts/ablation_v24/ablation_status.sh
scripts/ablation_v24/ablation_status.sh A1_structonly_s42 60     # tail one run's log

# 5. score + tabulate (CPU, will not preempt training)
scripts/ablation_v24/score_ablation.py
scripts/ablation_v24/make_ablation_table.py
```

The queue skips any run that already has a `ckpt_best.pt`, so appending seeds later is
safe — just re-run it with the full list.

Outputs:

* checkpoints `/data1/leihuang/TFScope_store/ablation_v24/checkpoints/ablation_v24/<tag>/seed<N>/`
* scores `results/ablation_v24/unified_ablation.json`
* table `results/ablation_v24/ablation_table.md`, summary `ablation_summary.json`

## Cost

~3 h (A1, 5× less data) to ~15 h (full-data variants) per run on one A6000; 8 variants ×
3 seeds = 24 runs. The queue takes whatever cards are free and holds the rest, so this is
a multi-day drain, not a single batch.
