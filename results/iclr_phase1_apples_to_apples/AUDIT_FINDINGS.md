# Phase-I apples-to-apples audit — findings

**Decision status: `CURRENT_DECISION = PENDING_B8`.**
No model has been evaluated against v24 under the same harness. The historical
v24 numbers (0.461 row / 0.523 gene-bal covR) come from `docs/` and must not be
used as a B8 substitute. Do **not** claim B0 beat v24 or that v24 failed.

## 1. Split hygiene (B0/B1) — clean

- train/test filename overlap: **0**
- gene_symbol overlap train∩test: **0** (gene-disjoint)
- test rows whose PWM is byte-identical to a train row: **0**
- B0 family prototype and B1 nearest-neighbor are both built from `split['train']`
  only; no test PWM/gene enters either. `_load(..., "train")` vs `"test"`.
- split hashes: train `06398a8bce961104`, test `9601fd382f8596fe` (291 rows).

## 2. B0 family-prototype confound (IMPORTANT, not leakage)

B0 groups by `family_id`, but `family_id` ≠ biological family for the two
zero-shot families:

| test family (name) | rows (% of test) | family_id | what B0 actually averages |
|---|---|---|---|
| p53 | 101 (35%) | 9 | **880 train rows all labeled "Other"** |
| POU | 46 (16%) | 4 | **1235 train "Homeodomain" rows** |

So for **51% of the test set**, B0's "family average" is a mismatched/coarse bin
(the "Other" grab-bag for p53; Homeodomain for POU). This is *not* test leakage,
but it means **B0-by-`family_id` is not a clean per-family prior floor** — its
0.539 is partly a coarse-bin artifact plus oracle-alignment inflation (§3).

Consequence for interpretation:
- v24 uses the *same* `family_id` taxonomy, so B0-by-`family_id` is at least on the
  same taxonomy as v24 (fair on that axis).
- But the honest "family prior floor" should be reported **two ways**:
  (a) by `family_id` (matches v24 conditioning), and
  (b) by corrected biological family, with a pre-registered **global-train
      fallback** for zero-shot families (p53/POU get a global-train mean, reported
      as a separate bucket — never a test-derived average).
- Neither current B0/B1 number is a final ranking.

## 3. Evaluation-protocol inconsistency B0/B1 vs v24 (must fix)

Both paths oracle-align (offset + reverse-complement) the prediction to the
target core (`align_pwm(consider_revcomp=True)`), so RC/shift is consistent.
The inconsistency is the **length/coverage mechanism**:

| | prediction length | coverage denominator | length penalty |
|---|---|---|---|
| **B0/B1** (`baselines._r_cov`) | full family-avg / NN PWM, no gate | target-core length | **none** — never penalised for wrong length |
| **v24** (`eval_full_metrics.panel_full`, predicted gate) | truncated to predicted span (gate>0.5) | target-core length | **yes** — short/long gate loses coverage |

So B0/B1 enjoy a free length advantage (no gate to get wrong) while v24 pays a
gate-length penalty. `covR = r_overlap × coverage` is therefore **not
comparable** across the two as currently computed.

Also to standardise: `ic_thresh` for target-core trimming (0.25 in both here,
confirm), gene aggregation (equal-weight mean over gene groups — consistent).

## 4. Fix = unified evaluator with two separated panels (`iclr/unified_eval.py`)

- **Panel A — oracle-content:** every model (baselines + v24 + Bx) uses the
  **GT motif length** and the **same** shift/RC registration; scores PWM content
  only. Removes the gate/length confound → the clean "is the motif right" number.
- **Panel B — end-to-end:** models use their predicted gate/length; baselines use
  a **pre-registered length policy that never reads the test target** (family/
  global-train median length); coverage, gate-length MAE, and length bias are
  reported **separately** rather than folded silently into one number.

## 4b. v24 checkpoint FOUND — B8 unblocked (corrects earlier note)

The canonical v24 checkpoint **is on this node**:
`/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/`
with both `ckpt_best.pt` and `config.json` (loadable, epoch 59, best oracle-r
0.4576, recipe matches the run script: v18 head + contact supervision +
N-chain max_chains=4 + residue MoE 8 experts + LoRA16). Earlier I had only
checked the AFS-symlinked `checkpoints/` tree (a different store). B8 can be run
via `iclr.score_checkpoint_unified --device cpu` (no GPU preemption).

## 4c. Panel-A evaluator bug for GATED models (caught via first B8 run)

First B8 run gave Panel A content_r=0.453 — LOWER than its Panel B covR=0.530,
which is impossible (covR = content_r x coverage ≤ content_r), and contradicts
v24's documented overlap-r 0.592. Cause: the adapter fed v24's **full 42-column**
PWM to the panels; `align_pwm`'s ±10 shift cannot reach a motif that sits late in
the padded tensor, so Panel A scored a wrong window. **Fix:** extract the
predicted gate span (`span_start:span_start+span_length`) before scoring, so both
panels see the actual predicted motif. Re-running B8 with the fix.
Panel B (0.530) had matched the documented covR (0.523) because the span is
near-left-anchored, but the extraction is now made explicit for both panels.

**Same harness, single seed (Panel A bug now FIXED — A ≥ B holds for all):**

| model | Panel A content_r | Panel B covR | coverage | gate_len_mae |
|---|---|---|---|---|
| B0 family-avg | 0.602 | 0.539 | 0.764 | 2.69 |
| B1 nearest | 0.519 | 0.384 | 0.811 | 2.69 |
| v24 (B8, seed42) | **0.629** | 0.530 | 0.795 | 3.52 |

Decomposition (the reason the two panels exist):
- **Content (Panel A): v24 (0.629) > B0 (0.602) > B1** — v24's learned motif
  content DOES beat the family-average floor once the length/coverage confound is
  removed. v24's Panel A (0.629) is consistent with the documented overlap-r 0.592.
- **End-to-end (Panel B): B0 (0.539) ≳ v24 (0.530)** — v24's better content is
  offset by a worse gate-length error (3.52 vs 2.69 bp), so end-to-end it only
  ties the family prior.

Still NOT a decision: single seed; B0 `family_id` confound (§2) inflates B0;
param-matched B2/B3 and the v24 ablations B5–B7 are not done; no multi-seed or
paired-bootstrap CIs yet; family-residual (§7 of the plan) not computed.
`CURRENT_DECISION = PENDING_FULL_AUDIT` (B8 single-seed now done).

## 5. Actions gated on B8 / running jobs

- B5–B7 not started yet (wave-2). Currently running: B2/B3/B4 (frozen ESM).
- B8 (frozen v24) requires the real checkpoint + `config.json` (see
  `b8_v24_checkpoint_search.json`). Historical doc numbers are **not** a substitute.
- No architecture work (Set Transformer / interface mixer / mutation head / v25)
  until: fixed B0/B1 eval, B2/B3 done, B5–B7 done, real B8 same-harness, family-
  residual metrics, and paired-bootstrap CIs are all complete.

## 6. Gate-swap, family_id shortcut, family-residual (from saved v24 preds)

**2x2 gate-swap** (gene_covR; B0 = exact-family content):

| content | length | gene_covR | coverage | len_mae | kind |
|---|---|---|---|---|---|
| v24 | v24 | 0.530 | 0.795 | 3.52 | deployable (v24 e2e) |
| v24 | B0  | 0.513 | 0.807 | 2.69 | diagnostic |
| B0  | v24 | 0.505 | 0.698 | 3.52 | diagnostic |
| B0  | B0  | 0.518 | 0.774 | 2.69 | deployable (B0 e2e) |

- Giving v24's content B0's length does NOT help (0.530 -> 0.513): v24's gate
  length is already well-matched to its own content. The gate-swap does **not**
  reveal a hidden end-to-end advantage; end-to-end v24 (0.530) ~ exact-B0 (0.518)
  (the +0.012 is n.s. per §5 CI).

**family_id shortcut:** corrupting family_id (rolled) leaves predictions almost
unchanged — pred corr true-vs-rolled = **0.995**, L1 = 0.003, covR 0.530 -> 0.518.
=> **v24 does NOT use family_id at inference; it is effectively metadata-free**
(the family-conditioning component is inert). B0, which DOES use the family
label, is therefore a *generous* comparator. A dedicated no-family-embedding
ablation is redundant (rolled-id already shows no effect).

**family-residual:** mean corr(v24 deviation-from-family-mean, true
deviation-from-family-mean) = **0.39** (n=272) => v24 DOES add within-family
specificity beyond the family prior. Whether this is ESM alone or v24's
specialized parts is unresolved until B2/B3 (simple ESM) and B5-B7 report.

## 7. Wave-1 simple baselines scored (frozen unified evaluator)

B2/B3/B4 finished 225 epochs and were scored through the frozen Panel A/B
evaluator (ckpt_best.pt = validation-selected; no test-best re-selection):

| model | Panel A content_r | Panel B covR | gate_len_mae |
|---|---|---|---|
| v24 (B8, seed42) | **0.629** | **0.530** | 3.52 |
| B0 coarse (family_id) | 0.602 | 0.539 | 2.69 |
| B0 exact-family (fair) | 0.584 | 0.518 | 2.69 |
| B3 frozen ESM + attn-pool (n=3) | 0.558±0.039 | 0.455±0.035 | 3.99 |
| B4 frozen ESM + span-gate (n=2*) | 0.538±0.028 | 0.457±0.038 | 3.85 |
| B1 nearest PWM | 0.519 | 0.384 | 2.69 |
| B2 frozen ESM + mean-pool (n=3) | 0.485±0.023 | 0.395±0.027 | 2.88 |
| B0 global | 0.484 | 0.323 | 2.69 |

(*B4 seed7 re-scoring on CPU; B4 n=3 shortly.)

**Q3 — does simple ESM match v24? NO.** v24 (0.629/0.530) is clearly above every
frozen-ESM readout (best B3 0.558/0.455) on both panels. The frozen-ESM heads
even fall below the family-average prior => v24's LoRA-tuning + structure add
value over a frozen readout. End-to-end, v24 still only ties the family prior
(0.530 vs B0 0.518-0.539; earlier CI n.s.), and its content edge over the prior
is p53/POU-driven.

**Operational note:** the detached driver died ~02:00 on a transient AFS glitch
("No module named iclr.variants" + "Permission denied"); wave-1 checkpoints were
intact but the driver's eval + wave-2 launch failed. Recovered: scored B2/B3/B4
via score_wave1.sh; relaunched B5/B6/B7 via run_wave2.sh (commands captured once
at launch to survive AFS hiccups). Dead CUDA device is index 9 (use pool 0-8).

`CURRENT_DECISION = PENDING_FULL_AUDIT` (need B5-B7 + multi-seed v24 + CIs).

## 8. Registration sensitivity: oracle-r naming + canonical-fixed cross-check

"oracle-r" (validation selector, `run_oracle_r_eval`) = coverage-aware,
gene-balanced covR with the PREDICTED gate, aligned by best offset + RC (peeks at
target for registration only). It is NOT oracle-length; misleading name — should
be "registration-oracle covR". Selection uses this (offset+RC) frame, but the
deployable frame is canonical-fixed (no peeking) — an inconsistency.

Canonical-fixed (no-peeking) cross-check on the SAME predictions (gene_covR):

| model | oracle Panel B | canon (pred-only) | canon (symmetric) |
|---|---|---|---|
| v24 (seed42) | 0.530 | 0.045 | 0.127 |
| B1 nearest | 0.384 | 0.070 | 0.087 |
| B0 exact-family | 0.518 | -0.015 | 0.057 |
| B0 coarse | 0.539 | — | 0.065 |
| B0 global | 0.323 | — | -0.007 |

Findings:
- Canonical-fixed collapses ~4-10x vs oracle and is UNSTABLE (3x for v24 between
  two reasonable target-canonicalization choices) => dominated by residual
  offset/strand misregistration, not motif content. This is why the harness uses
  oracle offset+RC (nuisance removal applied equally), not to inflate.
- Ranking is NOT robust to registration: under pred-only canonical, B1 (nearest
  training PWM) BEATS v24 (0.070 vs 0.045); v24's oracle lead does not survive the
  honest deployable frame.
- Implication: v24's apparent advantage over baselines is substantially a product
  of oracle registration on a registration-hard benchmark. Strong evidence that
  v24's complexity is not deployably justified. Reinforces PENDING_FULL_AUDIT.

## 9. FINAL Phase-I decision (single-seed-v24 caveat recorded)

DECISION: **KEEP_V24. Do NOT proceed to new architecture (Candidate A/B / v25).**

Endpoint (oracle Panel B covR): v24(1 seed) 0.530 | B7 0.485 | B6 0.477 | B3 0.455 |
B5 0.438 | B2 0.395. Family prior B0 0.518-0.539. Monomer/multimer: v24
0.564/0.494 vs B0_exact 0.535/0.483 (single seed).

Why keep v24 (no robust, reproducible, deployable component gain):
- No trained variant beats the family-average prior end-to-end; only single-seed
  v24 reaches it.
- Advantage is oracle-registration-dependent: under canonical (no-peeking) frame
  everything collapses ~4-10x and B1(nearest) > v24.
- MoE benefit non-monotonic (B7 removes more than B5 yet scores higher) =>
  co-dependent stack, not clean necessity.
- Content edge over prior is p53/POU zero-shot-driven & not significant; family
  head inert (family_id ignored at inference).
CAVEAT: v24 is single-seed (no CI on v24 itself); ablations are 3-seed. A formal
§7 CI needs >=3 v24 seeds (not run). Decision on point estimates + ablation SDs.

There is NO v25. Candidate A (chain_set_encoder) / B (interface_pair) remain
untrained module scaffolds; the gate to promote them did not pass. See
phase1_decision.json.

## 10. DeepPBS on the primary 291 benchmark — GENE-DISJOINT RETRAIN (done)

The earlier segfault was an ENV bug, not a DeepPBS bug: DeepPBS pins
torch 2.3.0+cu121 / PyG 2.5.0 / torch-cluster 1.6.3, but it was being run in the
`multiflow` env (torch 2.6.0+cu124, PyG 2.6.1). The torch-scatter/sparse/cluster
C++ extensions are ABI-locked to their build torch, so the mismatch crashed.
Fix: dedicated `deeppbs` conda env with the pinned stack (build_deeppbs_env.sh).

**Leakage was total.** Mapping DeepPBS's 523 training structures to genes
(gene-name + JASPAR-id + PDB-chain joins) shows ALL 20/20 primary-test genes that
have a co-crystal are in DeepPBS's training folds => the pretrained 0.806 is fully
gene-level leaky and must NOT be quoted.

**Fair experiment.** Retrained the DeepPBS 5-model ensemble on the 477 structures
whose gene is NOT in the 291 test set (gene-disjoint, matching TFScope's split;
paralogs kept, as in TFScope's split), then predicted on the 20 test-gene
structures. Scored through the SAME unified_eval as v24.
(`iclr/run_deeppbs_retrained.py`, `deeppbs_291_retrained.json`.)

| model (20 struct-having test genes) | Panel A content_r | Panel B covR |
|---|---|---|
| DeepPBS (gene-disjoint retrain, 5-model ens.) | 0.720 | 0.720 |
| v24 (same 20 genes)                           | 0.685 | 0.631 |

Paired over 20 genes: Δ(DeepPBS − v24) content_r = **+0.034, 95% CI
[−0.017, +0.086]** (crosses 0) — a **statistical tie** (DeepPBS wins 13/20).
covR is the wrong axis here: DeepPBS emits a full-length PWM and, like the B0/B1
baselines (§3), gets the free "no-gate length" advantage, so covR==content_r for
it; the fair axis is Panel A content_r.

Reading: on DeepPBS's home turf (co-crystal available; subset is 11 ETS + 7 FOX +
CLOCK + p53) the retrained structure model slightly edges sequence-only v24 but
not significantly. Two framing points stand: (i) v24 needs NO structure at
inference and covers all 51 test genes vs DeepPBS's 20; (ii) consistent with the
leakage-clean cluster40 comparison (TFScope 0.630 ~ DeepPBS 0.633). The 20-gene
subset is narrow and ETS/FOX-dominated, so it is a structural-subset sanity check,
not the headline benchmark.

## 11. v24 5-seed ensemble + seed variance (fair 5-vs-5 vs DeepPBS)

DeepPBS ships a 5-model ensemble, so single-seed v24 was an unfair comparator.
Trained 4 more v24 seeds (1,7,13,23) with the EXACT contact_v24 recipe
(run_v24_contact_ddp.sh: incl --two-chain-input --chain-id-embedding --max-chains 4;
single-GPU batch12 accum3 = global batch 36). NOTE: a first attempt trained the
seeds single-chain (dropped the multichain flags) and scored a bogus 0.505 —
discarded and retrained correctly.

**Seed variance (full 291, PanelA content_r):** seed1 0.665, seed42 0.629,
seed13 0.571, seed23 0.541, seed7 0.529 -> mean 0.587 ± 0.055. seed1 BEATS the
shipped seed42, so v24's 0.629 is a favorable-seed point estimate; honest v24 mean
≈ 0.59 ± 0.06. This is the CI the audit flagged as missing.

**5-model ensemble** (register-aligned averaging: each member commits to its motif
core, members aligned offset+RC to seed42's core, then averaged — naive averaging
in the padded 42-col frame gave a misleading 0.505):
- full 291: content_r 0.664, covR 0.560 (> single seed42 0.629 / 0.530).
- on the 20 DeepPBS-overlap genes: v24_ens5 0.707 vs DeepPBS-5model(fair) 0.731,
  paired Δ = -0.024, 95% CI [-0.072, +0.021] -> **statistical tie** (v24 wins 11/20).

Bottom line stands under every framing (single vs ensemble): sequence-only v24 TIES
retrained structure-based DeepPBS on the ~20 genes where a co-crystal exists, and
needs no structure while covering all 51 test genes. See v24_ensemble_summary.json.

## 12. Full metric suite (DeepPBS-5model vs v24-ensemble, 20 genes)

Beyond Pearson-r, a 9-metric battery (both oracle-registered to GT core the same
way; iclr/compare_full_metrics.py, full_metric_suite.json):

| metric | dir | v24_ens5 | DeepPBS | winner |
|---|---|---|---|---|
| pearson_r | ↑ | 0.707 | 0.694 | v24 |
| cosine | ↑ | 0.828 | 0.836 | DeepPBS |
| topbase_acc (consensus letter) | ↑ | 0.760 | 0.724 | v24 |
| auroc (base-enrichment) | ↑ | 0.790 | 0.748 | v24 |
| macroF1 (consensus 4-class) | ↑ | 0.633 | 0.568 | v24 |
| mae | ↓ | 0.171 | 0.153 | DeepPBS |
| rmse | ↓ | 0.242 | 0.221 | DeepPBS |
| jsd_bits | ↓ | 0.178 | 0.151 | DeepPBS |
| ic_mae | ↓ | 0.769 | 0.667 | DeepPBS |

Clean split matching training objectives: v24 wins every classification/ranking
metric (top-base, AUROC, macro-F1, Pearson) = gets the correct dominant/enriched
bases; DeepPBS wins every distribution/calibration metric (MAE, RMSE, JSD, IC-err,
cosine) = exact probability magnitudes, because it is trained with an L1(MAE) loss
directly on PWM values, whereas v24 optimizes shape terms (IC-PCC/top-base/cov-r).
Reading: v24 recovers motif IDENTITY better; DeepPBS recovers probability
MAGNITUDES better. AUROC label = target base prob > 0.25 (pooled base×column);
macro-F1 over per-column argmax base.

## 13. PRIMARY protocol = UNTRIMMED DeepPBS (v24's gate is learned)

Correction to §10/§11 framing. Those quoted DeepPBS IC-trimmed = 0.731. But
IC-trimming DeepPBS to its motif core uses the information-content of the answer to
localize the motif FOR DeepPBS — a clean-up v24 never gets, because v24's motif
localization is a LEARNED, scored gate. So the fair primary comparison uses
DeepPBS's actual RAW output (untrimmed); align_pwm still gives both models the same
oracle offset+RC registration, so DeepPBS is not penalised for "where", only judged
on its real prediction. (The §12 9-metric suite already used untrimmed DeepPBS.)

20 struct-having genes, PanelA content_r:
| model | content_r | vs DeepPBS-untrimmed |
|---|---|---|
| DeepPBS untrimmed (raw, PRIMARY) | 0.694 | — |
| DeepPBS IC-trimmed (generous upper bound) | 0.731 | (hand-localized) |
| v24 seed42 | 0.685 | Δ -0.009 CI[-0.060,+0.040] tie (v24 11/20) |
| v24 ensemble | 0.707 | Δ +0.013 CI[-0.034,+0.061] tie, v24 ahead (11/20) |

Bottom line (fair, untrimmed): single-seed v24 TIES DeepPBS; the 5-seed v24 ensemble
slightly EDGES it (0.707 vs 0.694), and wins every ranking/identity metric (§12:
top-base, AUROC, macroF1, Pearson) while DeepPBS keeps the calibration metrics
(MAE/RMSE/JSD/IC) it is directly trained on. 0.731 is retained only as a
generous-to-DeepPBS upper bound.
