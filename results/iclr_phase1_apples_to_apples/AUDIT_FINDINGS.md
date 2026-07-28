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

**Trustworthy so far (same harness, single seed):** Panel B end-to-end covR —
B0 0.539, B1 0.384, v24 0.530. i.e. v24 ≈ B0 on end-to-end covR. This is NOT a
final ranking (single seed, B0 family_id confound §2, Panel A pending, no CIs).

## 5. Actions gated on B8 / running jobs

- B5–B7 not started yet (wave-2). Currently running: B2/B3/B4 (frozen ESM).
- B8 (frozen v24) requires the real checkpoint + `config.json` (see
  `b8_v24_checkpoint_search.json`). Historical doc numbers are **not** a substitute.
- No architecture work (Set Transformer / interface mixer / mutation head / v25)
  until: fixed B0/B1 eval, B2/B3 done, B5–B7 done, real B8 same-harness, family-
  residual metrics, and paired-bootstrap CIs are all complete.
