# TFScope Mutation-Sensitivity Experiments

## Why
TF specificity is set by the DBD sequence; single-residue DBD mutations can re-program
it (disease variants, engineered swaps). A useful PWM model must be *mutation-sensitive*
— predict how a point mutation changes the motif (the "recognition code"). A model that
outputs the same PWM for WT and a specificity-altering mutant has learned family lookup,
not recognition.

## MyoD1 case (probe)
MyoD1 (class-II bHLH) binds the E-box; **L112R** in the basic region switches the
muscle E-box (CAGCTG, central GC) toward the canonical CACGTG (central CG) — a
one-residue → central-dinucleotide flip. Obligate heterodimer with an E-protein (TCF3).

## Barrera-2016 benchmark
55 single-residue human homeodomain variants (20 genes), PBM WT+MUT PWMs (rCLAMPS/CIS-BP);
incl. HOXD13 Q325K (position-50 switch, MyoD1 analog). Metric: predict WT & MUT PWMs,
Δpred = 1−corr(P̂_WT,P̂_MUT) vs measured Δtrue; corr(Δpred,Δtrue); directional accuracy;
14/55 "impactful" (Δtrue>0.2). Builder: `scripts/barrera_mutation_benchmark.py`.

## Baseline result — v24 is mutation-blind (on specificity)
| | mean Δpred | measured | corr | directional |
|---|---|---|---|---|
| all 55 | 0.005 | 0.180 | −0.05 | 40% |
| 14 impactful | 0.003 | 0.593 | −0.09 | — |

MyoD1: v24 gets WT E-box CAGCTG right but does NOT flip L112R→CACGTG. A 55-pair
plain-supervision fine-tune did not help (Δpred 0.008; held-out corr −0.07). Results:
`results/mutation_benchmark/SUMMARY.json`.

## E0 — layer-wise signal diagnosis (MyoD1 WT vs L112R)
| stage | relΔ @mut-pos | relΔ mean-DBD | cosine(WT,MUT) |
|---|---|---|---|
| ESM (backbone) | 0.354 | 0.174 | — |
| post-MoE residues | 0.394 | 0.179 | — |
| projection (pooled vector) | — | 0.157 | **0.988** |
| PWM attention | — | 0.431 | 0.903 |
| final PWM logits | — | 0.656 | 0.755 |

WT `CAGCTG|TTGGCC`, MUT `CAGCTG|TGCCCC`; per-col |Δlogit| concentrated in the 3′ FLANK
(pos 7–14, up to 5.2), core CAGCTG ~unchanged.

**Conclusion:** the mutation signal is NOT lost (survives ESM→MoE→attention→logits;
MoE does not smooth it). The pooled/global query path IS washed out (cos 0.988), but the
cross-attention-to-residues path preserves it and the output DOES change. The problem is
**decoder ROUTING**: the change lands in the flanks, not the specificity-determining
central base. So it's a decoder/recognition-code mapping issue, not signal loss — which
makes a **paired delta objective on centered logits (E1)** the right intervention.

## E1 — paired delta objective, freeze all but PWM head (`scripts/e1_paired_finetune.py`)
Supervise the WT→MUT difference directly in a shared registration frame:
`Δz_pred = centered(z_MUT) − centered(z_WT)`; loss = absolute-KL (keep) + 3·Δ-L1
+ 0.5·magnitude-match (anti-collapse) + 1·directional-cosine (impactful). Only
`pwm_head` trains (1.16M params); ESM/MoE/projection/gate frozen. 55 HD pairs split
fit 24 / val 9 (gene-disjoint) / held-out test 22.

**Result — NEGATIVE under honest model selection.**
| selection | held-out test corr(Δpred,Δtrue) | mean Δpred | WT-core r |
|---|---|---|---|
| baseline v24 | −0.15 | 0.008 | 0.521 |
| test-peek (step 300, 1st run) | +0.53 | 0.016 | 0.588 |
| **gene-disjoint val (step 150)** | **0.00** | 0.001 | 0.621 |

The corr=0.53 seen when evaluating *on the test set directly* is **test-set
overfitting** — it evaporates to 0.00 when the checkpoint is picked on a gene-disjoint
val split, and val corr never exceeds 0.11 (9 pairs, noisy). Crucially Δpred stays
**collapsed** (~0.001–0.026 vs measured 0.253) at every step, so even the "peek" gain
was correlating noise-level deltas. WT-core r *rose* (0.52→0.62) — the paired absolute
term is just extra HD supervision, not mutation learning.

**Conclusion:** an explicit paired delta objective does NOT make v24 mutation-sensitive
at this data scale. The bottleneck is DATA (24 train pairs, one family), not the
objective or a missing head — so **E2 (a dedicated MutationDeltaHead) would add params
to overfit the same 24 pairs and is not worth running before E3**. This is the clean,
rigorous limitation for the paper: seq-only PWM models are mutation-blind, and the
blindness is not fixable by objective/architecture design alone with available paired data.

## Phase-2 — where does the MyoD1 signal die? (`scripts/phase2_localize.py`)
Instrumented ESM→ResidueMoE→projection→contact-attn→PWM-logits for MyoD1 WT vs L112R
(DBD pos 11), 4 conditions. Target switch: WT CAGCTG (central **GC**, pos2=G/pos3=C)
→ MUT CACGTG (central **CG**, pos2=C/pos3=G).

| stage | WT/MUT cosine | reading |
|---|---|---|
| ESM, at mutation site (res 11) | **0.938** (L2 132) | signal strongly present |
| ResidueMoE, at site | **0.921** (L2 **155**, ↑) | MoE does NOT smooth it — amplified |
| projection (pooled vector) | 0.988 | global/query path washes out |
| contact-attention | 0.903 | per-residue KEY path preserves it |

Decoder output (normal): WT cons `CAGCTG`, MUT cons `CAGCTG` — **consensus does not flip**,
Δpred 0.162. BUT the **signed central-base log-odds move in exactly the correct direction**:
pos2 ΔlogOdds C=**+0.76**, G=−0.64 (G→C ✓); pos3 ΔlogOdds G=**+1.25**, C=−0.60 (C→G ✓).
So direction is right, **magnitude is collapsed** (shift too small to flip the argmax).

- **Oracle 1D contact (recognition_residues):** does NOT recover the switch — it makes
  WT/MUT *more* similar (Δpred 0.162→0.046), because forcing both onto the same contact
  residues erases the mutation's attention difference. → perception is **not** the bottleneck.
- **Force L112→attention (bias 8.0):** over-writes the whole motif (cons→CACCTG), an
  artifact of the heavy bias, not a clean switch. Uninformative.

**Conclusion / decision (per user rule):** the mutation signal is present and
*correctly-directed* all the way to the logits, and does NOT die in the MoE — so **no MoE
bypass**. Oracle contact still fails → the fix is to **replace/augment the PWM decoder**:
its residue→base recognition contribution is directionally right but magnitude-collapsed
(swamped by the family/global prior). This is the GO for Phase 3 — an explicit
recognition-energy decoder with a **low-capacity** family prior so the residue→base term
is not swamped. Repro: `python scripts/phase2_localize.py`.

## Phase-3 — bHLH recognition-energy decoder (`scripts/phase3_{prep,train,verify}.py`)
Minimal decoder `z[j,b] = r[fam][b,j] + λ_d·Σ_i C[j,i]φ(a_i,h_i)[b] + λ_s·Σ_{i,k}C[j,i]A[i,k]ψ(...)`,
frozen RAW ESM-2 (no v24 LoRA), **low-capacity family prior** r (4×24 params), soft
contact C (no hard mask), φ/ψ never read the family id (mutation flows only through the
residue reps). Trained on **465 native bHLH rows / 88 genes with MyoD1's cluster fully
held out** (registration fixed to a shared bHLH consensus). Module:
`src/tfscope/models/recognition_energy.py`. MyoD1 L112R tested **zero-shot** (no mutant label).

Go/no-go (target: WT CAGCTG → MUT CACGTG; Δ_switch = [S(CACGTG)−S(CAGCTG)]_MUT − _WT):
| model | WT covR (vs measured) | Δ_switch (→CACGTG) | central-base signed |
|---|---|---|---|
| WT-copy | — | 0 | none |
| FamilyCode (bHLH family-avg) | 0.869 | 0 (WT≡MUT) | none |
| v24 (Phase-2) | (CAGCTG ok) | ~0, no flip | correct dir, collapsed |
| **recog-energy direct** | **0.944** | **+1.81** (oracle +2.18) | pos2 G→C ✓, pos3 C→G ✗ |
| recog-energy full (2nd-shell) | — | +0.86 (oracle +1.85) | — |

**Verdict: PARTIAL GO (no hard no-go hit).**
- ✓ **WT absolute covR 0.944 > FamilyCode 0.869** — the decoder predicts MyoD1's WT E-box
  excellently zero-shot and beats the family-average baseline.
- ✓ **Beats v24 and WT-copy on direction** (Δ_switch +1.81 vs ~0) across all 4 configs;
  oracle contact consistently strongest. Δpred stays *small* (0.03) → the change is
  specific, not diffuse.
- ✗ **Switch is only half-correct**: central pos2 moves G→C ✓ but pos3 moves the wrong way
  (stays C, want G) → model drifts toward CAC**C**TG, not CAC**G**TG; consensus does NOT flip,
  **magnitude still collapsed**.
- No no-go triggered: oracle beats WT-copy; direction not wrong (pos2 right); MyoD1 fully
  held out; recog-energy > FamilyCode.

**Reading:** the recognition-energy decoder FORM is validated — explicit low-rank residue→base
energy with a low-capacity prior gives excellent zero-shot WT PWMs AND a correctly-directed
(if incomplete) mutation response that v24's free decoder cannot. The residual gap (pos3,
magnitude) is exactly what **Phase 5 counterfactual equivariance** (enforce full-forward
centered Δ ≈ local mutation transport, with magnitude + reverse/path consistency) targets.
Repro: `python scripts/phase3_prep.py && python scripts/phase3_train.py && python scripts/phase3_verify.py`.

## Phase-4 — go/no-go battery (`scripts/phase4_battery.py`)  →  **GO**
Train recog-energy on 70 bHLH genes, hold out 18 genes (cluster-disjoint) + MyoD1 (zero-shot).

(A) **Gene-held-out WT covR** (96 rows / 18 genes):
| method | WT covR |
|---|---|
| **recog-energy** | **0.878 ± 0.123** |
| FamilyCode (family-avg) | 0.798 ± 0.176 |
| nearest-paralog | 0.688 ± 0.319 |

recog-energy beats FamilyCode on **73/96 (76%)** genes → clears the FamilyCode no-go gate.

(B) **MyoD1 mutation battery** (zero-shot; neutral calibration):
| mutation | Δ_switch | Δpred |
|---|---|---|
| L112R (specificity) | **+0.56** | 0.016 |
| L112K (unseen substitution) | +0.14 | 0.006 |
| neutral@40→A | −0.03 | 0.002 |
| neutral@35→G | +0.04 | 0.004 |

Specificity residue shift is positive/correct and ~8–14× the neutral mutations (≈0) → strong
neutral calibration + specificity; an unseen substitution (L112K) still shifts correctly.

**Verdict: GO.** ✓ beats WT-copy, v24, FamilyCode, nearest-paralog; ✓ correct signed direction;
✓ neutral-calibrated & specific; ✓ WT accuracy high and generalizes gene-held-out; ✓ holds for
an unseen substitution. No no-go triggered. **Remaining gap:** absolute Δ magnitude still small
(consensus doesn't flip; pos3 direction from Phase-3) → carry to **Phase 5 counterfactual
equivariance**. Repro: `python scripts/phase4_battery.py`.

## Phase-5 — counterfactual equivariance (`scripts/phase5_equivariance.py`)
No matched bHLH mutant PWMs exist, so the mutation transport is trained on NATIVE bHLH
PAIRS (both labelled): `Δz_pred = center(F(S2))−center(F(S1)) ≈ center(logPWM2)−center(logPWM1)`
in the shared consensus frame (between-gene E-box differences are real, correct-magnitude
ΔPWMs — MAX CACGTG vs myogenic CAGCTG teaches the central-dinucleotide code; MyoD1 cluster
held out). Loss = L_abs + 2·L_eq + 0.5·L_id(neutral). MyoD1 L112R stays zero-shot.

| | Δ_switch | Δpred | pos2 (want G↓,C↑) | pos3 (want C↓,G↑) | held-out WT covR |
|---|---|---|---|---|---|
| abs-only (control, same budget) | +0.45 | 0.018 | G **+0.27** ✗, C −0.09 | G +0.03, C −0.08 (flat) | 0.870 |
| **+ EQUIVARIANCE** | **+1.22** | 0.068 | G **−0.43** ✓, C −0.06 | **G +0.35, C −0.49** ✓ | 0.874 |

**Result — equivariance works as designed:**
- **pos3 direction FIXED** (Phase-3's failure): now C↓/G↑ correctly (C −0.49, G +0.35) toward CACGTG.
- **pos2 improved**: the WT base G is now suppressed (−0.43) instead of *increased* (+0.27 control).
- **Magnitude un-collapsed ~3×** (Δ_switch +0.45→+1.22; Δpred 0.018→0.068), still correctly directed.
- **No WT regression** (held-out covR 0.870→0.874); identity/neutral loss stays ≈0 (calibration kept).
- Consensus still does not fully flip (both central argmaxes don't all move at once), but BOTH central
  bases now move in the CACGTG direction — the signed-central-base go-condition is met.

**Verdict: GO** — counterfactual equivariance, trained only on native pairs, corrects the direction
(esp. pos3) and roughly triples the magnitude zero-shot without hurting WT accuracy. Repro:
`python scripts/phase5_equivariance.py`. Next: Phase 6 (binary weak-sup, JS effect-score) / Phase 7
(HD extension) / Phase 8 (fold back into full TFScope with WT distillation).

## Phase-7 — homeodomain (second mechanism) (`scripts/phase7_{prep,train_eval}.py`)  →  **GO**
Train recog-energy (+ equivariance on native HD pairs) on 926 native HD rows / 149 genes with a
gene-held-out set AND all Barrera-gene clusters excluded; test Barrera 55 pairs zero-shot.
| | recog-energy WT covR (held-out HD) | FamilyCode | paralog | Barrera AUROC(Δpred, spec.change) |
|---|---|---|---|---|
| equiv | 0.863 | 0.733 | 0.736 | **0.624** |
| abs-only | 0.871 | 0.733 | 0.736 | **0.636** |
| v24 (prior) | — | — | — | 0.502 (chance) |

- **WT covR 0.86 ≫ FamilyCode/paralog 0.73** — decoder form generalises to HD (clears no-go gate).
- **Barrera AUROC 0.62–0.64 > v24's 0.502** — recog-energy DISCRIMINATES specificity-changing from
  neutral HD mutations zero-shot (Barrera clusters held out); v24 was at chance. meanΔ Yes>No both.
- Note: on HD, equivariance does NOT beat abs-only on the Barrera AUROC (0.624 vs 0.636) — HD has ~2.5×
  more native training data than bHLH, so the recognition-energy FORM alone already yields mutation
  sensitivity; equivariance is the decisive lever when native data is scarce (bHLH). Honest nuance.

**Verdict: GO for both mechanisms** (bHLH E-box + HD TAAT). Per the plan this unlocks the
cross-recognition-mechanism claim and Phase 8 (fold decoder into full TFScope with WT distillation).
Repro: `python scripts/phase7_prep.py && python scripts/phase7_train_eval.py`.

## Phase-8 Stage-1 — fold into full TFScope (`scripts/phase8_integrate.py`)  →  **STRONG SUCCESS**
Stage 1 = "train only the new decoder": RecognitionEnergyDecoder runs on **v24's own frozen
encoder features** (post-MoE residue reps, 1280-d) — replacing the free PWMHeadV18 regression path
while keeping v24's encoder / N-chain / span gate. Combined bHLH+HD (family prior n_fam=2, low
capacity). Loss = L_abs + 2·L_eq (equivariance on native pairs) + 0.3·L_dist (KL to v24's predicted
WT PWM, anti-regression distillation). 1374 train / 228 held-out rows.

| metric | recog-energy on v24 feats | v24 (own PWMHeadV18) |
|---|---|---|
| **held-out WT covR** (n=228) | **0.792** | **0.471** |
| MyoD1 L112R Δ_switch (bHLH) | **+2.42** (Δpred 0.087) | ~0 |
| Barrera AUROC(Δpred, spec.change) (HD) | **0.692** | 0.502 |

**Headline: the recognition-energy decoder is a far better PWM head than PWMHeadV18 even on v24's
OWN features** — held-out WT covR 0.792 vs 0.471 (+0.32, same rows/metric) — AND it adds the mutation
sensitivity v24 lacks (MyoD1 Δ_switch +2.42, the strongest yet; Barrera AUROC 0.692, the best yet).
So folding the decoder in *replaces* the free regression path with a better one in Stage 1 alone
(encoder still frozen). Note recog-on-v24-feats (0.792) < recog-on-raw-ESM (0.86 in Phase-4/7): v24's
post-MoE features are a slightly worse substrate than raw ESM → either keep a raw-ESM branch or let
stages 2–4 adapt the encoder. (WT consensus display "GCAACA" is a core-window artifact; Δ_switch
scans all placements/orientations and is robust — verify WT E-box recovery separately.)
Artifacts: `results/mutation_benchmark/phase8_integrate.json`, ckpt
`/data1/leihuang/TFScope/phase8_recog_energy.pt`, feats `phase8_feats.npz` (cached).
**Next (ATTENDED):** Stage 2–4 encoder unfreeze (projection → MoE-local → few LoRA) + Phase-9
ICLR pack (structured-transport module, full metrics, ablations). Repro: `python scripts/phase8_integrate.py`.

## Phase-9 — ICLR metric suite (`scripts/phase9_metrics.py`)  [in progress]
Full "must-report" metric list on the integrated Phase-8 model over Barrera HD (55 pairs, held out).
**Trustworthy (within-experiment):** WT covR **0.845**, MUT covR **0.784**, effect AUROC **0.692**
(v24 0.502), AUPRC **0.463** (prevalence 0.255), neutral meanEffect 0.0088 (No) vs 0.0129 (Yes).
**Cross-platform-confounded → all ≈ chance:** centered-delta corr −0.01, per-position −0.01,
directional 0.49, signed-base-change 0.45, consensus-switch 0.35, magnitude ratio 0.03.

**Key methodological finding:** every *signed-delta-vs-measured* metric is unusable on Barrera HD
because WT (CIS-BP) and MUT (Barrera-PBM) are cross-platform — consistent with the earlier result
that the measured Barrera Δ itself anti-tracks spec.change (AUROC 0.31). So the ICLR eval must:
(1) use the within-experiment **spec.change** label (AUROC/AUPRC) for HD phenotype, (2) validate
**signed-delta** metrics on held-out **native same-platform pairs** (between-gene ΔPWMs, the
equivariance target — clean) and on literature switch cases (MyoD1 CAGCTG→CACGTG), NOT on the
cross-platform Barrera deltas. Brier/ECE (0.21/0.20) are descriptive (raw normalized effect, no Platt fit).

**Signed-delta on CLEAN held-out native same-family pairs** (`scripts/phase9_native_delta.py`) —
where between-gene ΔPWMs are same-platform, the transport metrics are strong (vs ≈chance on
cross-platform Barrera), confirming the model is NOT magnitude-collapsed on real specificity
differences — Barrera's cross-platform deltas were the problem, not the model:
| metric (held-out native pairs) | bHLH (n=398) | HD (n=400) |
|---|---|---|
| directional accuracy | **0.894** | **0.905** |
| centered-delta corr | 0.452 | 0.414 |
| per-position delta corr | 0.489 | 0.398 |
| pred/true magnitude ratio | 0.527 | 0.390 |
| signed-base-change acc | 0.509 | 0.506 |

**Consolidated Phase-9 picture:** WT covR 0.79–0.85 / MUT covR 0.78 (≫ v24 0.47, FamilyCode 0.73);
HD effect AUROC 0.692 (v24 0.502); transport directional acc ~0.90 + Δcorr ~0.43 + magnitude ratio
~0.4–0.5 on clean native pairs (both families); MyoD1 single-residue switch Δ_switch +2.42 with
equivariance fixing pos3; neutral mutations ≈0.

**Structured-transport module — falsifying test (`scripts/phase9_transport.py`).** φ(a,h) is a
potential, so LOCAL transport (flip only the AA one-hot at residue k, keep WT ESM context h) is a
potential difference → identity/reverse/path exactly 0. Full-forward transport is a difference of the
state fn cent(z_x), so it ALSO telescopes to 0 on identity/reverse/path. **⇒ reverse/path/identity
are NON-discriminating (trivially satisfied by any difference-based transport); the Phase-5 reverse/
path losses were effectively no-ops.** Decisive result at MyoD1 pos 11:
| transport | identity | reverse | path | L112R Δ_switch |
|---|---|---|---|---|
| full-forward (ESM re-embeds) | 0 | 0 | ~0 | **+2.42** |
| local (AA one-hot only, WT h) | 0 | 0 | 0 | **+0.001** |

**⇒ The mutation signal lives entirely in ESM's RE-EMBEDDING of the mutant, NOT in the explicit AA
channel of φ.** Flipping the one-hot alone does ~nothing. So the model's mutation sensitivity is
**ESM-contextual-readout, not a local/additive/composable recognition code** — one of the plan's
anticipated falsification outcomes ("local/additive recognition-code 假设不成立"). Honest reframing
for the paper: the recognition-energy *decoder form* (low-rank, contact-weighted, low-capacity prior)
+ equivariance (native-pair Δ supervision) delivers the WT accuracy and zero-shot mutation direction/
magnitude — but the effect is carried by ESM, and structured local-transport consistency is not the
lever. To force a truly composable code one must bottleneck h or directly supervise the local (AA-
channel) transport — a concrete future direction, not a current claim.

Remaining Phase-9: consolidated ablation table; ASCL2/HES2, HD-Q50K, C2H2 cases; Platt-fit
calibration. Repro: `python scripts/phase9_metrics.py && python scripts/phase9_native_delta.py &&
python scripts/phase9_transport.py`.

## AUDIT (Section I) — direct MyoD1 L112R output (`scripts/audit_myod1.py`)
Integrated Phase-8 decoder, actual PWM columns (not just Δ_switch):
- WT E-box = **CAGCTG** ✓ (earlier "GCAACA" was a window-display artifact — verified).
- MUT E-box = **CAGCTG — does NOT flip to CACGTG.**
- pos2 (WT G→C): Δcentered-logit C **+0.77**, G **−0.78**; G-prob 0.48→0.29 (≈tied w/ C) — CORRECT dir.
- pos3 (WT C→G): C decreases but mass → **A**, not G (G stays 0.02) — WRONG target base.
- change is localized at central E-box (pos2 dominant) → not arbitrary; oracle contact ≈ normal (+2.19).
- **Verdict: +2.42 = correct direction + correct position + INSUFFICIENT magnitude + incomplete (pos3
  fails). CAGCTG→CACGTG is NOT fully recovered** on the integrated (v24-feature) model. MUT still
  prefers CAGCTG (S 4.20 > 2.73). Raw-ESM Phase-5 model fixed pos3 (v24 feats are a worse substrate).
  Per go/no-go this is a CONCERN for the ICLR main line — pending the ablation/simple-ESM audit.

## AUDIT (Section III) — consolidated ablation (`scripts/audit_ablation.py`)
bHLH, raw ESM, MyoD1 zero-shot, identical protocol. (native = held-out native-pair signed-Δ.)
| variant | WT covR | MyoD1 Δsw | pos2 | pos3 | native dir | native Δcorr | native mag | neutral Δsw | params |
|---|---|---|---|---|---|---|---|---|---|
| v24 (cited) | 0.47 | ~0 | — | — | — | — | — | — | full |
| WT-copy | — | 0 | — | — | 0.50 | 0 | 0 | 0 | 0 |
| FamilyCode | ~0.80 | 0 | — | — | 0.50 | 0 | 0 | 0 | 0 |
| **simple_esm_mean** | **0.871** | **+3.65** | ✓ | ✓ | 0.81 | 0.31 | 0.40 | −0.56 | **88k** |
| simple_esm_attn | 0.869 | +0.77 | ✓ | ✓ | 0.82 | 0.27 | 0.41 | +0.19 | 88k |
| recog_full | 0.866 | +1.58 | ✓ | ✗ | 0.86 | 0.28 | 0.39 | −0.47 | 169k |
| recog_equiv | 0.871 | +1.00 | ✗ | ✓ | 0.85 | **0.47** | **0.53** | +0.10 | 169k |
| recog_noAA | 0.860 | +1.75 | ✓ | ✗ | 0.85 | 0.28 | 0.41 | −0.28 | 169k |

**Answers to the audit questions:**
1. **Does a simple ESM head match the full model? YES.** WT covR is identical (0.871 / 0.866 / 0.871),
   and simple_esm_mean gives the *largest* MyoD1 switch (+3.65) with BOTH central bases correct.
2. **Does the recognition-energy decoder add independent gain? NO on WT covR or the switch; a small edge
   on native-pair directional acc (0.86 vs 0.81) only.**
3. **Is native-pair delta supervision the main lever? Only for the DELTA metrics** — recog_equiv lifts
   native Δcorr 0.28→**0.47** and magnitude 0.39→**0.53** and neutral −0.47→+0.10, but does NOT improve
   WT covR or the single MyoD1 switch (+1.00 < +1.58).
4. **Is the explicit AA channel used? NO** — recog_noAA (φ without AA one-hot) ≈ recog_full (covR 0.860,
   Δsw +1.75, native dir 0.85). Removing it changes almost nothing → confirms the Phase-9 transport finding.
5. **Reverse/path/identity losses: algebraic no-ops** (state-difference telescoping), never real gradients.
6. **Contact grounding: oracle ≈ learned** (audit_myod1 +2.19 vs +2.42) → no significant contribution.

**MECHANISTIC VERDICT: the gain is ESM contextual re-embedding, not the recognition-energy architecture.**
A 88k-param mean-pool MLP on frozen ESM matches the full 169k recognition-energy model on WT covR (0.87,
≫ v24 0.47) AND on the MyoD1 switch. The structured decoder / explicit AA channel / contact grounding /
reverse-path losses add no clear independent value; only native-pair Δ-supervision has an isolated, real
effect on delta correlation/magnitude + neutral calibration.

**GO/NO-GO (Sections VII–VIII) → conclusion (B).** A simple ESM head is comparable to the full model, and
MyoD1 CAGCTG→CACGTG is not fully recovered (best is CAC**C**TG). Reposition the paper away from a structured
recognition-code claim toward: *"Contextual PLM representations give strong zero-shot WT PWMs (≫ v24) and
contain mutation-ranking information (native directional acc 0.81–0.86, HD effect AUROC 0.69), but reliable
consensus-level mutant PWM prediction remains unresolved; a structured recognition-energy decoder adds no
clear independent value over a simple ESM head."* Repro: `python scripts/audit_ablation.py`.

## AUDIT v2 — APPLES-TO-APPLES (`scripts/audit_full.py`, 3 seeds, bootstrap CI, identical bHLH test set)
The prior v24=0.47 (228 bHLH+HD rows) vs simple=0.87 (96 bHLH genes) was NOT the same samples. Recomputed
on the IDENTICAL held-out bHLH genes with the same covR metric (v24 via cached phase8 predictions). MyoD1's
whole cluster excluded from training; test = gene-held-out bHLH; MyoD1_wt_seq NOT in train.
| model | WT covR | 95% CI | MyoD1 Δsw | pos3 top=G | **consensus=CACGTG** | native dir | params |
|---|---|---|---|---|---|---|---|
| **v24** | **0.383** | — | ~0 | — | — | — | full |
| FamilyCode | 0.790 | — | 0 | — | 0/3 | 0.50 | 0 |
| **simple_esm_mean** | **0.880** | [.815,.934] | 3.65 | 1/3 | **0/3** | 0.834 | 88k |
| simple_esm_attn | 0.879 | [.822,.928] | 0.58 | 0/3 | **0/3** | 0.849 | 88k |
| free_attn+delta | 0.856 | [.790,.907] | 1.09 | 0/3 | **0/3** | 0.832 | 88k |
| recog_full | 0.868 | [.816,.914] | 0.83 | 0/3 | **0/3** | 0.812 | 169k |
| recog_equiv | 0.864 | [.803,.912] | 2.54 | 0/3 | **0/3** | 0.821 | 169k |
| recog_noAA | 0.877 | [.819,.923] | 1.26 | 0/3 | **0/3** | 0.85 | 169k |

**Decisive, apples-to-apples:** (1) WT covR of all trained heads 0.86–0.88 with **overlapping CIs**, ≫ v24
**0.383** (same samples) > FamilyCode 0.790 — simple mean-pool head is at the top; recognition-energy adds
NO WT advantage. (2) **consensus=CACGTG in 0/3 seeds for EVERY model** — the MyoD1 switch is never completed;
Δsw is correct-direction-only. (3) native directional acc ~0.81–0.85 across all; `recog_noAA` highest (0.85)
→ AA channel unused. **Confirms conclusion (B).** (pos2/pos3 "✓" in the older AUDIT-III table meant signed
DIRECTION, not top-base/consensus — resolved here.) Repro: `python scripts/audit_full.py`.

**MyoD1 per-column (Part B) — which base fails.** WT E-box = CAGCTG (all models). Best MUT is **CACCTG**
(simple_esm_mean: pos2 flips G→C, C=0.457>G=0.357). **pos3 is the failure**: want C→G, but G-prob peaks at
0.328 (recog_equiv) / 0.178 (simple) and C stays dominant (0.53–0.69). NO model reaches CACGTG; best is one
base short (CAC**C**TG). recog_full+oracle even distorts the WT to CACCTG. So +3.65/+2.54 = correct direction,
pos3 magnitude insufficient.

**ESM embedding-intervention (Part C) — signal is DISTRIBUTED, AA channel unused.** MyoD1 Δ_switch vs
replacement radius (swap WT→MUT ESM embeddings around site 11):
| radius (swap) | simple_esm_mean | recog_equiv |
|---|---|---|
| site only | 0.14 | 0.76 |
| ±1 | 0.27 | 0.75 |
| ±3 | 0.64 | 0.75 |
| ±5 | 1.01 | 0.75 |
| whole DBD (full mutant) | **3.65** | **1.46** |
| AA one-hot only (WT emb) | **0.0** | **0.0** |

simple head: site-only captures only 4% of the full effect → the mutation signal is **distributed across the
whole DBD** by ESM's re-encoding. recog (contact-gated): site+contacts carry ~half (0.76), whole-DBD the rest.
**AA-one-hot-only = 0.0 in both → the explicit AA channel is entirely unused.** (Caveat per plan: WT-context +
swapped-site embeddings are OOD; this diagnoses where the signal lives in the representation + AA-channel usage,
NOT that no local biological code exists.)

**AUDIT COMPLETE → conclusion (B), rigorously.** Mechanism: mutation sensitivity = reading ESM's distributed
re-encoding of the mutant sequence; recognition-energy decoder / explicit AA channel / contact grounding /
reverse-path losses add no independent value; a simple ESM head matches the full model apples-to-apples; and
no model recovers CACGTG. Still NOT done (matched-data-limited): strict position/substitution splits and
ASCL2/HES2·Q50K·C2H2 case studies (only Barrera HD has multi-mutation data, cross-platform).

## E3-eval — cross-family phenotype benchmark (`scripts/barrera_crossfamily_*.py`)
The trustworthy Barrera signal is the binary **`spec.change`** label (within-experiment
ref-vs-alt 8-mer specificity change), available for 117 variants / 41 genes across
families — NOT the PWM deltas (WT from CIS-BP vs MUT from Barrera-PBM are cross-platform;
`AUROC(Δtrue_PWM, spec.change)=0.31`, i.e. the measured PWM deltas *anti-track* the
labels — which is why the E1 PWM-delta objective could not learn). So we score the model
by whether its **Δpred = 1−corr(P̂_WT,P̂_MUT)** over the WT gate core ranks spec-changing
mutations above neutral ones. WT/MUT DBD crops built from the trusted training crop +
CIS-BP/UniProt full proteins (`assert full[pos−1]==from`); 101 pairs / 7 families.

| stratum | AUROC(Δpred, spec.change) | n | note |
|---|---|---|---|
| **55 HD pairs (S6-matched)** | **0.502** | 55 | chance |
| cross-family, in-DBD | **0.556** | 85 | ≈ chance overall |
| — Homeodomain | 0.492 | 62 (16 Yes) | chance; misses Q50K switch |
| — **C2H2-short (KLF/EGR)** | **1.000** | 8 (5 Yes) | suggestive real signal, small n |
| — C2H2-medium | 0.200 | 6 (1 Yes) | noisy |
| — Forkhead / C2H2-long / NR | n/a | 0 Yes each | no positives to score |

**Conclusion (refined):** v24 is *broadly* mutation-blind (in-DBD AUROC 0.556 ≈ chance;
Δpred collapsed <0.02 vs measured ~0.25 everywhere). But blindness is **family-dependent**:
chance on homeodomains yet a real (if underpowered, n=8) discrimination on **C2H2-short
zinc fingers**, consistent with modular ZF recognition being more locally ESM-encoded than
distributed homeodomain effects. The model can *rank* ZF spec-changers but still does not
produce mutation-*sized* changes. This sharpens the honest limitation and points E3-train
at C2H2/DMS data where the signal is learnable.

## Experiment order (status)
- **E0** layer diagnosis — DONE (signal survives to logits, routed to flanks).
- **E1** paired delta loss, freeze-but-head — DONE, **negative** (overfits, data-limited).
- **E2** MutationDeltaHead — DEPRIORITIZED (more params, same 24 pairs → worse overfit).
- **E3** expand mutant/PBM data (full Barrera ~1000 + DMS across families) — the real
  lever; a data-acquisition task, not a quick fine-tune.
- **E4** staged unfreeze — only meaningful once E3 provides data.

Scripts: `scripts/barrera_mutation_benchmark.py` (benchmark), `scripts/e1_paired_finetune.py` (E1).
