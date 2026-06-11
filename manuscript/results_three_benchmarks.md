# Results — Benchmarks

TFScope predicts a transcription factor's DNA-binding specificity — a position weight
matrix (PWM) — from the protein sequence alone, without any structure of the protein–DNA
complex at inference. To establish that this sequence-only setting matches or exceeds the
structure-based state of the art, and to characterize honestly where it generalizes and
where it does not, we evaluated TFScope under three benchmarks of monotonically increasing
stringency: (i) a head-to-head, leakage-controlled comparison against the structure-based
method DeepPBS on its own 130-TF blind split; (ii) **cluster40**, a clustered hold-out at
40% protein identity that removes near-homologue shortcuts; and (iii) **leave-family-out
(LFO)**, in which an entire structural family is withheld, probing transfer to a fold the
model never saw during training. Throughout, we draw a sharp line between *oracle-aligned*
scores (which grant every method — including DeepPBS — the correct motif offset and strand,
and so serve only for fair *ranking*) and *deployable* scores (fixed registration, no
alignment), and between models trained or retrieved through a leakage-free leave-gene-out
(LGO) index (honest) versus the legacy index that permits same-source donors (leaky). Only
honest numbers are reported as headline claims; leaky numbers are labelled as such and used
solely as upper-reference anchors.

## Benchmark 1 — Head-to-head against the structure-based state of the art

On base composition — the core quantity the seed model is asked to predict — honest,
leakage-free TFScope matched or exceeded DeepPBS despite using no structure at inference.
We scored every method, including DeepPBS, with one identical protocol: each predicted PWM
was trimmed to the target's informative core (per-column information content, IC ≥ 0.25) and
granted oracle offset-plus-reverse-complement alignment, so that all methods compete on a
fair *ranking* footing (absolute values are upper bounds, applied equally to all). On the
116 TFs of the DeepPBS blind split covered by all methods, honest leakage-free **TFScope**
beat DeepPBS on 8 of 11 metrics, including mean and median per-column Pearson *r*, top-1 base
accuracy, MCC, F1, AUC, and both error metrics (Table 1). DeepPBS retained an advantage only
on the two *calibration* metrics, cross-entropy and KL divergence, and — as shown below — on
deployable registration.

**Table 1. Motif-level metrics (trimmed core IC ≥ 0.25, oracle offset+RC aligned; fair
ranking, upper-bound absolute values). 116 DeepPBS-covered TFs.** Honest LGO TFScope vs
structure-based DeepPBS. The ablation column (TFScope with the contact-aware branch disabled)
isolates the contribution of that branch: it supplies the entire +0.10 mean *r*, from only
0.55 M added parameters, with the rest of the model frozen. «Fig. 1»

| Metric | TFScope (−contact, ablation) | **TFScope** | DeepPBS (structure) |
|---|---|---|---|
| Mean Pearson r | 0.701 | **0.802** | 0.750 |
| Median Pearson r | 0.723 | **0.831** | 0.759 |
| IC-weighted r | 0.954 | **0.972** | 0.968 |
| Top-1 base accuracy | 0.753 | **0.836** | 0.793 |
| F1 (macro) | 0.646 | **0.765** | 0.724 |
| MCC | 0.615 | **0.765** | 0.698 |
| AUC (macro OvR) | 0.879 | **0.939** | 0.937 |
| MAE ↓ | 0.141 | **0.108** | 0.133 |
| RMSE ↓ | 0.242 | **0.184** | 0.204 |
| Cross-entropy ↓ | 1.14 | 0.99 | **0.84** |
| KL ↓ | 0.666 | 0.517 | **0.358** |

*(Values verified against `results/full_metrics/panel.json`. TFScope uses the leakage-free
leave-gene-out retrieval index; DeepPBS scored from its stored predictions under the same
trimmed-core aligner. The −contact ablation corresponds to the frozen prior branch alone.)*

These oracle-aligned numbers measure base composition with registration removed. To assess
what a method actually *emits*, we re-scored every PWM with fixed canonical registration —
trimming and canonical-strand normalization applied identically to prediction and target,
with no oracle alignment (Table 2). Here all methods lost roughly 0.43 in mean *r*,
collapsing TFScope from 0.802 to 0.420 and DeepPBS from 0.750 to 0.419 — i.e., honest
TFScope **ties** the structure-based method on the deployable metric.

**Table 2. Deployable metrics (canonical-fixed registration, no oracle alignment). 116
TFs.** «Fig. 2»

| | TFScope (−contact, ablation) | **TFScope** | DeepPBS |
|---|---|---|---|
| Mean r | 0.338 | **0.420** | 0.419 |
| Median r | 0.368 | 0.456 | 0.648 |

*(Verified against `results/canonical_reg/scores.json`: TFScope mean 0.4195, DeepPBS mean
0.4185.)*

The dominant message is sobering and method-agnostic: **registration — placing the motif
in the right frame — is the largest single error for every method, DeepPBS included.** The
honest deployable register is therefore explicitly a limitation of the current seed model:
its competitive standing on base composition does not yet translate into a competitive
*placed* PWM without an alignment oracle. As context, DeepPBS attains mean per-column *r* =
0.702 (median 0.728) on the full untrimmed 130-TF split under its native protocol
(`results/deeppbs_blind_benchmark/metrics.json`, n = 130).

## Benchmark 2 — cluster40: honest out-of-distribution generalization

Removing near-homologue shortcuts lowered absolute accuracy but preserved a coherent,
honest signal. The blind split above still admits training TFs that are close homologues of
test TFs; to control for this we built **cluster40** — CD-HIT clustering at 40% protein
identity yielding 389 clusters from 1,320 unique proteins, family-stratified into 2,983
train / 625 validation / 639 test PWMs (split sizes verified against
`data/processed/splits/cluster40/split.json`). No test TF shares >40% identity with any
training TF, so performance reflects genuine sequence→specificity transfer rather than
look-up. TFScope, trained on the cluster40 split (checkpoint `fulldata_cluster40_v18a`,
epoch 125), reached oracle-aligned per-column *r* of mean 0.535 / median 0.505 on the 639-TF
held-out set (Table 3).

**Table 3. cluster40 honest OOD panel (639 test TFs, oracle-aligned to IC ≥ 0.25 core).**
«Table 3», «Fig. 3» (`figures/pred_vs_gt_cluster40.pdf`)

| Metric | TFScope (cluster40) |
|---|---|
| Mean oracle r | 0.535 |
| Median oracle r | 0.505 |
| IC-weighted r | 0.553 |
| MAE ↓ | 0.198 |
| Top-1 accuracy | 0.630 |
| AUC | 0.797 |
| F1 | 0.603 |
| MCC | 0.472 |

*(All Table 3 values are **«VERIFY»** — sourced from the project memory record of the
checkpoint evaluation; no committed results JSON for this panel was located in `results/`
at the time of writing. The 639-TF split size and the figure are independently confirmed.)*

Per-family accuracy spanned a wide range, from bZIP and Homeodomain (mean oracle *r* ≈
0.72 and 0.68) down to the C2H2 zinc-finger subfamilies and ETS (Table 4). The bottleneck
is **C2H2_long** (181 TFs, 28% of the test set, mean *r* ≈ 0.43): long zinc-finger arrays
read DNA through per-finger recognition codes that do not collapse to a single shared family
consensus, so a family-level prior helps little.

**Table 4. cluster40 per-family mean oracle r («VERIFY», memory-sourced).**

| Family | Mean oracle r |
|---|---|
| bZIP | 0.72 |
| Homeodomain | 0.68 |
| Nuclear_Receptor | 0.66 |
| Forkhead | 0.60 |
| bHLH | 0.50 |
| C2H2_medium | 0.50 |
| Other | 0.49 |
| C2H2_long | 0.43 |
| ETS | 0.40 |
| C2H2_short | 0.39 |

We stress two leakage caveats specific to this split, because they would otherwise inflate
the headline: the same cluster40 checkpoint scores ~0.76 under an earlier training split
(`trim_ep200`) and ~0.80 on the DeepPBS 130-TF benchmark, but **both are leaky** —
the former trained on most cluster40 test TFs, and 90/130 DeepPBS test entries fall in the
cluster40 training set. The only clean cluster40 number is **0.535 on its own held-out
test set**, and only that number is used here.

## Benchmark 3 — Leave-family-out: transfer to an unseen fold

The most stringent test withheld an entire structural family at once, and it exposed how
much of the in-distribution accuracy was family memorization rather than transferable
sequence→specificity mapping. For each of the 10 families we trained on the other nine and
tested on the held-out family (validation drawn from the nine via 40%-identity clustering),
yielding 10 splits over 4,241 test TFs. The macro performance was a per-column oracle *r* of
**0.479 (mean) / 0.447 (median)** pooled across all 4,241 TFs — a value we independently
reproduced from `results/lofo/per_tf_oracle_r.json` (pooled mean 0.4786, pooled median
0.4471; the per-family means below were reproduced exactly).

The signature finding is a **collapse of variance** (Table 5). The wide in-distribution
per-family spread on cluster40 (0.39–0.72) compressed to a tight LFO band (~0.42–0.61). The
in-distribution high-flyers — bZIP, Homeodomain, Nuclear_Receptor — fell hardest (Δ = −0.20
to −0.24), indicating that their strong in-distribution scores were largely **family
memorization** of a conserved binding grammar rather than transfer; conversely, families
that were weak in-distribution (C2H2_short, ETS) held or even rose, partly because the LFO
test set includes all members of a family rather than only the clustered-out subset. The
practical reading is that **~0.48 is the model's sequence-only transfer floor to a novel
family.**

**Table 5. LFO per-family oracle r and its change versus the cluster40 in-distribution
score.** LFO column reproduced from `results/lofo/per_tf_oracle_r.json`; Δ uses the
cluster40 per-family means of Table 4. «Fig. 4»

| Family | LFO mean r | cluster40 in-dist | Δ (LFO − c40) |
|---|---|---|---|
| bZIP | 0.485 | 0.72 | −0.235 |
| Homeodomain | 0.478 | 0.68 | −0.202 |
| Nuclear_Receptor | 0.454 | 0.66 | −0.206 |
| Forkhead | 0.464 | 0.60 | −0.136 |
| bHLH | 0.508 | 0.50 | +0.008 |
| C2H2_medium | 0.514 | 0.50 | +0.014 |
| Other | 0.473 | 0.49 | −0.017 |
| C2H2_long | 0.447 | 0.43 | +0.017 |
| ETS | 0.424 | 0.40 | +0.024 |
| C2H2_short | 0.613 | 0.39 | +0.223 |

*(LFO column verified; the cluster40 column and hence Δ inherit the «VERIFY» status of
Table 4. Important caveat: the cluster40 and LFO test sets are different populations, so the
Δ mixes a genuine transfer gap with a test-set-composition effect and should be read
qualitatively, not as a clean causal delta.)*

The supporting LFO panel — IC-weighted *r* 0.490, MAE 0.223, top-1 0.589, AUC 0.734, F1
0.462, MCC 0.383 — is consistent with the macro *r* but is **«VERIFY»**: it is recorded in
the project memory from `eval_lofo.py --full-metrics`, and only the per-TF `oracle_r` array
was available as a committed file for independent recomputation.

## Synthesis

Across three benchmarks of increasing stringency, a consistent and honestly-bounded picture
emerges. (1) On base composition with registration controlled, sequence-only TFScope already
matches or exceeds the structure-based state of the art (8/11 motif-level metrics over
DeepPBS, honest LGO; Table 1) — a notable result given that DeepPBS requires a crystal
structure of the complex at inference. (2) The headline gain does not survive deployment:
registration is the dominant error for *every* method including DeepPBS (~0.43 *r* lost
aligned→fixed), and on the deployable metric TFScope ties rather than beats DeepPBS. (3) Under
genuine OOD evaluation (cluster40), accuracy settles to mean oracle *r* ≈ 0.535, and under
the hardest leave-family-out test it converges to a ~0.48 transfer floor as the
family-memorization component is stripped away. The most actionable conclusions for the
method are therefore that the next gains lie in (a) the *emitted register* (a placement/gate
head) rather than further base-composition tuning, and (b) reducing the in-distribution-to-
transfer collapse — motivating contrastive anti-collapse objectives and large-scale
protein–DNA pretraining. We report these limitations explicitly: the competitive base-
composition result is real and leakage-controlled, but it is an upper bound on the placed,
deployable PWM, and out-of-family generalization remains bounded near 0.48.

---

## Figures & tables needed

| Callout | Content | Source / status |
|---|---|---|
| «Fig. 1» | Motif-level metric bars/radar, TFScope vs −contact ablation vs DeepPBS (Table 1) | **To generate** from `results/full_metrics/panel.json`. Candidate logo overlays exist: `results/v18_compare/all_pwms_truth_v18a_deeppbs_LGO3.pdf`, `figures/pred_vs_gt_deeppbs.pdf` |
| «Fig. 2» | Aligned→fixed registration drop, all methods (Table 2) | **To generate** from `results/canonical_reg/scores.json` |
| «Fig. 3» | cluster40 predicted-vs-truth logo panel | **Exists:** `figures/pred_vs_gt_cluster40.pdf` |
| «Fig. 4» | LFO variance-collapse plot (per-family in-dist vs LFO, with Δ) | **To generate** from `results/lofo/per_tf_oracle_r.json` + Table 4 |
| «Table 1» | Motif-level metrics, 116 TFs | **Verified:** `results/full_metrics/panel.json` |
| «Table 2» | Deployable canonical-fixed metrics | **Verified:** `results/canonical_reg/scores.json` (TFScope + −contact ablation + DeepPBS) |
| «Table 3» | cluster40 honest OOD panel | **«VERIFY»** — memory-sourced; no committed results JSON found |
| «Table 4» | cluster40 per-family means | **«VERIFY»** — memory-sourced |
| «Table 5» | LFO per-family r and Δ | LFO column **verified** (`results/lofo/per_tf_oracle_r.json`); Δ inherits «VERIFY» |
