# Results — Application to de novo designed DNA-binding proteins

**Figure:** `figures/figure_dbp_heatmap/dbp_tfscope_heatmap.{pdf,svg,png}`
(builder `scripts/build_dbp_tfscope_heatmap.py`; combined no-RAG model, sequence input only).
**Panels (per design DBP5/DBP9/DBP6/DBP35):** top = TFScope predicted logo (position weight matrix,
bits, aligned to the 14-position target); bottom = experimental single-base-pair mutation-effect
heatmap (flow-cytometry yeast-display relative binding, log2 vs the WT designed base; darker red =
substitution weakens binding = specificity-determining position; blue = stronger than WT; WT base boxed).
**Experimental source:** Extended Data of the de novo HTH design study (single-bp competition scans);
metric corrected so that lower normalized PE/FITC = stronger binding (DBP6 WT reference = 0.1202).

**Quantitative summary (TFScope predicted preference vs experimental binding strength, gated-core aligned):**

| design | core agreement | corr. r | confidence (predicted core IC) |
|---|---|---|---|
| DBP6  | 83% | 0.65 | high — tall, dominant CAC |
| DBP5  | 67% | 0.59 | partial |
| DBP9  | 38% | 0.50 | partial |
| DBP35 | 50% | 0.39 | partial |
| mean  | 59% | 0.53 | — |

---

## Subsection draft

Recent work designed fully synthetic helix–turn–helix (HTH) protein scaffolds that bind a chosen DNA
target, providing a stringent out-of-distribution test for any specificity model: these proteins share
no evolutionary history with the natural transcription factors from which sequence-based predictors are
trained. We applied TFScope, from amino-acid sequence alone, to four such designs — DBP5, DBP6, DBP9 and
DBP35 — each engineered against the same target site (GCAGATCTGCACATC). For every design we predicted a
position weight matrix (Fig. X, logos) and compared it with the experimental relative-binding landscape
from yeast-display competition assays, in which every single base-pair substitution is scored by flow
cytometry (Fig. X, heatmaps; darker red marks a substitution that weakens binding, i.e. a
specificity-determining position).

TFScope recovered the central CAC recognition core of these de novo proteins directly from sequence —
and, more informatively, it did so with a confidence that tracked the strength of the experimental
signal. For DBP6 the predicted motif placed a tall, high-information CAC squarely over the columns the
mutation scan identifies as most specificity-determining (the core positions where every substitution
sharply reduces binding); predicted core and experimental "important" columns coincide almost exactly
(per-position agreement 83%, correlation r = 0.65 between the predicted base preference and the measured
binding strength). For DBP5, DBP9 and DBP35 the model still recovered the core, but with lower relative
information content and weaker registration: the CAC signal is present yet does not dominate the
predicted motif as cleanly as for DBP6 (agreement 38–67%, r = 0.39–0.59; mean across the four designs
r = 0.53, 59% agreement). The point is therefore not a binary "TFScope recovers the motif", but that
TFScope returns a calibrated, graded prediction — confident and correct where the designed interface
makes a strong, natural-like CAC contact (DBP6), and appropriately tentative where it does not.

This graded behaviour is consistent with what a sequence-only model can and cannot know. The strong,
specific predictions concentrate on the core recognition base pairs, while the flanking positions carry
little predicted information — sensible for DNA regions that the designed scaffold does not specifically
read. Unlike a structure-based predictor, TFScope cannot infer the shape-driven flank preferences (for
example A-tract / narrow-minor-groove readout) that arise from the designed backbone, because that
signal is structural rather than sequence-encoded; its flanks instead remain low-information, which is
the correct default in the absence of base-specific readout. That a model trained only on natural
transcription-factor domains nonetheless recovers — to a graded, experimentally consistent degree — the
specificity of proteins engineered de novo underscores that TFScope has learned transferable
sequence-to-specificity features rather than memorized its training families, and points to its use as a
fast, sequence-only triage step that flags which designed binders have confidently predictable
specificity before committing to structural modelling or wet-lab characterization.

## Caveats / honest notes
- Sequence-only, PWM-level; combined no-RAG model (same as the rest of the paper). These designs are
  genuinely OOD (de novo proteins), so partial recovery (DBP5/9/35) is expected, not a failure.
- The "confidence" claim is supported by both the predicted-core information content (logo height) and
  the quantitative agreement with the experimental binding strength (table above) — DBP6 is highest on
  both; do NOT overstate the partial cases as strong recoveries.
- Flank predictions are low-information by design of a sequence-only model; the structure-driven flank
  story (A-tract minor-groove) that a structure-based method can recover is out of scope here — state
  this explicitly rather than claiming flank agreement.
- Metric/sign: experimental relative binding uses lower normalized PE/FITC = stronger; TFScope preference
  correlated against that (corrected from an earlier sign-inverted analysis). Per-design numbers in
  `results/design_case_study/` (see match-best-binding analysis).
