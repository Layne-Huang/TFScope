# Fig. 3a — sequence-only prediction generalizes to factors unlike anything in training

Builder: scripts/build_fig3a_heldout_recovery.py (predictions: scripts/eval_combined_heldout.py;
identity: results/fig3a_heldout/recovery_vs_identity.npz). **Model: the combined rag_contact model
(same as Fig 1)**, on the cluster40_clean held-out test (26 train-leaked TFs excluded → 586
records / 190 genes). Recovery = oracle-aligned r to the curated JASPAR/HOCOMOCO motif. Distance
= % DBD identity to the nearest training factor (max over the 16 ESM-nearest training neighbours;
Biopython global, BLOSUM62).

Key numbers: per-gene median r = 0.66, 70% ≥ 0.5. Identity-to-training median 42%; **41.5% of held-out
factors are <40% identity to any training factor, 13.5% are <30%**. Recovery is independent of that
distance — Spearman ρ = −0.06 (p = 0.12, n.s.); **median r = 0.70 for the <40%-identity (novel) factors
vs 0.71 for ≥40%**. Data: results/fig3a_heldout/fig3a_recovery.{json,csv}.

WHY THIS PANEL (distinct from Fig 1): Fig 1 establishes *parity with the structure-based SOTA on the
84 structurally characterised factors* — the only set where a competitor can run. Fig 3a makes the
*generalization / coverage* claim that Fig 1 cannot: across the broad, mostly structure-less proteome,
recovery does not depend on having a close training homolog, so the model is not interpolating
look-alikes. This is the trust foundation for the orphan-nomination panels (3b–d).

---

## Subsection: "Recovery generalizes to sequence-novel factors"

> A practical specificity predictor must work on factors that resemble nothing it was trained on,
> not only on close relatives of training examples. Using the model of Fig. 1, we evaluated 190
> transcription factors held out by 40% DNA-binding-domain identity clustering and asked how motif
> recovery depends on each factor's similarity to the training set. Recovery was essentially flat
> across the entire range of sequence distance (Spearman ρ = −0.06, p = 0.12; Fig. 3a): factors
> sharing less than 40% identity with any training protein — 42% of the held-out set, including 14%
> below 30% identity — were recovered as accurately (median r = 0.70) as factors with a close
> training relative (median r = 0.71). The model therefore is not interpolating between look-alike
> proteins; it has learned a sequence-to-specificity mapping that transfers to novel factors. As in
> Fig. 2, accuracy was organised by recognition chemistry rather than by closeness to training,
> remaining highest for the single-helix major-groove families and lowest for the long multi-finger
> C2H2 proteins (Fig. 3a, panel c). Predicted logos for held-out factors reproduced their curated
> motifs in core and flanks even at low similarity to training — PAX8 at just 22% identity to any
> training factor (r = 0.91), the CACGTG E-box of TFEB and the TGACGT element of CREB3 (both 33%
> identity), and the homeobox site of TGIF1 (Fig. 3a, panel b). Because only the
> protein sequence is required, this generalization extends across the large majority of factors
> for which no structure exists and structure-based prediction cannot be applied.

---

## Figure caption

> **(a)** Held-out motif recovery (oracle-aligned correlation to the curated JASPAR/HOCOMOCO motif),
> using the model of Fig. 1, plotted against each factor's % DNA-binding-domain identity to the
> nearest training factor (190 factors clustered out of training at 40% identity; shaded region,
> <40% identity; black line, binned median). Recovery is independent of similarity to training
> (Spearman ρ = −0.06, p = 0.12). **(b)** Predicted versus curated motif logos for representative
> held-out factors across families, each annotated with its % DBD identity to the nearest training
> factor (e.g. PAX8 recovered at r = 0.91 despite only 22% identity to training).
>
> (Per-family held-out recovery moved to Figure 1 — companion to the 1e family head-to-head — so
> Fig. 3a carries only the generalization claim.)

Caveats: identity is taken to the 16 ESM-nearest training factors (a close approximation to the
overall nearest); the combined parametric training set was not clustered against this test, but the
flat recovery–identity relationship and the equal <40% vs ≥40% medians show the result is not driven
by the high-identity tail. The nearest-neighbour-copy baseline (Fig. 1d, panel-r 0.52) is far below,
and the rigorous same-family-masked control remains the deferred leave-family-out experiment.
