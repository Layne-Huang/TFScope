# Fig. 3a — sequence-only prediction generalizes to factors unlike anything in training

Builder: scripts/build_fig3a_heldout_recovery.py (predictions: scripts/eval_combined_heldout.py;
identity: scripts/compute_identity_to_training.py). **Model: the canonical combined model
`v19_combined_fm_deeppbs_contact`, no-RAG / sequence-only — the SAME model as Fig 1 (0.643 panel-r)
and Fig 2 (mutagenesis, recognition code, structure-less).** (RAG was tested and did not help:
no-RAG ≥ RAG, so the deployed model carries no retrieval.) Evaluated on the cluster40_clean held-out
test; the 26 test TFs in the combined training split were excluded → 586 records / 190 genes.
Recovery = oracle-aligned r to the curated JASPAR/HOCOMOCO motif. Distance = % DBD identity to the
nearest training factor (max over the 16 ESM-nearest training neighbours; Biopython global, BLOSUM62).

Key numbers: per-gene median r = 0.67, 68% ≥ 0.5. Identity-to-training median 42%; **41.5% of held-out
factors are <40% identity to any training factor (13.5% < 30%)**. Recovery does NOT increase with
proximity to training — **median r = 0.74 for the novel (<40%-identity) factors vs 0.67 for ≥40%**
(Spearman ρ = −0.14, p = 0.001; the mild negative slope tracks family composition — the low-identity
bins are enriched for short, easy homeodomains — not a benefit of novelty per se). Because the model
uses no retrieval, this recovery cannot be attributed to copying a retrieved neighbour at all.
Data: results/fig3a_heldout/fig3a_recovery.{json,csv}, recovery_vs_identity.npz.

WHY THIS PANEL (distinct from Fig 1): Fig 1 establishes *parity with the structure-based SOTA on the
84 structurally characterised factors* — the only set where a competitor can run. Fig 3a makes the
*generalization / coverage* claim Fig 1 cannot: across the broad, mostly structure-less proteome, a
sequence-only model with no retrieval recovers known motifs without needing a close training homolog.
This is the trust foundation for the orphan-nomination panels (3b–d).

---

## Subsection: "Recovery generalizes to sequence-novel factors"

> A practical specificity predictor must work on factors that resemble nothing it was trained on, not
> only on close relatives of training examples. Using the model of Fig. 1 — sequence-only, with no
> retrieval — we evaluated 190 transcription factors held out by 40% DNA-binding-domain identity
> clustering and asked how motif recovery depends on each factor's similarity to the training set.
> Recovery did not improve for factors with a close training relative: factors sharing less than 40%
> identity with any training protein — 42% of the held-out set, including 14% below 30% identity —
> were recovered at least as accurately (median r = 0.74) as factors with a near relative in training
> (median r = 0.67; Fig. 3a). The model therefore is not interpolating between look-alike proteins —
> indeed, because it uses no retrieval, it cannot copy a neighbour at all — but has learned a
> sequence-to-specificity mapping that transfers to novel factors. As in Fig. 2, accuracy was
> organised by recognition chemistry rather than by closeness to training. Predicted logos for
> held-out factors reproduced their curated motifs in core and flanks even at low similarity to
> training — PAX8 at just 22% identity to any training factor (r = 0.94), the CACGTG E-box of TFEB and
> the TGACGT element of CREB3 (both 33% identity) (Fig. 3a, panel b). Because only the protein sequence
> is required, this generalization extends across the large majority of factors for which no structure
> exists and structure-based prediction cannot be applied.

---

## Figure caption

> **(a)** Held-out motif recovery (oracle-aligned correlation to the curated JASPAR/HOCOMOCO motif),
> using the sequence-only model of Fig. 1, plotted against each factor's % DNA-binding-domain identity
> to the nearest training factor (190 factors clustered out of training at 40% identity; shaded region,
> <40% identity; red lines, median recovery in each region; black line, binned median). Recovery does
> not increase with proximity to training — novel factors (median r = 0.74) recover as well as factors
> with a close training relative (0.67). **(b)** Predicted versus curated motif logos for representative
> held-out factors across families, each annotated with its % DBD identity to the nearest training
> factor (e.g. PAX8 recovered at r = 0.94 despite only 22% identity to training).

Caveats: identity is taken to the 16 ESM-nearest training factors (a close approximation to the
overall nearest); the combined parametric training set was not clustered against this test, but the
novel (<40%) factors recovering as well as the close ones shows the result is not driven by a
high-identity tail. The rigorous same-family-masked control remains the deferred leave-family-out
experiment. Per-family held-out recovery is a companion panel for Figure 1
(figures/figure1e_heldout_perfamily).
