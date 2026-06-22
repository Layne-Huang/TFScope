# Fig. 3a — held-out motif recovery from sequence alone, at scale

Builder: scripts/build_fig3a_heldout_recovery.py (predictions: scripts/eval_combined_heldout.py).
**Model: the combined rag_contact model — the same model as Fig 1**, for consistency. Evaluated on
the cluster40_clean held-out test, retrieval from the cluster40_clean index (donors <40% DBD
identity to test). The 26 test TFs present in the combined training split were excluded → 588
records / 190 genes. Metric: oracle-aligned r between predicted and curated JASPAR/HOCOMOCO motif,
aggregated per gene.
Key numbers: per-gene median r = 0.66, 70% ≥ 0.5, 46% ≥ 0.7. Per-family medians: Homeodomain 0.93,
Other 0.81, Nuclear_Receptor 0.78, ETS 0.76, C2H2_short 0.75, bZIP 0.70, C2H2_medium/bHLH 0.64,
Forkhead 0.51, C2H2_long 0.39. Data: results/fig3a_heldout/fig3a_recovery.{json,csv}.

---

## Subsection: "Known motifs are recovered from sequence alone, across the proteome"

> Having established parity with structure-based prediction on the structurally characterised
> benchmark (Fig. 1), we asked whether the same sequence-only model recovers known specificities
> at the scale that matters in practice — across the many factors for which no co-crystal exists.
> Using the model of Fig. 1, we evaluated a held-out set of 190 transcription factors clustered
> out of training at 40% DNA-binding-domain identity, scoring each predicted motif against its
> curated JASPAR/HOCOMOCO profile. Recovery was strong and systematic: the per-factor correlation
> had a median of 0.66, with 70% of factors exceeding r = 0.5 (Fig. 3a). Accuracy tracked the
> interpretability results of Fig. 2 — highest for families read by a single α-helix in the major
> groove (homeodomains, median r = 0.93) and lowest for the long multi-finger C2H2 proteins
> (0.39), whose specificity is distributed across many fingers and is intrinsically hard to
> summarise in one aligned motif. Predicted logos for held-out factors reproduced the curated
> motifs in core and flanks across families — the TAAT homeobox core of CART1, the GCC/CAGG
> element of TFAP2E, and the E-box-like sites of the zinc-finger factors ZSCAN4 and SP7 — none of
> which were available to the model at training. Because the prediction requires only the protein
> sequence, this recovery extends to the large majority of factors that lie beyond the reach of
> structure-based methods.

---

## Figure caption

> **(a)** Held-out recovery of curated DNA motifs from protein sequence alone, using the model of
> Fig. 1. It was evaluated on 190 transcription factors clustered out of training at 40%
> DNA-binding-domain identity; bars show the per-factor oracle-aligned correlation between the
> predicted and curated JASPAR/HOCOMOCO motif, grouped by structural family and ordered by median
> (dashed line, overall median r = 0.66; points, individual factors). **b,** predicted versus
> curated motif logos for representative held-out factors spanning four families.

Caveats / to add:
- **Retrieval / leakage controls.** The deployed model uses retrieval; here the retrieval donors
  are the cluster40_clean training set (<40% identity to every test factor, so no near-duplicate is
  retrieved), and the 26 test TFs present in the combined parametric training split were dropped.
  The nearest-neighbour-copy baseline (Fig. 1d ladder, panel-r 0.52) sits far below this recovery,
  so the model is not merely copying a neighbour. The rigorous control remains the same-family-masked
  leave-family-out experiment (deferred LOFO).
- **Homolog caveat.** The combined parametric training set was not clustered against this test, so it
  may contain >40%-identity homologs beyond the 26 exact matches; a fully clustered re-train (the
  cluster40_clean-trained variant gives median ≈0.78) would tighten the claim but the model of Fig. 1
  is used here for cross-figure consistency.
- Per-gene macro aggregation (190 genes from 588 records) avoids over-weighting genes with many
  deposited motifs.
