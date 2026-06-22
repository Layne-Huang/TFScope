# Fig. 3a — held-out motif recovery from sequence alone, at scale

Builder: scripts/build_fig3a_heldout_recovery.py. Source: e5b (deployed retrieval-augmented
model, family-register) predictions on the cluster40_clean held-out test
(results/v19_e9_model_composition/e5b_test_predictions.npz). Metric: oracle-aligned r between
predicted and curated JASPAR/HOCOMOCO motif, aggregated per gene.
Key numbers: 193 held-out factors (40%-DBD-identity clustered out of training), per-gene median
r = 0.78, 82% ≥ 0.5, 57% ≥ 0.7. Per-family medians: Homeodomain 0.99, Forkhead 0.89,
C2H2_medium 0.84, ETS 0.84, bHLH 0.81, C2H2_short 0.78, bZIP 0.77, Nuclear_Receptor 0.73,
C2H2_long 0.66, Other 0.64. Data: results/fig3a_heldout/fig3a_recovery.{json,csv}.

---

## Subsection: "Known motifs are recovered from sequence alone, across the proteome"

> Having established parity with structure-based prediction on the structurally characterised
> benchmark (Fig. 1), we asked whether the sequence-only model recovers known specificities at
> the scale that matters in practice — across the many factors for which no co-crystal exists.
> We evaluated the deployed model on a held-out set of 193 transcription factors clustered out
> of training at 40% DNA-binding-domain identity, scoring the predicted motif against its
> curated JASPAR/HOCOMOCO profile. Recovery was strong and systematic: the per-factor
> correlation had a median of 0.78, with 82% of factors exceeding r = 0.5 and 57% exceeding
> 0.7 (Fig. 3a). Accuracy tracked the interpretability results of Fig. 2 — recovery was highest
> for families with a single α-helical major-groove reader (homeodomains, median r = 0.99;
> forkhead, 0.89) and lowest for the heterogeneous "Other" class and the long multi-finger
> C2H2 proteins (0.64–0.66), whose specificity is distributed across many fingers and is
> intrinsically harder to summarise in a single aligned motif. Predicted logos for held-out
> factors reproduced the curated motifs in both core and flanks across families — the
> GGGG-rich element of RREB1, the TAAT homeobox core, the GC-box of ZNF213 and the CACGTG
> E-box of USF3 (Fig. 3a, right) — none of which were available to the model at training.
> Because the prediction requires only the protein sequence, this recovery extends to the
> large majority of factors that lie beyond the reach of structure-based methods.

---

## Figure caption

> **(a)** Held-out recovery of curated DNA motifs from protein sequence alone. The deployed
> model was evaluated on 193 transcription factors clustered out of training at 40%
> DNA-binding-domain identity; bars show the per-factor correlation (oracle-aligned) between the
> predicted and curated JASPAR/HOCOMOCO motif, grouped by structural family and ordered by
> median (dashed line, overall median r = 0.78; points, individual factors). Right, predicted
> versus curated motif logos for representative held-out factors spanning four families.

Caveats / to add: (i) the deployed model uses retrieval from the training set; the
nearest-neighbour-copy baseline (Fig. 1d ladder, panel-r 0.52) is far below this recovery,
showing the model does not merely copy a neighbour, but a same-family-masked (leave-family-out)
control is the rigorous test and is the deferred LOFO analysis. (ii) per-gene macro aggregation
(193 genes from 600 records) avoids over-weighting genes with many deposited motifs.
