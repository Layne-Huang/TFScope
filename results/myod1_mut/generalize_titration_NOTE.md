# Cross-family titration — exploratory, CONFOUNDED (do not use as a clean generalization)

scripts/generalize_titration.py picks, per family, the MOST motif-divergent well-predicted TF pair
and titrates a recognition-module swap. Result (generalize_titration.json): the clean graded
"resolution scales with determinant size" titration does NOT generalize to these pairs, because the
most-divergent within-family pairs are different STRUCTURAL SUBTYPES / different lengths
(NR4A2 monomeric orphan vs NR3C2; POU3F1 vs NKX2-5; MyoD1 vs NPAS4 bHLH-PAS), so the alignment-based
swap is not a smooth path and cannot reconstruct the target (indels) — corr-to-target only partially
rises (0.4-0.6) and is non-monotonic; the C2H2 pair only "resolves" at 100% (chimera ≈ target = trivial).

CLEAN, DEFENSIBLE titration = closely-related paralogs differing in a LOCALIZED determinant:
nuclear receptors GR<->ER / AR->ER (P-box -> full module), which rise monotonically 0.34 -> 0.98 and
cross the resolved threshold at 50-75% (results/myod1_mut/multimutant_titration.json,
figures/figure4a_titration/). These remain the canonical Fig 4a titration cases.
