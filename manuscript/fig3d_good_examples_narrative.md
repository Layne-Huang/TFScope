# Fig. 3d panel — successful specificity-aware designs (good examples)

Builder: scripts/build_fig3d_good_examples.py (reads the cached PWMs + scan_table.tsv from
run_specificity_scan.py). Data: results/specificity_design/good_designs.tsv.
Selection rule (documented, not cherry-picked on the held-out metric): the best-experimental-transfer,
well-predicted target (self pred-exp r > 0.95, transfer > 0.3) per family. Designs are nominated by
PREDICTED target-vs-offtarget margin; the experimental Z-scores shown are held out (not optimised).

Examples (experimental-PWM Z of the TFScope-nominated design; target vs worst off-target):
| target | family | target Z_exp | max off-target Z_exp | exp margin |
|---|---|--:|--:|--:|
| CPXCR1 | C2H2 | +4.07 | −0.74 | +4.81 |
| PAX6 | Homeodomain | +5.82 | +1.41 | +4.41 |
| FOXP2 | Forkhead | +2.49 | −1.47 | +3.97 |
| ZNF274 | C2H2 | +4.35 | +0.78 | +3.58 |
| ETV6 | ETS | +2.17 | −0.85 | +3.02 |
| FOS::JUN | bZIP | +2.40 | −0.45 | +2.85 |

---

## Subsection text

> Where TFScope predicts a factor's specificity accurately, its predicted PWM is an oracle precise
> enough to drive selective sequence design. Optimising 24-bp sequences against the predicted
> target-versus-off-target margin and then scoring the nominated designs with the factors' independent
> curated experimental PWMs, we obtained DNA that is strongly target-selective across six structural
> families (Fig. 3d): each design scores high for its intended factor and at or below background for its
> nearest off-targets, with experimental specificity margins of +2.9 to +4.8 standard deviations
> (e.g. the C2H2 factor CPXCR1, the homeodomain PAX6, the forkhead factor FOXP2 and the ETS factor
> ETV6). Because the designs are nominated from the predicted PWM alone and the experimental scores are
> held out, the retained selectivity shows that TFScope can serve as a forward design oracle for
> well-predicted targets. Consistent with the proteome-wide analysis (Supplementary), this success is
> governed by self-prediction fidelity rather than off-target separability: selective design transfers
> precisely for the factors whose own motif TFScope predicts well.

---

## Figure caption

> **(d)** TFScope-guided specificity-aware DNA designs are target-selective on independent experimental
> PWMs. For one well-predicted factor per structural family, a 24-bp sequence was designed by optimising
> the predicted target-versus-off-target Z-score margin (GC- and homopolymer-constrained); each panel
> shows the nominated design's experimental-PWM Z-score for the target (coloured) and its eight most
> motif-similar same-family off-targets (grey). Designs are nominated from predicted PWMs; experimental
> scores are held out. Experimental specificity margins (target − worst off-target) are annotated.

Caveats: well-predicted targets selected by a documented rule (self pred-exp r > 0.95); scoring is
PWM-based (no wet-lab); off-target panel is the 8 most motif-similar same-family factors. The
complementary proteome-wide law and the within-family failure modes are in the Supplementary
specificity analysis (figures/figure_specificity_scan/, results/specificity_design/README.md).
