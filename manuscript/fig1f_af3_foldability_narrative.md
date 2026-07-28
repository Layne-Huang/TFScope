# Fig. 1f — AF3 foldability head-to-head (TFScope vs DeepPBS predicted consensus)

Builder: scripts/build_fig1f_af3_foldability.py. Source: AF3 paired folds at
/data1/leihuang/project/TFScope/AF3_consensus_folding (84 TFs × {TFScope, DeepPBS}
consensus, replicate runs). Key numbers: n=41 TFs, TFScope mean ipTM 0.795 vs DeepPBS
0.744 (Δ +0.051), TFScope higher in 30/41 (73%), Wilcoxon p=5.0e-3; length-controlled
(Δ–length rho=-0.14, p=0.38; same-length subset Δ=+0.054, 7/10).

SCOPE: this is the planned Fig 1f — a TFScope-vs-DeepPBS *foldability* comparison on the
84 cluster40 *test* TFs (which have experimental structures as the GT reference; inputs in
manuscript/AF3_consensus_folding_inputs.md). It is an orthogonal, structure-based validation
that does not depend on the oracle-aligned PWM metric. It is NOT the structure-less set
(LHX5/MYOG/… in results/structureless_af3_inputs/), which was prepared for a separate
generalization panel and is disjoint from these benchmark TFs.

---

## Subsection: "An independent structure predictor favours TFScope's predicted motifs"

> A predicted motif is only useful if it represents a DNA sequence the transcription
> factor can actually engage. To test this without reusing our own benchmark, we asked an
> orthogonal, structure-based referee — AlphaFold3 — to evaluate the motifs. For each
> factor we folded its DNA-binding domain together with the consensus DNA predicted by
> TFScope and, separately, with the consensus predicted by the structure-based method
> DeepPBS, and compared the interface confidence (ipTM) of the two resulting protein–DNA
> complexes. Across 41 transcription factors, AlphaFold3 assigned a higher interface
> confidence to the TFScope-predicted complex in 30 cases (73%; mean ipTM 0.795 versus
> 0.744; Wilcoxon signed-rank p = 5 × 10⁻³; Fig. 1f). The advantage was not an artefact of
> DNA length — although DeepPBS consensus oligomers were uniformly 14 bp and TFScope's were
> on average shorter, the per-factor ipTM gap was uncorrelated with the length difference
> (Spearman ρ = −0.14, p = 0.38) and was unchanged (Δ ipTM = +0.05, 7 of 10 factors) when
> restricted to factors whose two oligomers were the same length. The largest gains
> appeared for factors whose specificity DeepPBS captures poorly, including HNF4A, SRY, E2F4
> and DUX4. That a structure predictor trained independently of either method prefers the
> sequence-only predictions indicates that TFScope's motifs are not merely accurate against
> curated databases but correspond to DNA sequences that form more confident physical
> complexes with their factors.

---

## Figure caption

> **(f)** AlphaFold3 interface confidence (ipTM) for protein–DNA complexes folded from each
> method's predicted consensus motif (84 factors, replicate folds; n = 41 with paired
> completed folds). **a,** per-factor ipTM of the TFScope-consensus complex versus the
> DeepPBS-consensus complex; points above the diagonal (shaded) are factors whose TFScope
> motif folds more confidently; coloured by structural family; the four largest gains
> labelled. **b,** paired ipTM with medians (bars); TFScope's predicted motifs fold more
> confidently in 30 of 41 factors (Wilcoxon p = 5 × 10⁻³). The effect is length-controlled
> (Δ = +0.05 ipTM on the 10 same-length factor pairs).
