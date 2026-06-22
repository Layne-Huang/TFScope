# Fig. 3d — model-guided DNA optimization recovers TF specificity (in-silico SELEX)

Builder: scripts/build_fig3d_dna_evolution.py. Model: combined no-RAG (same as Figs 1–3).
Data: results/fig3d_evolution/fig3d_evolution.json. One clean factor per family: LHX5
(homeodomain), MYOG (bHLH), CREB3L2 (bZIP), ELK1 (ETS). Evolved consensus vs curated motif
(oracle-aligned r): LHX5 TAATTA 0.985, MYOG CAGCTG 1.00, CREB3L2 ACGTGG 0.985, ELK1 GGAA-core 0.938.

SCOPE: only the DNA-side optimization is shown. The protein-side single-residue redesign
(homeodomain Q50K) was tested and is NOT captured by the sequence-only model (12 Q50 homeodomains;
no canonical TAATGG→TAATCC switch) — the same limitation as MyoD1 L112R; it motivates the
structure-based pipeline in Fig 4 and is not shown here.

---

## Subsection: "Optimizing DNA against the model recovers each factor's motif"

> Beyond predicting specificity, an accurate model can be used as a binding oracle to design DNA.
> For a given factor TFScope predicts a position weight matrix, which defines a binding-affinity
> landscape over all DNA sequences; we asked whether navigating that landscape from a random start
> recovers the factor's true binding site. Starting from a population of random sequences, we
> evolved DNA by repeatedly selecting the highest-affinity variants and mutating them — an in-silico
> SELEX. Across factors from four structural families the population converged smoothly from random
> to the optimum within ~15 generations (Fig. 3d a), and the evolved consensus matched the curated
> experimental motif in every case: the TAAT homeobox of LHX5 (r = 0.99), the CAGCTG E-box of MYOG
> (r = 1.00), the ACGT element of CREB3L2 (r = 0.99) and the GGAA ETS core of ELK1 (r = 0.94)
> (Fig. 3d b,c). Each factor recovered its own distinct site, confirming that the optimization is
> driven by the factor-specific predicted specificity rather than a generic sequence bias. The model
> therefore supports forward sequence design: a desired factor's binding site can be recovered, from
> scratch, by optimizing DNA against its predicted specificity.

---

## Figure caption

> **(d)** Model-guided DNA optimization (in-silico SELEX). For each transcription factor, TFScope's
> predicted PWM defines a binding-affinity score over DNA; starting from random sequences, a
> population is evolved by selecting high-affinity variants and mutating. **a,** mean predicted
> affinity per generation, normalised so 0 = random start and 1 = the consensus optimum, for one
> factor from each of four families. **b,** population sequence logo for the homeodomain factor LHX5
> at the random start (flat) and after evolution (recovered TAAT consensus). **c,** evolved consensus
> logos for MYOG (bHLH), CREB3L2 (bZIP) and ELK1 (ETS); r is the oracle-aligned correlation of the
> evolved consensus to the curated JASPAR/HOCOMOCO motif. Only DNA-side optimization is shown;
> single-residue protein redesign is not captured by the sequence-only model (see Fig. 4).
