# Fig. 3d — Methods (model-guided DNA design)

Scripts: scripts/build_fig3d_dna_evolution.py (SELEX), scripts/run_specificity_design.py +
scripts/run_specificity_scan.py (specificity design + scan), scripts/build_fig3d_good_examples.py,
scripts/build_fig3d_composite.py. All use the combined no-RAG model (v19_combined_fm_deeppbs_contact),
the same model as Figs 1–3. Config: configs/specificity_design.yaml.

## Predicted and experimental PWMs
For each transcription factor, TFScope was run on its DNA-binding domain (sequence only, no
retrieval) to obtain a predicted position weight matrix (the gated motif core). The independent
"experimental" PWM is the factor's curated JASPAR/HOCOMOCO motif from the training corpus, trimmed to
its informative core (per-column information > 0.25 bits). Experimental PWMs were used only for
evaluation and never during optimisation.

## In-silico SELEX (Fig. 3d a)
A factor's predicted PWM defines a binding score over DNA, S(s) = Σ_j log P_j(s_j). Starting from a
population of 400 random sequences, each generation scores all sequences, retains the top 50%, and
refills by point mutation; this was run for 24 generations from a fixed seed. Convergence is reported
as the population-mean score rescaled so 0 is the random start and 1 the consensus optimum. The evolved
population consensus was compared to the curated experimental motif by oracle-aligned correlation.

## Specificity-aware design and scoring (Fig. 3d b)
Designs are 24-bp double-stranded sequences over {A,C,G,T} constrained to 35–65% GC and no homopolymer
run > 3 nt; invalid sequences are rejected. For factor k and sequence s, the score is the maximum PWM
log-odds over all windows and both strands, S_k(s) = max_{window,strand} Σ_j log[(P_{k,j}+ε)/P_bg]
with P_bg = 0.25 and ε = 10⁻³. Scores are standardised per factor against 20,000–50,000 random
GC-constrained background sequences, Z_k(s) = (S_k − μ_k)/σ_k, with μ_k, σ_k computed separately for
predicted and experimental PWMs. For a target t with off-target set O_t, the specificity margin is
M_t(s) = Z_t(s) − max_{o∈O_t} Z_o(s). Off-targets are the eight most motif-similar same-family factors,
selected by maximum predicted-PWM correlation (over shift and strand). A genetic algorithm (population
≥400, ≥35 generations, 5% elite, 20% parents, 4–5% per-base mutation, 30% crossover, multiple seeds)
maximised J = Z_t − λ·max_o Z_o subject to a target-binding floor (Z_t ≥ 60% of the consensus Z_t) that
prevents the degenerate low-affinity margin solution; λ = 1. The design TFScope nominates is the
highest predicted-margin valid sequence; its specificity is then evaluated with the held-out
experimental PWMs. Baselines: random constrained sequences, consensus-embedding (target consensus
embedded at the best position/strand), and a target-only GA (J = Z_t).

## Independent evaluation, oracle bounds, and the proteome-wide law (Fig. 3d c)
The primary read-out is the experimental specificity margin M_t^exp of the nominated designs. Two
oracle bounds contextualise it: a predicted-oracle (the margin a GA reaches on predicted PWMs, i.e.
how separable TFScope thinks the factors are) and an experimental-oracle (the margin a GA reaches
optimising directly on the experimental PWMs — an in-sample upper bound on PWM-space feasibility, not a
TFScope result). Across 375 factors from nine structural families (scripts/run_specificity_scan.py;
no hand-picked targets), the experimental transfer margin of TFScope-guided designs was governed by
the target's self-prediction fidelity — the correlation between its predicted and experimental PWMs
(Spearman ρ = 0.55, p = 3×10⁻³¹) — and was essentially independent of off-target separability
(Spearman ≈ 0). Selective design transferred to the experimental PWMs only for well-predicted targets
(self-prediction r > 0.95: 53% with positive margin), exemplified across six families in Fig. 3d b
(experimental margins +2.9 to +4.8). The experimental-oracle margin was positive for ~95% of factors,
so failures reflect TFScope's within-family resolution, not task infeasibility.

## Statement of scope
These analyses are PWM-based and computational; they support specificity-aware *forward sequence
design for factors TFScope predicts accurately* and do not constitute experimental validation of
selective binding. Fine-grained discrimination among same-family factors with near-identical predicted
motifs remains beyond current resolution (Supplementary).
