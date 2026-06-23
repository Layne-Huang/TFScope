# Fig. 4a — sequence localizes/responds but does not resolve mutation-induced specificity switches

Builders: scripts/build_fig4a.py (figure), scripts/test_mutation_cases.py (sweep across literature
switches → results/myod1_mut/mutation_sweep.json). Combined no-RAG model (same as Figs 1–3).
Cases shown: MyoD1 L112R (bHLH basic region) and the ER↔GR P-box swap (nuclear-receptor, the textbook
3-residue specificity determinant; Umesono & Evans 1989).

## Sweep result (which switches TFScope responds to)
| mutation | family | TFScope behavior |
|---|---|---|
| MyoD1 L112R | bHLH | responds; localizes the affected E-box position; predicts A, not the true switch base G |
| ER↔GR / GR↔ER P-box swap | nuclear receptor | responds strongly (L1 0.7–1.3); but stays closer to its own WT motif (corr 0.81–0.89) than the target (0.47–0.74) — does not switch GRE↔ERE |
| homeodomain Q50K (8 factors) | homeodomain | insensitive — consensus unchanged |
| KLF4 K409Q | C2H2 | insensitive — consensus unchanged |

## Subsection text

> A practical test of a specificity model is whether it tracks the effect of mutations that are known
> to change DNA recognition. We examined two well-characterised determinants. The myogenic bHLH factor
> MyoD1 carries a basic-region substitution (L112R) that switches its E-box from CASSTG to the canonical
> CACGTG; TFScope, from sequence alone, correctly localized the change to the central E-box position
> (the largest per-position shift in its predicted motif) but assigned the wrong new base (A rather than
> G; Fig. 4a, left). The nuclear-receptor P-box — three residues that determine binding to the
> glucocorticoid versus estrogen response element — provides a second, textbook case: swapping the P-box
> between receptor classes produced a clear response in TFScope's predicted motif (per-position change
> up to 1.3), yet the prediction remained closer to the receptor's own motif than to the target class
> and did not switch GRE↔ERE (Fig. 4a, right). Across other families the model was simply insensitive to
> single-residue switches (homeodomain Q50K, the C2H2 factor KLF4 K409Q; consensus unchanged). Thus
> sequence-only prediction registers and localizes specificity-determining residues but does not resolve
> the precise new specificity of de-novo variants — a limit that the structure-based pipeline of
> Fig. 4b–c overcomes.

## Figure caption

> **(a)** Predicted motif change under specificity-switching mutations (combined model, sequence only).
> Left, MyoD1 wild-type (CACCTG) versus L112R (CACATG) with the per-position prediction change (Δ, L1);
> TFScope localizes the affected E-box position but predicts A where the experimental switch base is G.
> Right, the estrogen-receptor DBD wild-type (ERE, AGGTCA half-site) versus an estrogen→glucocorticoid
> P-box swap; the predicted motif responds (Δ up to 1.3) but does not switch to the glucocorticoid
> element. Yellow, the position of maximal predicted change.

Caveats: PWM-level, sequence-only; MyoD1 L112R is a natural single substitution, the P-box swap is the
classic engineered 3-residue determinant. The structure-based resolution of the MyoD1 switch is Fig.
4b–c. Sweep of additional mutations in results/myod1_mut/mutation_sweep.json.
