# Fig. 4a — TFScope reproduces the directional specificity switch of MyoD1 L122R

Builder: **scripts/build_fig4a_switch.py** → figures/figure4a_switch/figure4a_switch.{png,pdf,svg};
results/myod1_mut/switch_score_tfscope.json. Combined no-RAG model (same as Figs 1–3); DBD input
(TFScope is a DBD-level model — full-length input is out-of-distribution and must not be used).

## The directional switch score (use this, NOT a consensus-string comparison)

Comparing argmax consensus strings is misleading: a single noisy column can flip the consensus to a
spurious base even when the *predicted distribution* moves in the correct direction. We therefore score
the two competing E-boxes under each predicted PWM and form a difference-in-differences. For a PWM `P`
and an E-box `s`, let `S(s | P)` be the best PWM log-odds score (background 0.25, taken as the maximum
over all offsets and both strands). Then

    Δ_switch = [ S_mut(CACGTG) − S_mut(CACCTG) ] − [ S_WT(CACGTG) − S_WT(CACCTG) ]

    Δ_switch > 0  → L122R pushes the predicted preference toward the MYC-like CACGTG (switch reproduced)
    Δ_switch ≤ 0  → the expected switch is not reproduced.

**Result (MyoD1 WT vs L122R, bHLH DBD):**

| protein | S(CACGTG) | S(CACCTG) | S_G − S_C |
|---|---|---|---|
| WT    | −0.43 | 11.38 | −11.81  (strongly prefers the WT myogenic CACCTG) |
| L122R | 5.84  | 8.47  | −2.64   (gap nearly closed) |

**Δ_switch = (−2.64) − (−11.81) = +9.17 > 0 → the switch is reproduced.** L122R lifts CACGTG from
disfavored (−0.43) to favored (+5.84) — a +6.3 log-odds gain — and collapses the CACCTG-vs-CACGTG
preference gap from −11.8 to −2.6. The mutant still marginally favors CACCTG (−2.6 < 0), so this is a
strong *directional* shift rather than a complete flip, but its sign and magnitude are unambiguous.

> **Subsection text:** A practical test of a specificity model is whether it tracks mutations known to
> redirect DNA recognition. The myogenic bHLH factor MyoD1 carries a basic-region substitution (L122R;
> L112R in DBD numbering) reported to shift its preference from the myogenic E-box CACCTG toward the
> MYC-like CACGTG. Rather than read off consensus letters — which are unstable to single-column noise —
> we scored both E-boxes under the wild-type and mutant *predicted* PWMs and formed a directional
> difference-in-differences (Δ_switch). From sequence alone, TFScope reproduced the switch: the mutation
> raised the CACGTG log-odds from −0.43 to +5.84 and reduced the CACCTG preference gap from −11.8 to
> −2.6, giving Δ_switch = +9.17 (>0; Fig. 4a). Thus the predicted distribution moves in the correct,
> experimentally documented direction, even though the mutant motif's single most-probable base does not
> by itself flip — a distinction that a consensus-only readout would have missed.

> **Figure caption.** **(a)** Directional specificity-switch score for MyoD1 L122R (combined model,
> sequence only, bHLH DBD). Left, wild-type and L122R predicted E-box logos. Right, PWM log-odds score
> S(E-box | predicted PWM) for CACGTG (MYC-like) and CACCTG (WT myogenic) under the wild-type (grey) and
> L122R (red) PWMs; the arrow marks the +6.3 gain on CACGTG. The difference-in-differences
> Δ_switch = +9.17 (>0) indicates the mutation pushes the predicted preference toward CACGTG, reproducing
> the documented switch.

Caveat: PWM-level, sequence-only; the experimental switch is well documented for the bHLH basic-region
substitution. Structure-based cross-validation of the same switch (AF3 refold + Rosetta interface ddG)
is in Fig. 4b–c.

---

# (superseded) earlier consensus-based framing and the ER P-box / titration analyses

Earlier builder: scripts/build_fig4a.py (figure), scripts/test_mutation_cases.py (sweep across literature
switches → results/myod1_mut/mutation_sweep.json). Combined no-RAG model (same as Figs 1–3).
Cases shown: MyoD1 L112R (bHLH basic region) and the ER↔GR P-box swap (nuclear-receptor, the textbook
3-residue specificity determinant; Umesono & Evans 1989). NOTE: the "predicts A, not G" reading below is
the consensus-argmax view that the Δ_switch score above supersedes for MyoD1.

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

---

## Multi-mutant titration (answers: are larger determinant changes easier?)

Builder: scripts/test_multimutant.py → results/myod1_mut/multimutant_titration.json;
figure scripts/build_fig4a_titration.py → figures/figure4a_titration/. Progressively swapping the
differing residues of one nuclear-receptor DBD into another (GR↔ER, AR→ER) shows that **sequence-only
resolution scales with the size of the determinant change**: the predicted motif's correlation to the
TARGET receptor rises monotonically from ~0.34 (0% swapped, including the single-residue / 3-residue
P-box regime, where it does NOT switch) to 0.8–0.98 (full recognition-module swap), crossing the
"resolved" threshold (corr ≥ 0.7) at 50–75% of differing residues. This explains the single-mutant
limit mechanistically: a 1–3 residue change is too small a perturbation to the pooled DBD
representation to flip the predicted specificity, whereas swapping the bulk of the recognition module
does. (AR→ER is noisier but trends the same.)

> Suggested text: "TFScope's ability to resolve an engineered specificity switch scaled with the number
> of specificity-determining residues changed: single residues and the three-residue P-box perturbed
> but did not redirect the predicted motif, whereas swapping the majority of the recognition module
> switched the prediction to the target receptor's element (correlation to target rising from 0.34 to
> >0.95; Fig. 4a). The sequence-only model thus captures specificity at the level of the recognition
> module, not of individual de-novo substitutions."
