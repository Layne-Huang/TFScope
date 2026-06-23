# Specificity-aware DNA design (TFScope_specificity_aware_DNA_design_MVP_plan.md) — RESULT: does NOT pass independent validation

Implements the plan in full (4 targets LHX5/MYOG/CREB3L2/ELK1; same-family hardest off-targets by
predicted-PWM similarity; 24-bp design with GC + homopolymer constraints; Z-normalised PWM log-odds
scores; specificity-aware GA `J = Z_t − λ·max_off Z_off − αC` with a 60%-target-Z floor to prevent
the degenerate low-everything solution; baselines random / consensus-embedding / target-only GA;
independent evaluation on curated experimental PWMs). Code: `scripts/run_specificity_design.py`,
`configs/specificity_design.yaml`. Data: `final_designs.tsv`, `off_target_selection.tsv`, `endpoint.json`.

## Outcome: FAILS the plan's predefined success criteria → method demonstration / limitation, NOT a validated result

Median specificity margin (predicted vs experimental) and experimental target binding:

| target | method | margin_pred | margin_exp | target_Z_exp |
|---|---|--:|--:|--:|
| LHX5 | consensus | −0.18 | +0.07 | 3.31 |
| LHX5 | target_only | +0.07 | +0.27 | 3.33 |
| LHX5 | **proposed** | +0.07 | **+0.27** | 3.33 |
| MYOG | target_only | 0.03 | −0.99 | 2.24 |
| MYOG | **proposed** | **+1.20** | −0.59 | **−0.62** |
| CREB3L2 | consensus | 0.05 | −0.94 | 2.17 |
| CREB3L2 | **proposed** | +0.41 | **−2.33** | −0.75 |
| ELK1 | target_only | −0.01 | −1.59 | 3.00 |
| ELK1 | **proposed** | **+0.99** | −0.50 | **+0.38** |

## Experimental-oracle upper bound → the failure is TFScope's resolution, NOT task impossibility (Case B)

Optimising directly on the curated **experimental** PWMs (an upper bound on what is achievable in PWM
space; not a TFScope result) yields large positive experimental margins for **all four** targets:

| target | oracle (exp upper bound) | TFScope-proposed | consensus | verdict |
|---|--:|--:|--:|---|
| LHX5 | **+0.83** | +0.27 | +0.07 | B |
| MYOG | **+2.43** | −0.59 | −0.99 | B |
| CREB3L2 | **+2.24** | −2.33 | −0.94 | B |
| ELK1 | **+4.40** | −0.50 | −1.59 | B |

So the selective-design task **is solvable in PWM space** even against same-family off-targets — the
curated experimental PWMs *do* contain separating fine-grained differences. The failure is therefore
**Case B**: TFScope-guided design fails because TFScope predicts the within-family off-targets as
near-identical (mean predicted-PWM corr 0.89–0.99; figure panel b) and so cannot exploit the
distinctions the experimental PWMs encode. (Caveat: the oracle optimises and is evaluated on the same
experimental PWMs, so it is an in-sample upper bound that establishes PWM-space feasibility, not
wet-lab achievability.) Figure: `figures/figure_specificity_design/specificity_design.{png,pdf}`.

**Why it fails (consistent across targets):**
1. **The optimization works on the oracle but does not transfer.** Proposed reliably raises the
   *predicted* margin (e.g. MYOG 0.03→1.20, ELK1 −0.01→0.99) but the *experimental* margin stays
   **negative** (off-target binds ≥ target) — the within-family predicted-PWM differences the GA
   exploits are largely **noise** that the curated experimental PWMs do not share. This is exactly the
   plan's criterion-5 failure ("improves only on predicted PWM, vanishes on experimental").
2. **The nominal exp-margin "wins" are degenerate.** Where proposed's exp margin beats the baselines
   (MYOG, ELK1 → 2/4), its **experimental target binding collapses** (MYOG target_Z_exp 2.24→−0.62) —
   it gains margin by partially abandoning the target, violating plan criterion #2 (retain ≥80% target
   score). LHX5: proposed merely ties target-only.
3. **Root cause is the task itself.** The "hardest" off-targets are same-family TFs with near-identical
   real motifs (predicted-PWM corr 0.92–1.00; off_target_selection.tsv) — e.g. MYOG vs MYOD1 (1.00),
   CREB3L2 vs CREB3L1/ATF6B (1.00), ELK1 vs ETV1 (1.00). When two factors read the same motif there is
   **no DNA that is selective between them**, so even the consensus has margin ≈ 0.

**Predefined criteria:** ≥3/4 proposed > consensus AND target-only → only 2/4 (and degenerate);
target exp score retained → fails; improvement not oracle-only → fails. Per the plan, this is a
**method demonstration / Supplementary** at best, not independent validation.

## Recommendation
Keep the in-silico SELEX consensus-recovery panel as the main Fig 3d (honest, works). Specificity-aware
design is either (a) a Supplementary "limitation" panel — TFScope-predicted within-family PWM
differences are not accurate/large enough to support experimentally-transferable selective design — or
(b) dropped. Do NOT present it as validated selective design.

---

## Systematic scan (no hand-picked targets) — the governing law

`scripts/run_specificity_scan.py` scans **166 targets** across 6 families (full TF set), caching
predicted+experimental PWMs; per target it computes self pred-exp corr, target/off predicted &
experimental corr, predicted-oracle separability, experimental-oracle upper bound, and the
held-out experimental transfer margin of the TFScope-guided designs. `scripts/plot_specificity_scan.py`
makes `figures/figure_specificity_scan/`; per-target table `scan_table.tsv`; case picks `case_selection.tsv`.

**Result — a continuous law, but on a different axis than hypothesised:**
- **Experimental transfer ∝ target self-prediction fidelity** (Spearman **ρ=+0.65, p=2e-21**), monotonic
  across bins; only well-predicted targets (self pred-exp r >0.95) achieve positive transfer (median
  +0.12, **61% positive**); below 0.95 transfer is reliably negative.
- **Off-target separability is NOT the driver** (Spearman ρ=−0.16) and is confounded — apparently
  "easy-to-separate" targets are mostly just *poorly predicted* (their predicted PWM is so wrong it
  looks distinct), with the worst transfer.
- The experimental-oracle margin is positive for ~95% of targets (task is feasible in PWM space
  almost everywhere); TFScope's transfer is limited by **self-prediction accuracy**, not feasibility.
- Design genuinely **works for well-predicted targets** (e.g. FOXP2 +3.7, FOS::JUN +3.1 experimental
  transfer); it fails for poorly-predicted targets and for fine-grained within-family discrimination
  (the original 4 hand-picked hard cases).

**Takeaway / usable rule:** TFScope supports specificity-aware DNA design *for targets it predicts
accurately* (a self-consistency criterion checkable a priori); fine-grained within-family selectivity
remains beyond current resolution. This is a stronger, mechanistic explanation of the Case-B failures
than any single success example.
