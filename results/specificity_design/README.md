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
