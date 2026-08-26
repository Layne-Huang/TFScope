# Decision: 2-D PWM-column × residue supervision is DISABLED in v26

**Date:** 2026-08-14 · **Status:** settled negative result · **Gates:** Phase-5 stage C

## Summary

The empirical 2-D contact map (which PWM *column* each contacting residue reads) cannot be
recovered reliably enough to use as supervision. The 1-D contact labels (which residues touch DNA
at all) are unaffected and remain sound.

## Evidence

Alignment places the target PWM onto the crystallised oligo by scanning offsets and both
orientations. Two scoring versions were tested; the second added IC-weighting and a runner-up
margin specifically to fix the first's weak result.

| test | value | null | ratio |
|---|---|---|---|
| IC of contacted columns | 1.243 bits | 1.117 | **1.11** |
| PWM prob of observed base | 0.370 | 0.25 | 1.48 |
| consensus-base match rate | **0.387** | 0.25 | 1.55 |
| match on high-IC columns (>1.5 bits) | 0.458 | 0.25 | 1.83 |
| match on low-IC columns (<0.5 bits) | 0.294 | 0.25 | 1.18 |

Signal is real (p ≈ 1e-297, and stronger exactly where it should be — high-IC columns beat low-IC
columns), but the absolute level is far below what a correct alignment implies: a 2-bit column is
near-deterministic, so match should approach 0.85–0.9, not 0.46.

## Why filtering cannot rescue it

Per-alignment match rate (n=1,430 alignments with ≥4 contacted bases) is **unimodal, centred at
0.4–0.6**, not bimodal:

```
0.0-0.2: 102   0.2-0.4: 511   0.4-0.6: 742   0.6-0.8: 58   0.8-1.0: 14
only 2.7% of alignments reach match >= 0.7
```

So there is no good subset hiding inside a bad average. Confirmed by two independent filters:

- **runner-up margin**: corr with match = 0.114. Sweeping 0.0 → 1.0 leaves IC enrichment flat at
  1.11–1.13 while halving retention. Alignment ambiguity is therefore *not* the problem.
- **alignment score**: corr with match = 0.378 (Spearman 0.416). Thresholding moves match only
  0.426 → 0.445 while discarding 56% of the data.

## Interpretation

The likely cause is not computational. The target PWM comes from in-vitro selection
(CIS-BP / HOCOMOCO / JASPAR consensus), while the crystal contains one specific designed
oligonucleotide — frequently a variant site, a half-site, or a different spacing. A
position-for-position correspondence between "base in this crystal" and "column of that PWM" may
simply not exist for a large fraction of structures.

## Consequences

1. **v26 Phase-5 stage C trains the 1-D contact loss only.** `lambda_contact2D = 0`.
2. The 2-D artifacts are retained (`contacts2d_core.parquet`, with `align_score` / `align_margin`
   per contact) for interpretability analysis and possible future rescue, but are not a loss term.
3. **This retroactively questions v24.** v24 ran `contact_distill_weight = 0.2` on 2-D targets
   derived the same way (`scripts/contact_teacher/build_contact_targets.py`, offset+orientation
   match to the PWM consensus). On this evidence that channel was likely injecting near-noise.
   The v26 ablation should therefore include a v24-style 2-D-on run to measure what it cost.
4. A genuine fix would need per-structure ground truth for which site was crystallised (e.g.
   parsing the deposited oligo annotation and matching it to the motif independently), not a
   better scoring function.

## Reproduce

```bash
scripts/v26/run_detached.sh --mirror fixcols scripts/v26/run_fix_columns.sh
# -> results/v26/pwm_column_alignment_diagnostic.json  (margin sweep)
```
