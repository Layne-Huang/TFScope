# Progress Log: TFScope — Nature Methods manuscript

**Started:** 2026-08-13
**Status:** v1 draft complete (grounded results wired; placeholders for fill-later)
**Target:** Nature Methods (Article), sequence-only TF specificity

## Timeline
- ✅ Phase 0 CONFIG: Paper Configuration Record (Nature Methods, methods/comp-bio+ML, Nature citation style, LaTeX)
- ✅ Grounding: pulled exact numbers from results/ + figures_v24_ensemble/ (no fabricated stats)
- ✅ Phase 2/3 ARCHITECTURE+ARGUMENT: IMRaD + Nature Methods structure, claim→evidence per figure
- ✅ Phase 4 DRAFTING: full v1 draft (abstract, intro, 7 results subsections, discussion, methods, declarations, 9 figures)
- ✅ Phase 5a partial: references.bib with 16 high-confidence foundational refs; compiles, 0 undefined citations
- ✅ Compiled: v1_draft.pdf (10 pages), image-based PDF review passed (clean layout, figures render)
- ⏳ Phase 1 literature search (broad related-work) — NOT run (offline); \TODO in discussion
- ⏳ Phase 5a full citation verification (DOIs/metadata) — PENDING (marked \TODO in .bib)
- ⏳ Phase 5b bilingual abstract — not added (Nature Methods English-only; can add if wanted)
- ⏳ Phase 6 peer review — pending

## Grounded numbers used
- Ensemble PanelA content-r 0.664 (single-seed 0.629); mean gate-oracle-r TFScope 0.657 vs DeepPBS 0.626 (Δ+0.031, p_boot=0.30)
- Gene-disjoint 20-gene: DeepPBS-retrain 0.720 vs v24 0.685 (tie); pretrained 0.806 leaky
- Foldability Boltz-2: 0.957 vs 0.923, 32/41, Wilcoxon p=1e-5, pLDDT 0.948 vs 0.921
- Contact probe AUROC 0.946 / AUPRC 0.781 (2324 complexes); family kNN purity 0.83, probe 0.96
- Family/within-family 41%/59% (p=2e-4); MyoD1 L122R Δ_switch +9.17; DBP6 r=0.65
- Mutation-blindness: pred Δ0.005 vs observed 0.180 (limitation, stated honestly)

## Open placeholders (1 TODO, 5 PLACEHOLDER)
- Author list / affiliations / correspondence / funding / CRediT / data+code availability / AI-use statement
- Fig 1a/b architecture schematic (not drawn); Fig 4 benchmark-bar+ablation-ladder panel (not assembled)
- Methods exactness: layer list, LoRA targets, MoE routing, head equations, data counts, metric equations, stats details
- Related-work positioning (needs literature search); Boltz-2 citation (currently Boltz-1 placeholder); DeepPBS DOI
- Within-family decomposition method paragraph; switch-score definition

## Files
- drafts/v1_draft.tex, v1_draft.pdf (10 pp)
- references/references.bib (16 refs; verification pending)
- figures/ (8 ensemble figures copied)
