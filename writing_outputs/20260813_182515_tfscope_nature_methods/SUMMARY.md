# TFScope — Nature Methods manuscript (v1 draft)

**What this is:** a complete, compilable v1 draft of the TFScope Nature Methods
Article, grounded in the repo's actual results, with every unfinished item
clearly marked `\TODO{}` (red) or `\PLACEHOLDER{}` (orange) so nothing is
silently invented.

## Deliverables
- `drafts/v1_draft.tex` + `v1_draft.pdf` — 10 pages, compiles clean (0 undefined citations)
- `references/references.bib` — 16 foundational refs (DOI/metadata verification pending)
- `figures/` — 8 ensemble figures (1c logos, 1e per-family, foldability, 2b mutagenesis, 2c recognition PyMOL, 3a recovery, 4a switch, DBP heatmap)
- `progress.md` — phase log + grounded-number ledger

## Structure (Nature Methods Article)
Abstract → Introduction → Results (model; leakage-controlled parity with DeepPBS;
independent Boltz-2 foldability; contact-residue + family probes; within-family
decomposition; case studies; limitations) → Discussion → Methods → Declarations →
Figures → References.

## Headline claim (fully grounded)
Sequence-only TFScope matches structure-based DeepPBS under strict leakage
control (mean gate-oracle r 0.657 vs 0.626, p_boot=0.30; ensemble content-r 0.664),
and its motifs fold more confidently by an independent predictor
(Boltz-2 ipTM 0.957 vs 0.923, 32/41, p=1e-5).

## To fill later (1 TODO + 6 PLACEHOLDER)
1. Front matter: authors, affiliations, correspondence, funding, CRediT, data/code availability, AI-use statement
2. Fig 1a/b architecture schematic; Fig 4 benchmark-bar + ablation-ladder panel
3. Methods exactness (layers, LoRA targets, MoE routing, head equations, data counts, metric equations, statistics)
4. Literature/related-work positioning (needs a lit-search pass) + Boltz-2 & DeepPBS citation metadata
5. Within-family decomposition method; switch-score definition

## Next steps (skill phases still available)
- Phase 1 lit-review (`ars-lit-review`) to populate related work
- Phase 5a `citation-check` to verify every DOI
- Phase 6 peer review (`academic-paper-reviewer`) once placeholders are filled

## Build
```
cd drafts && PATH=/data1/leihuang/texlive/2026/bin/x86_64-linux:$PATH \
  pdflatex v1_draft && bibtex v1_draft && pdflatex v1_draft && pdflatex v1_draft
```
