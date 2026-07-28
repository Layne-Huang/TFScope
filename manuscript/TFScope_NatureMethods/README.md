# TFScope — Nature Methods manuscript (aligned to the Fig 1–4 plan)

Self-contained. Compile with TeX Live / MacTeX:
    pdflatex TFScope_main && bibtex TFScope_main && pdflatex TFScope_main && pdflatex TFScope_main
Output: TFScope_main.pdf

Files:
  TFScope_main.tex  — manuscript (Fig 1–4 structure per TFScope_NatureMethods_PLAN.md)
  refs.bib          — references (all verified; Glasscock 2025 NSMB incl. DOI)
  figures/          — all 17 panel figures (no external paths)

Figure map (plan → file):
  Fig 1  a=architecture, b/c=benchmark, d=baseline ladder, e=per-family, f=AF3 foldability
  Fig 2  a=contact ablation, b=mutagenesis, c=recognition code, d=structure-less foldability
  Fig 3  a=held-out recovery, b/c=CRE enrichment, d=in-silico evolution, e=designed DBPs
  Fig 4  a=switch score + P-box titration, b/c=AF3+Rosetta structure calibration

Notes:
  - Fig 1a uses the chosen draft figure1a.png; replace with a vector (Illustrator) version for submission.
  - Fig 4d hybrid schematic is not yet drawn (prompt in the plan).
  - LOFO (leave-family-out) is NOT included as a result (experiment pending on the Harvard cluster).
