# TFScope — Nature Methods manuscript (self-contained)

## Files
- `TFScope_nature_methods.tex`  — manuscript source (latest, incl. mutation-switch + designed-DBP results)
- `TFScope_refs.bib`            — bibliography (all citations verified, incl. Glasscock 2025 NSMB)
- `figures/`                    — all 11 figures referenced by the .tex (PDF)

## Compile (TeX Live / MacTeX, any recent year)
```
pdflatex TFScope_nature_methods
bibtex   TFScope_nature_methods
pdflatex TFScope_nature_methods
pdflatex TFScope_nature_methods
```
Output: `TFScope_nature_methods.pdf`. No external files, custom styles, or absolute paths —
every `\includegraphics` points to `figures/<name>.pdf` inside this folder.

## Notes
- Uses only standard TeX Live packages (geometry, natbib, graphicx, booktabs, hyperref, lineno, etc.).
- Line numbers are on (`\linenumbers`); comment out `\usepackage{lineno}`/`\linenumbers` for a clean copy.
