# Figure 3 — orphan-TF motif validation against matched ChIP-seq

Fully reproducible, **all values from real repository / public data** — no synthetic motifs,
coordinates, scores, or null distributions.


## Reproducing panel b on a fresh checkout
The ChIP bigWigs (~2.2 GB) are NOT in the repo. One command fetches them (stable accession URLs):
```bash
bash scripts/download_figure3_chip_bigwigs.sh   # -> /data1/leihuang/chip_bigwigs/
```
Provenance/accessions: `results/figure3/bigwig_manifest.tsv`. Then run the plot script.

## Regenerate
```bash
python scripts/plot_figure3_orphan_tf_validation.py      # reads configs/figure3_orphan_tf_validation.yaml
```
Outputs: `figure3_orphan_tf_validation.{pdf,svg,png}` (PNG 600 dpi; PDF Type42 + SVG fonttype=none →
editable vector text), `figure3_plot_data.tsv` (QC table + representative-peak rows).

## Panels and their real data sources
- **a — predicted motifs.** Logos from `results/genome_cre_scan/pwms/{SOHLH1,ADNP,ZHX2,ZHX3}.npy`
  (TFScope combined no-RAG PWMs), information-content scaled, canonical base colours. Sublabels show
  the real ChIP context (cell line · caveat) from the config metadata.
- **b — representative ChIP peaks.** Real ENCODE **fold-change-over-control bigWig** signal at one
  auto-selected locus per TF (ADNP/ZHX2/ZHX3). Dotted line = real peak **summit** (narrowPeak col 10);
  ▲ + shaded box = best **MOODS** motif hit (p<1e-4) with strand; coordinates are real (hg38).
  *Selection rule (documented, not cherry-picked):* among the top-3000-signal peaks on standard
  chromosomes, not overlapping the ENCODE hg38 blacklist, take the highest-signal peak whose best
  motif score is ≥ the 75th percentile of motif scores. Selected peak IDs/coords in
  `figure3_plot_data.tsv`. The motif hits are **not** summit-centred (ADNP at the locus right edge,
  ZHX3 at the left) — shown honestly, consistent with indirect/chromatin-associated binding.
- **c — composition-controlled enrichment.** `log2_enrich` and `z` per TF from
  `orphan_tf_validation/results/enrichment/<TF>.json` (real motif hits in summit±250 peaks vs **20×
  per-peak Eulerian dinucleotide shuffle**, all peaks).
- **d — specificity vs column-shuffled-PWM nulls.** Left: percentile of the real motif within the
  null distribution; right: the **100** column-shuffled-PWM enrichments (`null_enrich`) with the real
  motif (`real_enrich_sub`) as a vertical line. Percentile and empirical-p are **recomputed in-script**
  from the stored null arrays so the displayed labels are internally consistent:
  `percentile = #(null<real)/N`, `p_emp = (1+#(null≥real))/(N+1)`.

## QC table (`figure3_plot_data.tsv`)
| TF | motif_len | n_peaks | log2 enrich | z | n_null | percentile | p_emp(nullPWM) | cell | caveat |
|----|--:|--:|--:|--:|--:|--:|--:|--|--|
| SOHLH1 | 10 | 567 | +0.40 | 3.0 | 100 | 0.87 | 0.139 | HEK293 | low-quality ChIP |
| ADNP | 9 | 8000 | +0.35 | 5.5 | 100 | 0.93 | 0.079 | K562 | tagged |
| ZHX2 | 9 | 8000 | +0.44 | 4.4 | 100 | 0.92 | 0.089 | HepG2 | — |
| ZHX3 | 8 | 4678 | +0.47 | 3.9 | 100 | 0.77 | 0.238 | HepG2 | CRISPR |

## Important caveats (built into the figure / caption)
1. **Panels c and d use different enrichment computations** — c = all peaks vs dinucleotide shuffle;
   d = 2000-peak subset vs column-shuffled PWMs — so the c-bar and the d-line are not the same number.
2. The dinucleotide empirical p is floored at 1/(20+1)=0.048 by 20 shuffles → **z is the discriminator**.
3. ADNP and ZHX3 are tagged/CRISPR-engineered ChIP; SOHLH1 is a GEO experiment flagged low-quality
   (573 peaks). Interpret as **occupancy-level plausibility**, not proof of direct TF–DNA binding.
4. ZGLP1/ADNP2 have no usable public TF ChIP-seq (0 ENCODE experiments) and are not in this figure.

## Inputs (real files)
motifs `results/genome_cre_scan/pwms/*.npy` · peaks `orphan_tf_validation/data/raw_peaks/*.bed`
(ENCODE IDR / GEO narrowPeak) · enrichment+null `orphan_tf_validation/results/enrichment/*.json` ·
bigWigs `/data1/leihuang/chip_bigwigs/{ADNP,ZHX2,ZHX3}.fc.bigWig` (ENCODE fold-change) ·
blacklist `hg38-blacklist.v2` · genome `/data1/leihuang/WholeGenomeFasta/genome.fa`.
