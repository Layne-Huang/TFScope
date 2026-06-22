# Orphan-TF motif validation against public ChIP-seq (Phase-1 MVP)

Implements `manuscript/TFScope_orphan_TF_minimum_viable_validation_plan.md` (Phase 1).
Tests whether TFScope's **sequence-only** orphan-TF motifs are enriched in the **real
chromatin-binding regions** of the corresponding factors, using public ENCODE ChIP-seq —
upgrading the Fig 3b–c cCRE-localization result to occupancy-level support.

## What was run
- **TFs / data:** ADNP (ENCODE ENCSR440VKE, K562 tagged, ENCFF083UZC), ZHX2 (ENCSR407BEZ,
  HepG2, ENCFF158NBU), ZHX3 (ENCSR367KYL, HepG2 CRISPR, ENCFF982WOX) — conservative IDR
  thresholded narrowPeak, GRCh38; **SOHLH1** (GEO GSE280248, "uncharacterized human TFs binding
  genomic dark matter", HEK293, GSM8589942 narrowPeak hg38, 573 peaks — note: GEO flags this ChIP
  "insufficient peak numbers", so it is a low-confidence experiment).
- **ZGLP1: no usable occupancy data** — 0 ENCODE experiments (any assay); GEO keyword hits collide
  with the unrelated GLP-1 hormone. Stays on the cCRE-enrichment evidence (Fig 3b–c); proper test
  is Phase-3 (germ-cell ATAC / perturbation transcriptomics). ADNP2 likewise has 0 ENCODE ChIP-seq.
- **Motifs:** the canonical combined no-RAG PWMs (`results/genome_cre_scan/pwms/*.npy`),
  frozen before analysis.
- **Pipeline** (`scripts/run_validation.py`): summit ±250 bp windows (top 8,000 peaks by
  signal) → bedtools getfasta → 20× per-peak **Eulerian dinucleotide shuffle** (exact
  dinucleotide composition preserved; `scripts/lib_shuffle.py`, self-tested) → **MOODS**
  scan (p<1e-4, both strands) → composition-controlled enrichment (log2 E, z, empirical p) +
  best-hit summit-distance density + **100× column-shuffled-PWM** null (percentile of the
  real motif). No MEME dependency.

## Result (`results/results_table.md`, `results/figures/orphan_chip_validation.png`)
| TF | cell | peaks | log2 enrich | z | hits ≤50 bp | null-PWM %ile |
|----|------|------:|---:|---:|---:|---:|
| SOHLH1 | HEK293 (low-qual) | 567 | +0.40 | 3.0 | **0.21** | 0.87 |
| ADNP | K562 (tagged) | 8000 | +0.35 | 5.5 | 0.17 | 0.93 |
| ZHX2 | HepG2 | 8000 | +0.44 | 4.4 | 0.09 | 0.92 |
| ZHX3 | HepG2 (CRISPR) | 4678 | +0.47 | 3.9 | 0.15 | 0.77 |

**Interpretation (honest):**
- **All four motifs are significantly enriched** in their factor's ChIP peaks beyond local base
  composition (z = 3.0–5.5) — composition-controlled support for plausibility.
- **SOHLH1's E-box shows the best summit-centering** (0.21, ≈ uniform-to-central) — consistent with
  a specific, direct-binding bHLH motif — though on a low-confidence ChIP (567 peaks).
- **The AT-rich homeodomain motifs (ADNP/ZHX2/ZHX3) are enriched but NOT summit-centered** (best
  hits ≈ uniform across the 500 bp window), consistent with their **indirect / chromatin-associated**
  binding (ADNP is a ChAHP-complex component) and dispersed AT-rich motifs — it argues *against*
  over-claiming direct summit occupancy.
- Against the null-PWM control, ADNP/ZHX2 are most specific (0.92–0.93); ZHX3/SOHLH1 moderate.
  (Null-PWM percentile is mildly run-order-dependent through the shared RNG; ±0.05.)

**Conclusion:** public TF-specific occupancy data support the biological plausibility of the
predicted motifs (enrichment in real bound regions), strongest for ADNP; they do **not** prove
direct TF–DNA binding, and the absence of summit-centering is reported transparently.

## Caveats / next phases
- emp-p floored at 1/(20+1)=0.048 by 20 shuffles → **z is the discriminator** (a 100-shuffle run
  gives exact p). 
- ADNP and ZHX3 are tagged/engineered systems → occupancy support, not native-state proof.
- **Phase 2 (not yet run):** GC/length-matched genomic negatives → AUROC/AUPRC; JASPAR/HOCOMOCO
  family-matched motif controls; 3×3 cross-TF specificity matrix.
- **Phase 3:** ZHX2 MCF-7 cross-cell-line replication; ZHX2 knockdown RNA-seq; mouse ADNP.

## Files
`config.yaml` (frozen params) · `scripts/{lib_shuffle,run_validation,make_figure}.py` ·
`data/raw_peaks/*.bed` · `results/enrichment/<TF>.json` · `results/orphan_chip_validation.json`
· `results/results_table.md` · `results/figures/orphan_chip_validation.{png,pdf}`
