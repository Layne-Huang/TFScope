# Orphan-TF motif validation against public ChIP-seq (Phase-1 MVP)

Implements `manuscript/TFScope_orphan_TF_minimum_viable_validation_plan.md` (Phase 1).
Tests whether TFScope's **sequence-only** orphan-TF motifs are enriched in the **real
chromatin-binding regions** of the corresponding factors, using public ENCODE ChIP-seq —
upgrading the Fig 3b–c cCRE-localization result to occupancy-level support.

## What was run
- **TFs / data:** ADNP (ENCODE ENCSR440VKE, K562 tagged, ENCFF083UZC), ZHX2 (ENCSR407BEZ,
  HepG2, ENCFF158NBU), ZHX3 (ENCSR367KYL, HepG2 CRISPR, ENCFF982WOX) — conservative IDR
  thresholded narrowPeak, GRCh38.
- **Motifs:** the canonical combined no-RAG PWMs (`results/genome_cre_scan/pwms/*.npy`),
  frozen before analysis.
- **Pipeline** (`scripts/run_validation.py`): summit ±250 bp windows (top 8,000 peaks by
  signal) → bedtools getfasta → 20× per-peak **Eulerian dinucleotide shuffle** (exact
  dinucleotide composition preserved; `scripts/lib_shuffle.py`, self-tested) → **MOODS**
  scan (p<1e-4, both strands) → composition-controlled enrichment (log2 E, z, empirical p) +
  best-hit summit-distance density + **100× column-shuffled-PWM** null (percentile of the
  real motif). No MEME dependency.

## Result (`results/results_table.md`, `results/figures/orphan_chip_validation.png`)
| TF | log2 enrich (vs shuffle) | z | hits ≤50 bp | null-PWM %ile |
|----|---:|---:|---:|---:|
| ADNP | +0.36 | 5.5 | 0.17 | **0.98** |
| ZHX2 | +0.51 | 4.5 | 0.09 | 0.91 |
| ZHX3 | +0.50 | 3.9 | 0.15 | 0.80 |

**Interpretation (honest):**
- **All three motifs are significantly enriched** in their factor's ChIP peaks beyond local
  base composition (z = 3.9–5.5) — composition-controlled support for plausibility.
- **ADNP is the strongest / most specific** (motif beats 98% of column-shuffled PWMs); ZHX2/ZHX3
  are moderate against the null-PWM control (0.80–0.91, below the 0.95 bar).
- **No summit-centering** for any factor (best hits ≈ uniform across the 500 bp window). This is
  consistent with the known **indirect / chromatin-associated** binding of these factors (ADNP is
  a ChAHP-complex component) and with AT-rich homeodomain motifs being dispersed — it argues
  *against* over-claiming direct summit occupancy.

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
