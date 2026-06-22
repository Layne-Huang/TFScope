# Figure 3 caption (draft)

**Figure 3 | TFScope-predicted orphan-TF motifs are enriched in matched ChIP-seq peaks.**
**(a)** Sequence logos (information content) of the motifs predicted by TFScope from protein
sequence alone for four orphan transcription factors (SOHLH1, ADNP, ZHX2, ZHX3); sublabels give the
ChIP-seq cell line and dataset caveat. **(b)** Representative loci showing real ENCODE
fold-change-over-control ChIP-seq signal (filled track) for ADNP, ZHX2 and ZHX3; the dotted line marks
the called peak summit and the triangle/shaded box the best predicted-motif match (MOODS, P < 10⁻⁴)
with its strand. Loci were selected by a fixed rule (highest-signal peak with a top-quartile motif
score, outside ENCODE blacklist); coordinates are hg38. **(c)** Composition-controlled enrichment of
each predicted motif in its factor's peaks, expressed as log₂(observed / expected) where the
expectation is the mean over 20 per-peak dinucleotide-preserving (Eulerian) shuffles; z above each bar
is relative to the shuffle null. **(d)** Specificity against 100 column-shuffled PWMs that preserve
each motif's length, per-column composition and information content while disrupting positional order:
left, the percentile of the real motif's enrichment within the null distribution (dashed line, 0.95);
right, the null enrichment distributions (step histograms) with the real motif's enrichment as a
vertical line. All four predicted motifs are significantly enriched in their factor's ChIP-seq peaks
beyond local base composition (z = 3.0–5.5), and the real motifs out-enrich most column-shuffled PWMs
(percentile 0.77–0.93). Predicted-motif matches are not concentrated at peak summits (b), consistent
with the indirect/chromatin-associated binding of these factors. Enrichment is measured relative to
dinucleotide-preserving controls, and column-shuffled PWMs preserve motif composition and information
while disrupting positional order. These data **support occupancy-level biological plausibility of the
predicted motifs and do not prove direct TF–DNA binding**; ADNP and ZHX3 derive from tagged/CRISPR-
engineered ChIP and SOHLH1 from a low-quality ChIP experiment (573 peaks).
