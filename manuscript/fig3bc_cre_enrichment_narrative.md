# Fig. 3b–c — nominated orphan-TF motifs are enriched in cis-regulatory elements

Builder: scripts/build_fig3bc_cre_enrichment.py. Data: results/genome_cre_scan/ (motifs from the
canonical combined no-RAG model v19_combined_fm_deeppbs_contact; MOODS scan of hg38, p<1e-4;
ENCODE cCRE promoters/enhancers; dinucleotide-shuffle composition control). Six orphan TFs:
SOHLH1 (bHLH), ADNP/ADNP2/ZHX2/ZHX3 (homeodomain), ZGLP1 (GATA-like).
Key numbers (composition-controlled vs shuffle): ADNP2 17.5× promoter (z=19.6), ZHX3 3.96× (z=8.1),
ADNP 3.37× (z=9.6), ZHX2 1.74× (z=3.1) in promoters; SOHLH1 E-box 1.38× in enhancers (z=7.2);
ZGLP1 depleted (0.78×) — 5/6 enriched.

---

## Subsection: "Nominated motifs localize to cis-regulatory elements"

> A motif predicted for an uncharacterised factor is more credible if its genomic occurrences are
> not scattered at random but concentrated where regulatory elements lie. We nominated motifs from
> sequence alone for six orphan transcription factors with little or no experimental specificity —
> the bHLH factor SOHLH1, the homeodomain factors ADNP, ADNP2, ZHX2 and ZHX3, and the GATA-like
> factor ZGLP1 — scanned each motif across the human genome, and measured its hit density in ENCODE
> candidate cis-regulatory elements relative to two baselines. Against the whole genome the
> AT-rich homeodomain motifs appeared paradoxically *depleted* in regulatory elements (Fig. 3b); this
> is a base-composition artifact, because cis-regulatory elements are GC-rich while these motifs are
> AT-rich. Controlling for composition with a dinucleotide-preserving shuffle of the elements
> themselves removed the artifact and reversed the conclusion: five of the six motifs were
> significantly enriched in cis-regulatory elements (Fig. 3c). The enrichment was organised by
> recognition chemistry — the homeodomain motifs were concentrated in promoters, most strongly for
> ADNP2 (17.5-fold, z = 19.6) and at 3–4 fold for ADNP and ZHX3, while the SOHLH1 E-box was enriched
> in enhancers (1.4-fold, z = 7.2), consistent with the promoter-proximal versus distal logic of the
> two element classes. The single exception, ZGLP1, remained depleted, reflecting a low-information
> nominated motif (see limitations). These localization signals indicate that the sequence-only
> nominations are functionally plausible candidate motifs, not random k-mers; they are a statement
> about where the motifs fall, not a measurement of factor occupancy.

---

## Figure caption

> **(b–c)** Genomic localization of TFScope-nominated orphan-TF motifs. **(a)** Motifs predicted from
> sequence alone (combined no-RAG model) for six orphan factors. **(b)** Naive enrichment of
> genome-wide motif hits in ENCODE cCRE promoters and enhancers relative to the whole genome (log2):
> AT-rich homeodomain motifs appear depleted, a base-composition artifact. **(c)** The same enrichment
> against a dinucleotide-shuffle composition control of the elements: the artifact is removed and 5/6
> motifs are enriched (asterisk, z > 2), homeodomains in promoters (ADNP2 17.5-fold) and the SOHLH1
> E-box in enhancers.

Limitations (honest): (i) candidate-level *plausibility*, not occupancy — no ChIP. (ii) Several
nominated motifs are low-information; ZGLP1's no-RAG motif (GCAAAA) is the model's weaker call (the
validated nomination is GATA-like), and it is the one depleted case. (iii) Empirical p is floored by
the shuffle count, so z-scores are the discriminator (100-shuffle run for exact p). Motif correctness
is validated separately (paralog r, same-family-masked controls).
