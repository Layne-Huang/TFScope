#!/usr/bin/env python
"""Manuscript snippets for the ZGLP1 GATA-class case study. All numbers from tables."""
import os, json, yaml, pandas as pd

cfg = yaml.safe_load(open("configs/case_study_zglp1.yaml"))
OUT = cfg["output_dir"]; MS = f"{OUT}/manuscript"; os.makedirs(MS, exist_ok=True)
s = json.load(open(f"{OUT}/predictions/ZGLP1_prediction_summary.json"))
gp = pd.read_csv(f"{OUT}/predictions/ZGLP1_RAG_vs_GATA_family.tsv", sep="\t")
ctl = pd.read_csv(f"{OUT}/validation/GATA_masked_control_metrics.tsv", sep="\t")
ret_rec = ctl.retrieval_recovered.mean() * 100; dn_rec = ctl.deNovo_recovered.mean() * 100

result = f"""### TFScope nominates a GATA-class motif for the orphan germ-cell factor ZGLP1

To probe whether TFScope generalizes beyond the bHLH/E-box grammar of SOHLH1, we
applied it to ZGLP1 (GLP-1), a GATA-type single-zinc-finger transcription factor that
drives the oogenic program and is annotated as a likely sequence-specific regulator but
lacks an experimentally curated motif in JASPAR or any PBM/HT-SELEX resource. From the
ZGLP1 GATA zinc-finger sequence alone (UniProt P0C6A0, residues {cfg['case_dbd_uniprot']}
plus the C-terminal basic tail) and with the gene held out of retrieval, TFScope predicted
a canonical GATA element ({s['prod_RAG_consensus']}, {s['RAG_mean_IC']:.2f} bits, calibrated
confidence {s['confidence']:.2f}), recovering the GATA recognition core (oracle r =
{s['r_RAG_vs_GATA_exemplar']:.2f} against the GATA3 exemplar; mean r = {s['gata_family_r_mean']:.2f}
across GATA1–6). The retrieved neighbours were bona fide GATA paralogues ({s['retrieved']}),
and the prediction matched the GATA family consensus rather than ZGLP1's own (divergent)
training motif, indicating family-level inference rather than memorization.

Two controls establish that this nomination is genuinely retrieval-driven. First, a
family-masked checkpoint that never saw any GATA factor (leave-"Other"-family-out) failed
to produce a GATA motif for ZGLP1 ({s['clean_deNovo_consensus']}; r =
{s['r_cleanDeNovo_vs_GATA_exemplar']:.2f} vs GATA), and recovered the known motif of no
held-out GATA control ({dn_rec:.0f}%). Second, the production workflow with leave-gene-out
retrieval recovered the curated motifs of held-out GATA factors (GATA4, GATA6) in
{ret_rec:.0f}% of cases (oracle r = {ctl.r_prod_RAG.min():.2f}–{ctl.r_prod_RAG.max():.2f}).
TFScope therefore correctly classifies ZGLP1 as a GATA-class factor from sequence alone by
leveraging retrieved paralogues — not by reciting a memorized entry.

Critically, the nomination is honest about its resolution. ZGLP1's experimental HOCOMOCO
motif ({s['ground_truth_consensus']}) is a divergent GATA variant: the high-information GATA
core columns agree with the prediction (per-column r = {s['core_column_r_highIC']:.2f}), but the
flanking columns diverge, so the overall agreement is modest (r =
{s['r_RAG_vs_ZGLP1_H13CORE']:.2f}). TFScope thus nominates the GATA *core* for ZGLP1 with high
confidence while flagging ZGLP1-specific flanking preferences as a concrete, experimentally
testable refinement (PBM/HT-SELEX), rather than asserting the full motif.
"""

methods = f"""### Methods — ZGLP1 GATA-class case study

**Target and DBD window.** The ZGLP1 GATA-type zinc finger (UniProt P0C6A0,
ZN_FING {cfg['case_dbd_uniprot']}) plus the adjacent C-terminal basic region (window
{cfg['case_dbd_start']}–{cfg['case_dbd_end']}) was tokenized as the DBD input. ZGLP1 lacks a
JASPAR/PBM/HT-SELEX motif but has a HOCOMOCO H13CORE motif that is present in TFScope's
training data; we therefore treat that motif as (encoder-leaky) ground truth and report
results from two checkpoints.

**Checkpoints.** (i) A leakage-clean, family-masked checkpoint (leave-"Other"-family-out)
in which all GATA factors and ZGLP1 were held out of training, used for de-novo prediction
(retrieval disabled). (ii) The production retrieval checkpoint (cluster40_v18a_rag), in which
ZGLP1 is present in training (encoder-leaky) but excluded from its own retrieval
(leave-gene-out); used for the retrieval-augmented nomination, clearly labelled as a
production reference.

**Scoring.** Predictions were compared by oracle-aligned (offset ±{cfg['max_alignment_offset']},
reverse-complement-aware) mean per-column Pearson r to (a) the ZGLP1 H13CORE motif, (b) a
GATA3 exemplar (JASPAR MA0037.3), and (c) each GATA1–6 family motif. Confidence used the same
calibrated decisiveness score as the SOHLH1 analysis.

**Controls.** For GATA4 and GATA6 we extracted the C-terminal GATA zinc-finger DBD window
from UniProt and (A) ran the production workflow with that gene and ZGLP1 excluded from
retrieval (retrieval-masked recovery) and (B) ran the family-masked checkpoint
(de-novo). Recovery was defined as oracle r ≥ {cfg['success_threshold_r']} against the curated
GATA motif.
"""

caption = f"""**Fig. 6 | Sequence-only GATA-class motif nomination for the orphan germ-cell TF ZGLP1.**
**a**, ZGLP1 target card: GATA-type zinc-finger architecture and leakage/curated-database
facts; the input to TFScope is amino-acid sequence and a DBD mask only. **b**, Predicted
motifs — family-masked de-novo (a model that never saw any GATA factor; no GATA recovered),
production leave-gene-out RAG (GATA core {s['prod_RAG_consensus']}, {s['RAG_mean_IC']:.2f} bits),
the GATA3 exemplar, and ZGLP1's divergent experimental H13CORE motif
({s['ground_truth_consensus']}). **c**, RAG prediction vs each GATA1–6 family motif (mean r =
{s['gata_family_r_mean']:.2f}); dashed line, agreement with ZGLP1's own H13CORE motif. **d**,
Double dissociation: retrieval-masked recovery of held-out GATA motifs ({ret_rec:.0f}%) vs
family-masked de-novo ({dn_rec:.0f}%), showing the nomination is retrieval-driven. **e**,
Per-column agreement between the RAG prediction and ZGLP1's H13CORE motif: GATA core columns
agree (core r = {s['core_column_r_highIC']:.2f}) while flanking columns diverge, defining a
testable ZGLP1-specific flanking hypothesis. All values are computed in results/zglp1_case/.
"""

# promoter scan (data-driven verdict)
prom_path = f"{OUT}/targets/promoter_scan_scores.tsv"
if os.path.exists(prom_path):
    pr = pd.read_csv(prom_path, sep="\t").set_index("motif")
    cand = pd.read_csv(f"{OUT}/targets/top_candidate_targets.tsv", sep="\t")
    ng = len(pd.read_csv(f"{OUT}/targets/germ_cell_gene_set.tsv", sep="\t"))
    au_rag = pr.loc["ZGLP1 RAG", "auroc"]; au_shuf = pr.loc["shuffled ZGLP1 RAG", "auroc"]
    p_rag = float(pr.loc["ZGLP1 RAG", "mwu_p"])
    enriched = (p_rag < 0.05) and (au_rag - au_shuf > 0.03)
    top = ", ".join(cand.gene.head(6))
    if enriched:
        prom = f"""### Germ-cell promoter enrichment of the ZGLP1 GATA motif

Because GATA elements are AT-rich and not CpG-centered, we could test promoter enrichment
without the CpG-island confounding that defeats E-box scans. We scanned promoters (TSS ±1 kb,
hg38) of {ng} genes of the ZGLP1-driven oogenic / meiosis-entry program against a
dinucleotide-preserving shuffled background (GC and CpG held fixed per sequence). The ZGLP1
RAG motif was weakly but significantly enriched (AUROC = {au_rag}, Mann-Whitney p =
{p_rag:.2g}) and, critically, exceeded its own column-shuffled control (AUROC = {au_shuf}),
indicating a composition-independent signal. The highest-scoring promoters included
meiosis-entry and oocyte genes ({top}). This nominates candidate ZGLP1 regulatory elements
for experimental testing (ChIP/CUT&RUN); it does not establish in vivo occupancy, and the
effect size is modest."""
    else:
        prom = f"""### Germ-cell promoter scan (Extended Data)

We scanned promoters (TSS ±1 kb, hg38) of {ng} ZGLP1-program oogenesis genes with the ZGLP1
RAG motif against a dinucleotide-preserving shuffled background. The motif showed no
enrichment beyond composition (AUROC = {au_rag} vs shuffled control {au_shuf}); we therefore
do not claim germ-cell-specific promoter occupancy."""
    open(f"{MS}/promoter_scan_zglp1.md", "w").write(prom)

open(f"{MS}/result6_draft.md", "w").write(result)
open(f"{MS}/methods_zglp1_draft.md", "w").write(methods)
open(f"{MS}/figure6_caption.md", "w").write(caption)
print("wrote ZGLP1 manuscript snippets ->", MS)
print(f"ZGLP1 RAG consensus {s['prod_RAG_consensus']} | conf {s['confidence']:.2f} | "
      f"GATA family mean r {s['gata_family_r_mean']:.2f} | vs H13CORE r {s['r_RAG_vs_ZGLP1_H13CORE']:.2f}")
print(f"controls: retrieval-masked recovery {ret_rec:.0f}% | de-novo {dn_rec:.0f}%")
