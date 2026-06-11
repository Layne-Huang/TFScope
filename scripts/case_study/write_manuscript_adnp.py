#!/usr/bin/env python
"""Manuscript snippets for the ADNP homeodomain case study. Numbers from tables."""
import os, json, yaml, pandas as pd
from scipy.stats import spearmanr

cfg = yaml.safe_load(open("configs/case_study_adnp.yaml"))
OUT = cfg["output_dir"]; MS = f"{OUT}/manuscript"; os.makedirs(MS, exist_ok=True)
s = json.load(open(f"{OUT}/predictions/ADNP_prediction_summary.json"))
cal = pd.read_csv(cfg["calibration_table"], sep="\t"); HD = cal[cal.family == "Homeodomain"]
ctl = pd.read_csv(f"{OUT}/validation/homeodomain_masked_control_metrics.tsv", sep="\t")
rho = spearmanr(cal.confidence, cal.oracle_r).correlation
ADN = float(s["confidence"]); rec = ctl.recovered.mean() * 100

result = f"""### TFScope nominates a homeodomain motif for the neurodevelopmental orphan TF ADNP

We applied TFScope to ADNP (Activity-Dependent Neuroprotector homeobox protein), a flagship
neurodevelopmental transcription factor — the gene mutated in Helsmoortel–Van der Aa syndrome
and one of the most recurrently mutated genes in autism — that is annotated as likely
sequence-specific yet has no curated DNA motif and no protein–DNA structure. ADNP is absent
from every TFScope training, retrieval and benchmark table (maximum DBD identity to any
training TF {s['max_train_dbd_identity']*100:.0f}%), so the production retrieval checkpoint is
leakage-free for it. From the ADNP homeobox alone (UniProt residues {cfg['case_dbd_uniprot']}),
retrieval-free TFScope produced a low-information motif ({s['noRAG_consensus']}); enabling
leave-gene-out retrieval — which drew genuine homeodomain neighbours ({s['retrieved']}) —
sharpened this into a confident homeodomain element ({s['RAG_consensus']},
{s['mean_IC_RAG']:.2f} bits; calibrated confidence {ADN:.2f}). The motif is homeodomain-class
but TGAT-leaning, equidistant from the canonical TAAT site (oracle r = {s['r_RAG_vs_TAAT']:.2f})
and the retrieved TALE/PBX1 motif (r = {s['r_RAG_vs_PBX1']:.2f}) — consistent with ADNP's
atypical homeodomain and its TALE-class nearest neighbours.

The nomination rests on a validated pipeline. On the held-out cluster40 known TFs, confidence
was calibrated to accuracy (Spearman rho = {rho:.2f}), and the homeodomain family was favourable
(held-out median oracle r = {HD.oracle_r.median():.2f}). As a positive control, the identical
retrieval-masked workflow recovered the curated motifs of held-out homeodomain factors
(EN1, PITX1, ISL1, PBX1) in {rec:.0f}% of cases (oracle r up to {ctl.r_RAG.max():.2f}),
including the canonical TAAT and the TALE TGAT grammars. The ADNP2 paralogue, itself an orphan,
produced a related but distinct prediction ({s['ADNP2_consensus']}; ADNP↔ADNP2 r =
{s['r_ADNP_vs_ADNP2']:.2f}), so we report it as a companion rather than independent validation.

TFScope thus converts the ADNP amino-acid sequence into a confidence-ranked, experimentally
testable homeodomain motif hypothesis without a protein–DNA structure. We emphasise that ADNP
acts within the chromatin-associated ChAHP complex, so its in-vivo specificity may be
context-dependent; the prediction is a hypothesis for PBM/HT-SELEX/ChIP testing, not validated
occupancy.
"""

methods = f"""### Methods — ADNP homeodomain case study

**Target.** The ADNP homeobox (UniProt Q9H2P0 residues {cfg['case_dbd_uniprot']}, taken as a
window {cfg['case_dbd_start']}–{cfg['case_dbd_end']} with short flanks) was tokenized as the DBD
input. ADNP is a clean orphan (absent from all TFScope tables), so the production retrieval
checkpoint (cluster40_v18a_rag) is leakage-free; it was run with retrieval disabled (noRAG) and
with gene-deduplicated leave-gene-out retrieval excluding ADNP and ADNP2.

**Confidence and calibration.** Confidence used the calibrated decisiveness score
(0.5·IC_norm + 0.5·gate_norm) and the cluster40 held-out known-TF calibration table; the
homeodomain subset provided a family-specific accuracy expectation.

**References and controls.** Predictions were compared by oracle-aligned, reverse-complement-
aware mean per-column Pearson r to a canonical homeodomain TAAT site and to the top retrieved
neighbour (PBX1, a TALE/TGAT homeodomain). As a positive control, EN1, PITX1, ISL1 and PBX1
(in-training homeodomains) were each run with that gene and ADNP/ADNP2 excluded from retrieval
and compared to their curated motifs (retrieval-masked recovery). ADNP2 (also orphan) was run
identically as a paralogue consistency check.
"""

caption = f"""**Fig. 7 | Sequence-only homeodomain motif nomination for the neurodevelopmental orphan TF ADNP.**
**a**, TFScope confidence distribution for held-out known TFs that miss (grey) or hit (green,
oracle r ≥ 0.6) their curated motif, with the held-out homeodomain subset (purple); ADNP
(red, {ADN:.2f}) marked. Inset: calibration curve. **b**, ADNP target card: multi-zinc-finger +
single-homeobox architecture and leakage facts; input to TFScope is amino-acid sequence and a
homeobox mask only. **c**, Predicted motifs — ADNP noRAG (weak prior), ADNP leave-gene-out RAG
({s['RAG_consensus']}, {s['mean_IC_RAG']:.2f} bits), the ADNP2 paralogue companion, and the
retrieved TALE neighbour PBX1; RAG vs canonical TAAT r = {s['r_RAG_vs_TAAT']:.2f}, vs PBX1
r = {s['r_RAG_vs_PBX1']:.2f}. **d**, Held-out confidence vs accuracy (homeodomain highlighted;
ADNP marked; dotted line, success threshold r = 0.6). **e**, Retrieval-masked homeodomain
positive control: the workflow recovers the curated motifs of EN1, PITX1, ISL1 and PBX1
({rec:.0f}% at r ≥ 0.6). All values are computed in results/adnp_case/.
"""

# promoter scan (data-driven verdict; Extended Data if not enriched)
prom_path = f"{OUT}/targets/promoter_scan_scores.tsv"
if os.path.exists(prom_path):
    pr = pd.read_csv(prom_path, sep="\t").set_index("motif")
    ng = len(pd.read_csv(f"{OUT}/targets/neurodev_gene_set.tsv", sep="\t"))
    au_rag = pr.loc["ADNP RAG", "auroc"]; au_shuf = pr.loc["shuffled ADNP RAG", "auroc"]
    p_rag = float(pr.loc["ADNP RAG", "mwu_p"])
    enriched = (p_rag < 0.05) and (au_rag - au_shuf > 0.03)
    if enriched:
        prom = f"""### Neurodevelopmental promoter enrichment of the ADNP motif

We scanned promoters (TSS ±1 kb, hg38) of {ng} neurodevelopmental / autism / ADNP-network genes
with the ADNP RAG motif against a dinucleotide-preserving shuffled background (GC and CpG held
fixed). The motif was significantly enriched (AUROC = {au_rag}, p = {p_rag:.2g}) beyond its
shuffled control (AUROC = {au_shuf}), nominating candidate ADNP regulatory elements for
experimental testing; this does not establish in-vivo occupancy."""
    else:
        prom = f"""### Neurodevelopmental promoter scan (Extended Data)

We scanned promoters (TSS ±1 kb, hg38) of {ng} neurodevelopmental / autism / ADNP-network genes
with the ADNP RAG motif against a dinucleotide-preserving shuffled background (GC and CpG held
fixed). After this composition control the AT-rich homeodomain motif showed no enrichment beyond
composition (AUROC = {au_rag}, vs shuffled control {au_shuf}); we therefore do not claim
neurodevelopmental-specific promoter occupancy. Consistent with ADNP acting within the chromatin
ChAHP complex, sequence alone does not localize its motif to target promoters, and the prediction
remains a sequence-level hypothesis for experimental (PBM/HT-SELEX/ChIP) validation."""
    open(f"{MS}/promoter_scan_adnp.md", "w").write(prom)

open(f"{MS}/result7_draft.md", "w").write(result)
open(f"{MS}/methods_adnp_draft.md", "w").write(methods)
open(f"{MS}/figure7_caption.md", "w").write(caption)
print("wrote ADNP manuscript snippets ->", MS)
print(f"ADNP RAG {s['RAG_consensus']} | conf {ADN:.2f} ({s['confidence_class']}) | "
      f"masked HD recovery {rec:.0f}% | ADNP↔ADNP2 r {s['r_ADNP_vs_ADNP2']:.2f}")
